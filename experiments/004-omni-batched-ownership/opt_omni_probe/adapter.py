"""SPIKE: a vLLM-Omni EngineAdapter half for experiment 004.

The same job as experiment 003's SGLang adapter, in a second engine: split one
DiT call into one owner per batch row, give each owner its own state, run the
*unchanged* experiment 002 policy per owner, and reassemble the rows. The
policy and the payload forms are imported from experiment 002.

Where vLLM-Omni keeps what the adapter needs (68003cf6a, vllm_omni/diffusion/):
  - row owners are known only to the model runner, never inside the
    transformer (`forward_context.py` has no per-row ids):
      request mode  `_execute_request_list(reqs)`: one request per row group,
                    all rows at the same step;
      step mode     `_prepare_batch_inputs` returns the `InputBatch`, whose
                    `request_ids` and `states` give each row its request and
                    its own `step_index` (`worker/input_batch.py:594`).
    So the adapter records the row owners at the runner and reads them in the
    trunk hooks of the very next DiT call.
  - per-owner state lives here, keyed by request id and CFG branch, and is
    dropped when the runner retires the request
    (`_cleanup_finished_step_requests`, or the end of `_execute_request_list`).
  - the row axis is dim 0 at both trunk edges.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import time

import torch

from opt_trunk_probe import policy
from opt_trunk_probe.adapter import apply, capture

_RUNNER = "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner"

_rows = None  # [(owner, engine_step or None)] for the next DiT call(s)
_owners = {}  # owner -> {"policy": ..., branch: state, "calls": {branch: n}}
_current = None  # the open TrunkCall


def _control(owner):
    """The policy whose key is part of `owner`; vLLM-Omni may decorate ids."""
    path = os.environ.get("OPT_OMNI_CONTROL")
    if owner is None or not path or not os.path.exists(path):
        return {"mode": "observe"}
    with open(path) as f:
        table = json.load(f)
    hits = [key for key in table if key in owner]
    return dict(table[max(hits, key=len)], key=max(hits, key=len)) if hits else {"mode": "observe"}


def _log(record):
    path = os.environ.get("OPT_OMNI_LOG")
    if path:
        record.update(pid=os.getpid(), wall=time.time())
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def _meta(t):
    if t is None:
        return None
    return dict(shape=list(t.shape), dtype=str(t.dtype).removeprefix("torch."))


def _resolve(path):
    module, _, attr = path.rpartition(".")
    try:
        return importlib.import_module(module), attr
    except ModuleNotFoundError:
        owner, _, cls = module.rpartition(".")
        return getattr(importlib.import_module(owner), cls), attr


def wrap(path, hook):
    """Replace a class attribute with `hook(original, self, *args, **kwargs)`."""
    holder, attr = _resolve(path)
    original = getattr(holder, attr)

    def wrapped(self, *args, **kwargs):
        return hook(original, self, *args, **kwargs)

    setattr(holder, attr, wrapped)


# --- runner side: who owns which rows ---------------------------------------


def around_request_list(original, self, reqs, *args, **kwargs):
    """Request mode: every row group is one request, all at the same step."""
    global _rows
    _rows = []
    for req in reqs:
        n = max(1, int(getattr(req.sampling_params, "num_outputs_per_prompt", 1) or 1))
        _rows += [(req.request_id, None)] * n
    try:
        return original(self, reqs, *args, **kwargs)
    finally:
        _rows = None
        for req in reqs:
            _retire(req.request_id)


def around_prepare_batch_inputs(original, self, *args, **kwargs):
    """Step mode: the InputBatch names each row's request and its own step."""
    global _rows
    states, input_batch, outputs = original(self, *args, **kwargs)
    _rows = None
    if input_batch is not None:
        _rows = []
        for rid, state in zip(input_batch.request_ids, input_batch.states or states):
            _rows += [(rid, int(state.step_index))] * int(state.latents.shape[0])
    return states, input_batch, outputs


def around_cleanup(original, self, scheduler_output, *args, **kwargs):
    for rid in scheduler_output.finished_req_ids or ():
        _retire(rid)
    return original(self, scheduler_output, *args, **kwargs)


def _retire(owner):
    if _owners.pop(owner, None) is not None:
        _log(dict(event="retire", owner=owner, live_owners=sorted(_owners)))


# --- model side: one trunk call, split into owners ---------------------------


class ItemCall:
    """One row of one trunk invocation: the `call` the policy sees."""

    def __init__(self, row, owner, step_index, engine_step, num_steps, branch, overridable, slot):
        self.row, self.owner = row, owner
        self.step_index, self.engine_step, self.num_steps = step_index, engine_step, num_steps
        self.branch, self.overridable = branch, overridable
        self.policy = slot["policy"]
        self.state = slot.setdefault(branch, {})
        self.entry = self.signal = self.override = None
        self.log = {}

    def record(self, **fields):
        self.log.update(fields)


class TrunkCall:
    def __init__(self, binding, model, overridable, timestep):
        self.binding, self.model = binding, model
        self.branch = "cond"  # CFG is off in every plan; see README
        self.overridable = overridable
        self.timestep = timestep
        self.rows_known = list(_rows) if _rows is not None else None
        self.items = None
        self.rows = None
        self.decision = None
        self.blocks_run = self.blocks_skipped = 0

    def open(self, entry, signal):
        self.rows = entry.shape[0]
        aligned = self.rows_known is not None and len(self.rows_known) == self.rows
        owners = self.rows_known if aligned else [(None, None)] * self.rows
        self.items = []
        for row, (owner, engine_step) in enumerate(owners):
            slot = _owners.setdefault(owner, {"policy": _control(owner), "calls": {}}) if owner \
                else {"policy": {"mode": "observe"}, "calls": {}}
            # Request mode gives no step index; count this owner's own calls.
            counted = slot["calls"].get(self.branch, 0)
            slot["calls"][self.branch] = counted + 1
            step = engine_step if engine_step is not None else counted
            num_steps = slot["policy"].get("num_steps")
            item = ItemCall(row, owner, step, engine_step, num_steps, self.branch,
                            self.overridable and aligned, slot)
            item.record(counted_step=counted)
            item.entry = entry[row : row + 1]
            item.signal = None if signal is None else signal[row : row + 1]
            item.override = policy.on_enter(item, item.policy, item.state)
            self.items.append(item)
        wants = [item.override is not None for item in self.items]
        self.decision = "skip" if all(wants) else "splice" if any(wants) else "compute"

    def close(self, x):
        pieces, replaced = [], False
        for item in self.items:
            xi = x[item.row : item.row + 1]
            if item.override is not None:
                pieces.append(apply(item.entry, item.override))
                item.record(exit_from="override")
                replaced = True
                continue

            def pack(kind, entry=item.entry, out=xi):
                return capture(kind, entry, out)

            payload = policy.on_exit(item, item.policy, item.state, pack)
            if payload is None:
                pieces.append(xi)
                continue
            rebuilt = apply(item.entry, payload)
            item.record(exit_from=f"identity:{payload[0]}",
                        identity_exact=bool((rebuilt == xi).all()))
            pieces.append(rebuilt)
            replaced = True
        return torch.cat(pieces, dim=0) if replaced else x


def around_dit(binding):
    def hook(original, self, *args, **kwargs):
        global _current
        timestep = kwargs.get("timestep")
        call = TrunkCall(binding, self, binding.overridable(kwargs), timestep)
        outer, _current = _current, call
        t0 = time.perf_counter()
        try:
            out = original(self, *args, **kwargs)
        finally:
            _current = outer
        ts = timestep.detach().float().cpu().tolist() if torch.is_tensor(timestep) else None
        for item in call.items or []:
            _log(dict(event="trunk", binding=binding.name, owner=item.owner,
                      key=item.policy.get("key"), row=item.row, rows=call.rows,
                      step_index=item.step_index, engine_step=item.engine_step,
                      row_timestep=None if ts is None else ts[item.row],
                      branch=item.branch, mode=item.policy["mode"],
                      overridable=item.overridable, call_decision=call.decision,
                      blocks_run=call.blocks_run, blocks_skipped=call.blocks_skipped,
                      live_owners=len(_owners), entry=_meta(item.entry),
                      dit_ms=round((time.perf_counter() - t0) * 1e3, 3), **item.log))
        return out

    return hook


def around_block(binding, kind):
    def hook(original, self, *args, **kwargs):
        call = _current
        if call is None or call.binding is not binding:
            return original(self, *args, **kwargs)
        if call.items is None:
            call.open(binding.entry(kind, args, kwargs), binding.signal(self, kind, args, kwargs))
        if call.decision == "skip":
            call.blocks_skipped += 1
            return binding.identity(kind, args, kwargs)
        call.blocks_run += 1
        return original(self, *args, **kwargs)

    return hook


def around_exit(binding):
    def hook(original, self, x, *args, **kwargs):
        call = _current
        if call is None or call.binding is not binding or call.items is None \
                or self is not binding.exit_module(call.model):
            return original(self, x, *args, **kwargs)
        return original(self, call.close(x), *args, **kwargs)

    return hook


def runner_hooks():
    return {
        f"{_RUNNER}._execute_request_list": around_request_list,
        f"{_RUNNER}._prepare_batch_inputs": around_prepare_batch_inputs,
        f"{_RUNNER}._cleanup_finished_step_requests": around_cleanup,
    }


@contextlib.contextmanager
def rows_for_test(rows):
    """Tests only: pretend the runner announced these row owners."""
    global _rows
    _rows = rows
    try:
        yield
    finally:
        _rows = None
