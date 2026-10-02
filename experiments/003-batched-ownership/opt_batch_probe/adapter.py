"""SPIKE: the SGLangAdapter half of experiment 003.

Experiment 002's adapter assumed one execution item per DiT call. Here one DiT
call may carry several requests (SGLang dynamic batching), and this adapter
splits it into one item per batch row, hands each item to the *unchanged*
experiment 002 policy, and puts the results back together. The policy, the
payload forms and the Bindings are imported from experiment 002, not copied:
if this experiment had to change them, that would be its result.

What the adapter needs from SGLang, and where it gets it:
  - which rows a call carries: `forward_batch.seeds`, one per output row in
    request order. A merged batch keeps no per-request id (the scheduler
    renames it `dynamic_batch::<first id>` and drops the other requests'
    `extra`), so the seed is the only per-row identity that survives.
  - per-item state: one slot per row key inside the merged request's `extra`,
    which lives exactly as long as the batch.
  - the row axis: dim 0 of the trunk's entry, signal and result, for both
    models. That is SGLang's convention, not a model's, so no Binding is asked.

A call whose items disagree (some override, some compute) still runs every
block on every row, then splices the overriding rows' payloads in at the exit:
each item gets what it asked for, but nothing is saved. Blocks are skipped
only when every row overrides.
"""

from __future__ import annotations

import contextvars
import json
import os
import time

import torch

from opt_trunk_probe import policy
from opt_trunk_probe.adapter import apply, capture

_KEY = "opt_batch"  # this probe's slot in the (merged) request's `extra`
_current = contextvars.ContextVar("opt_batch_call", default=None)


def _control(key):
    path = os.environ.get("OPT_BATCH_CONTROL")
    if key is None or not path or not os.path.exists(path):
        return {"mode": "observe"}
    with open(path) as f:
        return json.load(f).get(key, {"mode": "observe"})


def _log(record):
    path = os.environ.get("OPT_BATCH_LOG")
    if path:
        record.update(pid=os.getpid(), wall=time.time())
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def _meta(t):
    if t is None:
        return None
    return dict(shape=list(t.shape), dtype=str(t.dtype).removeprefix("torch."))


class ItemCall:
    """One row of one trunk invocation: the `call` the policy sees. It offers
    exactly the fields experiment 002's TrunkCall offered to the policy."""

    def __init__(self, row, key, step_index, num_steps, branch, overridable, slot):
        self.row, self.key = row, key
        self.step_index, self.num_steps, self.branch = step_index, num_steps, branch
        self.overridable = overridable
        self.policy = slot["policy"]
        self.state = slot.setdefault(branch, {})
        self.entry = self.signal = self.override = None
        self.log = {}

    def record(self, **fields):
        self.log.update(fields)


class TrunkCall:
    """One trunk invocation: the batch, its rows, and what happens to its blocks."""

    def __init__(self, binding, model, batch, step_index, overridable):
        self.binding, self.model = binding, model
        self.step_index = step_index
        self.request_id = getattr(batch, "request_id", None)
        self.branch = "uncond" if getattr(batch, "is_cfg_negative", False) else "cond"
        self.warmup = bool(getattr(batch, "is_warmup", False))
        self.overridable = overridable and not self.warmup and batch is not None
        self.keys = [str(s) for s in (getattr(batch, "seeds", None) or [None])]
        self._batch = batch
        self.items = None  # built at the first block, once the row count is known
        self.rows = None
        self.decision = None  # "compute" | "skip" | "splice"
        self.blocks_run = self.blocks_skipped = 0

    def open(self, entry, signal):
        """Split the trunk's entry into items and ask the policy about each."""
        self.rows = entry.shape[0]
        aligned = self.rows == len(self.keys)
        # A row count we cannot attribute (e.g. CFG branches stacked as rows)
        # gets one observe-only item; no row may be overridden for someone else.
        keys = self.keys if aligned else [None] * self.rows
        store = {} if self._batch is None or self.warmup else self._batch.extra.setdefault(_KEY, {})
        self.items = []
        for row, key in enumerate(keys):
            slot = store.setdefault(key, {"policy": _control(key)}) if key else {"policy": {"mode": "observe"}}
            item = ItemCall(row, key, self.step_index, getattr(self._batch, "num_inference_steps", None),
                            self.branch, self.overridable and aligned, slot)
            item.entry = entry[row : row + 1]
            item.signal = None if signal is None else signal[row : row + 1]
            item.override = policy.on_enter(item, item.policy, item.state)
            self.items.append(item)
        wants = [item.override is not None for item in self.items]
        self.decision = "skip" if all(wants) else "splice" if any(wants) else "compute"

    def close(self, x):
        """Give each row what its item decided, and the policy its on_exit."""
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
                        identity_exact=bool((rebuilt == xi).all()),
                        identity_mismatches=int((rebuilt != xi).sum()))
            pieces.append(rebuilt)
            replaced = True
        return torch.cat(pieces, dim=0) if replaced else x


def _forward_context():
    from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context

    try:
        return get_forward_context()
    except Exception:
        return None


def around_dit(binding):
    """trunk_observe: open one trunk invocation per DiT call."""

    def hook(original, self, *args, **kwargs):
        ctx = _forward_context()
        call = TrunkCall(binding, self, getattr(ctx, "forward_batch", None),
                         getattr(ctx, "current_timestep", None), binding.overridable(kwargs))
        token = _current.set(call)
        t0 = time.perf_counter()
        try:
            out = original(self, *args, **kwargs)
        finally:
            _current.reset(token)
        dit_ms = round((time.perf_counter() - t0) * 1e3, 3)
        for item in call.items or []:
            _log(dict(event="trunk", binding=binding.name, request_id=call.request_id,
                      key=item.key, tag=item.policy.get("tag"), row=item.row, rows=call.rows,
                      step_index=call.step_index, branch=call.branch, warmup=call.warmup, mode=item.policy["mode"],
                      overridable=item.overridable, call_decision=call.decision,
                      blocks_run=call.blocks_run, blocks_skipped=call.blocks_skipped,
                      entry=_meta(item.entry), signal=_meta(item.signal), dit_ms=dit_ms,
                      **item.log))
        return out

    return hook


def around_block(binding, kind):
    """The first block call opens the items; blocks are suppressed only when
    every item overrides."""

    def hook(original, self, *args, **kwargs):
        call = _current.get()
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
    """trunk_output_override: the exit norm's input is the trunk's result."""

    def hook(original, self, x, *args, **kwargs):
        call = _current.get()
        if call is None or call.binding is not binding or call.items is None \
                or self is not binding.exit_module(call.model):
            return original(self, x, *args, **kwargs)
        return original(self, call.close(x), *args, **kwargs)

    return hook
