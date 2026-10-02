"""SPIKE: the SGLangAdapter half of experiment 002.

Everything here is the same for every model SGLang serves: how one DiT call
learns its request, step and CFG branch, where per-request state lives, how
control reaches the worker, and how a trunk invocation is opened, intercepted
at its first block and closed at its exit. Which forward, which blocks and
which exit belong to a model is asked of a Binding; this file never names one.
"""

from __future__ import annotations

import contextvars
import json
import os
import time

from opt_trunk_probe import bindings as payloads
from opt_trunk_probe import policy

_KEY = "opt_trunk"  # this probe's slot in the request's own `extra` dict
_current = contextvars.ContextVar("opt_trunk_call", default=None)


def _control(request_id):
    path = os.environ.get("OPT_TRUNK_CONTROL")
    if not path or not os.path.exists(path):
        return {"mode": "observe"}
    with open(path) as f:
        return json.load(f).get(request_id or "", {"mode": "observe"})


def _log(record):
    path = os.environ.get("OPT_TRUNK_LOG")
    if path:
        record.update(pid=os.getpid(), wall=time.time())
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


class TrunkCall:
    """One trunk invocation, as the implementation sees it."""

    def __init__(self, binding, model, batch, step_index, overridable):
        self.binding, self.model = binding, model
        self.request_id = getattr(batch, "request_id", None)
        self.step_index = step_index
        self.num_steps = getattr(batch, "num_inference_steps", None)
        self.branch = "uncond" if getattr(batch, "is_cfg_negative", False) else "cond"
        self.warmup = bool(getattr(batch, "is_warmup", False))
        self.overridable = overridable and not self.warmup
        self.signal = None
        self.entry = None
        self.override = None
        self.blocks_run = 0
        self.blocks_skipped = 0
        self.log = {}
        if batch is None or self.warmup:
            self.policy, self.state = {"mode": "observe"}, {}
        else:
            slot = batch.extra.setdefault(_KEY, {"policy": _control(self.request_id)})
            self.policy = slot["policy"]
            self.state = slot.setdefault(self.branch, {})

    def record(self, **fields):
        self.log.update(fields)


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
        batch = getattr(ctx, "forward_batch", None)
        call = TrunkCall(binding, self, batch, getattr(ctx, "current_timestep", None),
                         binding.overridable(kwargs))
        token = _current.set(call)
        t0 = time.perf_counter()
        try:
            out = original(self, *args, **kwargs)
        finally:
            _current.reset(token)
        _log(dict(event="trunk", binding=binding.name, request_id=call.request_id,
                  step_index=call.step_index, branch=call.branch, warmup=call.warmup,
                  mode=call.policy["mode"], overridable=call.overridable,
                  blocks_run=call.blocks_run, blocks_skipped=call.blocks_skipped,
                  entry=_meta(call.entry), signal=_meta(call.signal),
                  dit_ms=round((time.perf_counter() - t0) * 1e3, 3), **call.log))
        return out

    return hook


def around_block(binding, kind):
    """The first block call is the trunk's entry; while an override is active
    every block returns its identity instead of computing."""

    def hook(original, self, *args, **kwargs):
        call = _current.get()
        if call is None or call.binding is not binding:
            return original(self, *args, **kwargs)
        if call.entry is None:
            call.entry = binding.entry(kind, args, kwargs)
            call.signal = binding.signal(self, kind, args, kwargs)
            call.override = policy.on_enter(call, call.policy, call.state)
        if call.override is not None:
            call.blocks_skipped += 1
            return binding.identity(kind, args, kwargs)
        call.blocks_run += 1
        return original(self, *args, **kwargs)

    return hook


def around_exit(binding):
    """trunk_output_override: the exit norm's input is the trunk's result."""

    def hook(original, self, x, *args, **kwargs):
        call = _current.get()
        if call is None or call.binding is not binding or self is not binding.exit_module(call.model):
            return original(self, x, *args, **kwargs)
        if call.override is not None:
            x = payloads.apply(call.entry, call.override)
            call.record(exit_from="override")
        else:
            entry = call.entry

            def capture(kind):
                return payloads.capture(kind, entry, x)

            replacement = policy.on_exit(call, call.policy, call.state, capture)
            if replacement is not None:
                rebuilt = payloads.apply(entry, replacement)
                call.record(exit_from=f"identity:{replacement[0]}", exit=_meta(x),
                            rebuilt=_meta(rebuilt),
                            identity_exact=bool((rebuilt == x).all()),
                            identity_mismatches=int((rebuilt != x).sum()))
                x = rebuilt
        return original(self, x, *args, **kwargs)

    return hook


def _meta(t):
    if t is None:
        return None
    return dict(shape=list(t.shape), dtype=str(t.dtype).removeprefix("torch."))
