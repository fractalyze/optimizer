"""SPIKE: the implementation layer of experiment 002.

This is the code a generic technique (e.g. a TeaCache-style cache) would be
written as. It must stay free of SGLang, model classes and tensor layouts: it
sees one trunk invocation through the capability API below and nothing else.
If it ever needs to name a model or index a tensor, that is the experiment's
result, not something to patch around.

Capabilities consumed:
  timestep_state       call.step_index, call.num_steps
  request_local_state  `state`, a dict owned by one request and one CFG branch
  trunk_observe        on_enter / on_exit are called once per trunk invocation
  trunk_output_override  on_enter may return a payload to use instead of
                       running the trunk; on_exit may replace the result
  signal_observe       call.signal: an opaque per-invocation tensor; the only
                       contract is "same shape for every call of one request
                       and branch"

Modes, one per request:
  observe            change nothing
  identity_output    run the trunk, then hand its own output back through the
                     override path
  identity_residual  run the trunk, then rebuild its output from a residual
                     payload (the form a cross-step cache stores)
  reuse              at step k store a residual; at step k+1 do not run the
                     trunk, rebuild its output from that residual instead
"""

from __future__ import annotations


def _rel_l1(cur, prev):
    """Relative L1 change between two signals of the same shape."""
    if prev is None or cur is None:
        return None
    return float((cur.float() - prev.float()).abs().mean() / prev.float().abs().mean())


def on_enter(call, policy, state):
    """Decide before the trunk runs. Returns a payload to skip it, or None."""
    state.setdefault("signal_prev", None)
    rel = _rel_l1(call.signal, state["signal_prev"])
    state["signal_prev"] = None if call.signal is None else call.signal.clone()
    call.record(signal_rel_l1=rel)

    if (
        policy["mode"] == "reuse"
        and call.step_index == policy["k"] + 1
        and call.overridable
        and state.get("payload") is not None
    ):
        call.record(decision="reuse")
        return state.pop("payload")
    call.record(decision="compute")
    return None


def on_exit(call, policy, state, capture):
    """After the trunk ran. `capture(kind)` packs its result into a payload.

    Returns a payload to use instead of the trunk's own result, or None.
    """
    mode = policy["mode"]
    if mode == "identity_output" and call.overridable:
        return capture("output")
    if mode == "identity_residual" and call.overridable:
        return capture("residual")
    if mode == "reuse" and call.step_index == policy["k"]:
        state["payload"] = capture("residual")
    return None
