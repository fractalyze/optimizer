"""SPIKE: one SGLang plugin that observes and controls denoising steps.

Experiment 001 asks whether a single engine-level integration can observe and
control the denoising steps of every model without knowing which model it is.
So this file must never import, name or branch on a model class. If it ever
needs to, that is the experiment's result, not a bug to patch around.

Control is per request. SGLang gives a plugin no field to carry its own
per-request settings, so the driver writes a JSON table keyed by
`request_id` and points OPT_PROBE_CONTROL at it. A request missing from the
table is only observed.

Modes (one per request, at step K):
  observe    run unchanged; log every step
  bypass     do not run step K at all: no model call and no scheduler update
  override   run step K's scheduler update with step K-1's prediction
  schedule   remove timestep K from the schedule before the loop starts
"""

from __future__ import annotations

import json
import os
import time

_DENOISING = "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.DenoisingStage"
_TIMESTEPS = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation."
    "TimestepPreparationStage.forward"
)
_BCG_REPLAY = (
    "sglang.multimodal_gen.runtime.breakable_cuda_graph.runner."
    "BaseBreakableCudaGraphRunner.replay"
)
_KEY = "opt_probe"  # this probe's slot in the request's own `extra` dict
# SGLang logs a graph-signature miss only once and never counts replays, so the
# probe counts them itself to show whether a step actually ran from a graph.
_replays = [0]


def _policy(request_id):
    path = os.environ.get("OPT_PROBE_CONTROL")
    if not path or not os.path.exists(path):
        return {"mode": "observe"}
    with open(path) as f:
        return json.load(f).get(request_id or "", {"mode": "observe"})


def _state(batch):
    """Request-local state, kept on the request object SGLang already owns."""
    state = batch.extra.get(_KEY)
    if state is None:
        state = {"policy": _policy(batch.request_id), "model_calls": 0, "last_pred": None}
        batch.extra[_KEY] = state
    return state


def _log(record):
    path = os.environ.get("OPT_PROBE_LOG")
    if path:
        record.update(pid=os.getpid(), wall=time.time())
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def _around_step(original, self, ctx, step, batch, server_args):
    state = _state(batch)
    policy = state["policy"]
    k = policy.get("k")
    is_k = step.step_index == k
    record = dict(
        event="step",
        request_id=batch.request_id,
        model=server_args.model_path,  # logged for the record only, never read
        warmup=bool(ctx.is_warmup),
        step_index=step.step_index,
        timestep=float(step.t_host),
        total_steps=len(ctx.timesteps),
        num_inference_steps=ctx.num_inference_steps,
        scheduler_index_before=getattr(ctx.scheduler, "step_index", None),
        mode=policy["mode"],
    )
    if policy["mode"] == "bypass" and is_k:
        record["action"] = "bypassed"
        _log(record)
        return None
    calls_before, replays_before = state["model_calls"], _replays[0]
    result = original(self, ctx, step, batch, server_args)
    record["action"] = "ran"
    record["model_calls_this_step"] = state["model_calls"] - calls_before
    record["graph_replays_this_step"] = _replays[0] - replays_before
    _log(record)
    return result


def _around_predict(original, self, *args, **kwargs):
    batch = kwargs["batch"]
    state = _state(batch)
    policy = state["policy"]
    step_index = kwargs["timestep_index"]
    if (
        policy["mode"] == "override"
        and step_index == policy.get("k")
        and state["last_pred"] is not None
    ):
        return state["last_pred"]
    state["model_calls"] += 1
    pred = original(self, *args, **kwargs)
    state["last_pred"] = pred
    return pred


def _around_timesteps(original, self, batch, server_args):
    policy = _policy(batch.request_id)
    if policy["mode"] == "schedule" and batch.sigmas is None and batch.timesteps is None:
        n = batch.num_inference_steps
        sigmas = list(server_args.pipeline_config.prepare_sigmas(None, n))
        del sigmas[policy["k"]]
        batch.sigmas = sigmas
        _log(dict(event="schedule", request_id=batch.request_id, removed=policy["k"], n=n))
    return original(self, batch, server_args)


def _around_replay(original, self, *args, **kwargs):
    _replays[0] += 1
    return original(self, *args, **kwargs)


def register():
    from sglang.multimodal_gen.runtime.platforms.plugins import HookType, plugin_hook

    plugin_hook(f"{_DENOISING}._run_denoising_step", HookType.AROUND)(_around_step)
    plugin_hook(f"{_DENOISING}._predict_noise_with_cfg", HookType.AROUND)(_around_predict)
    plugin_hook(_TIMESTEPS, HookType.AROUND)(_around_timesteps)
    plugin_hook(_BCG_REPLAY, HookType.AROUND)(_around_replay)
    _log(dict(event="registered"))
