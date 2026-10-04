"""SGLangAdapter's feasibility evaluator, traced at SGLang 8ca82118e.

Only rules a runtime experiment demonstrated. Nothing here names a technique,
an implementation or a model: execution requirements are judged by where a
capability's seam sits relative to compiled and graph-replayed code, and
native features by the settings that make SGLang run a different method.
Pure Python; SGLang is never imported.
"""

from __future__ import annotations

from optimizer.core.feasibility import CompileScope, Reason, ReasonKind, RuntimeContext
from optimizer.core.specs import (
    ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR, SIGNAL_OBSERVE, STEP_OBSERVE, STEP_PREDICTION_OVERRIDE,
    STEP_SCHEDULE_MUTATE, TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, Behavior, ExecutionRequirement,
)

# Seams in the denoising loop, outside the DiT call: both behaviors held in
# eager, compiled and graph-replayed runs (exp 001).
_LOOP = frozenset({STEP_OBSERVE, STEP_PREDICTION_OVERRIDE, STEP_SCHEDULE_MUTATE})
# Seams inside the DiT forward (exp 002, ADR 0013).
_IN_DIT = frozenset({TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, SIGNAL_OBSERVE})

# Settings that make `quantization="fp8"` run another method, without an
# error (exp 005, multimodal_gen/runtime/layers/quantization/fp8.py:126-130,
# srt/layers/quantization/fp8.py:278). Truthiness as SGLang's get_bool_env_var.
_FP8_FALLBACKS = (
    ("SGLANG_FORCE_FP8_MARLIN", lambda v: v.lower() in ("1", "true"),
     "weight-only FP8 (Marlin) would run instead of W8A8"),
    ("SGLANG_FP8_IGNORED_LAYERS", lambda v: bool(v.strip()),
     "only part of the quantizable layers would be FP8"),
)


def _unsatisfied(requirement: ExecutionRequirement, detail: str) -> Reason:
    return Reason(ReasonKind.EXECUTION_REQUIREMENT_UNSATISFIED, str(requirement), detail)


class SGLangFeasibility:

    def requirement(self, requirement: ExecutionRequirement, context: RuntimeContext) -> Reason | None:
        if requirement.capability in _LOOP:
            return None
        if requirement.capability not in _IN_DIT:
            return _unsatisfied(requirement, "no evidence that this seam keeps the behavior in SGLang")
        if context.graph_replay:
            # exp 002: under breakable CUDA graphs the hooks ran on 1 of 40 calls.
            return _unsatisfied(requirement, "code inside the model call does not run when the call "
                                             "replays from a graph")
        if requirement.behavior is Behavior.RUNS_EVERY_INVOCATION or context.compile_scope is CompileScope.NONE:
            return None
        # Compiled code sits between the override and the model; only an
        # identity check on the final output establishes exactness (ADR 0013).
        if context.identity_check_passed is None:
            return _unsatisfied(requirement, "the override's exactness under compile is not established: "
                                             "no identity check for this target and compile scope")
        if not context.identity_check_passed:
            return _unsatisfied(requirement, "the identity check failed: an override returning the "
                                             "computed value changed the output")
        return None

    def capability(self, capability: str, context: RuntimeContext) -> Reason | None:
        if capability != ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR:
            return None
        for var, is_set, consequence in _FP8_FALLBACKS:
            if is_set(context.engine_env.get(var, "")):
                return Reason(ReasonKind.KNOWN_NATIVE_FALLBACK, capability, f"{var} is set: {consequence}")
        return None
