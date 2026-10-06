"""SGLang's feasibility evaluator: how SGLang's runtime affects execution
requirements and native features, as measured at SGLang 8ca82118e.

Its policy is two pieces of data it owns, both from runtime experiments:
where each capability's seam sits relative to compiled and graph-replayed
code, and which settings make SGLang run a different native method without an
error. It names no technique, implementation or model, and imports nothing
from SGLang.
"""

from __future__ import annotations

import dataclasses
import enum
from types import MappingProxyType
from typing import Mapping

from optimizer.core.capabilities import (
    ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR, SIGNAL_OBSERVE, STEP_OBSERVE, STEP_PREDICTION_OVERRIDE,
    STEP_SCHEDULE_MUTATE, TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE,
)
from optimizer.core.feasibility import CompileScope, Reason, ReasonKind, RuntimeContext
from optimizer.core.specs import Behavior, ExecutionRequirement


class SeamRegion(enum.Enum):
    """Where SGLang runs a capability's seam."""

    DENOISING_LOOP = "denoising_loop"  # outside the DiT call; never compiled or replayed
    DIT_CALL = "dit_call"  # inside the DiT forward; compiled and graph-captured with it


class EnvTrigger(enum.Enum):
    """How SGLang reads an environment variable."""

    BOOL = "bool"  # get_bool_env_var: "1" or "true", any case
    NONEMPTY = "nonempty"  # a list that applies when it has any entry

    def is_set(self, value: str) -> bool:
        if self is EnvTrigger.BOOL:
            return value.lower() in ("1", "true")
        return bool(value.strip())


@dataclasses.dataclass(frozen=True)
class NativeFallback:
    """A setting that makes SGLang accept a native feature and run another method."""

    capability: str
    env_var: str
    trigger: EnvTrigger
    consequence: str


@dataclasses.dataclass(frozen=True)
class SGLangFeasibility:
    seams: Mapping[str, SeamRegion]
    native_fallbacks: tuple[NativeFallback, ...]

    def requirement(self, requirement: ExecutionRequirement, context: RuntimeContext) -> Reason | None:
        region = self.seams.get(requirement.capability)
        if region is None:
            return _unsatisfied(requirement, "no evidence that this seam keeps the behavior in SGLang")
        if region is SeamRegion.DENOISING_LOOP:
            return None
        if context.graph_replay:
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
        for fallback in self.native_fallbacks:
            if fallback.capability == capability and fallback.trigger.is_set(context.env(fallback.env_var)):
                return Reason(ReasonKind.KNOWN_NATIVE_FALLBACK, capability,
                              f"{fallback.env_var} is set: {fallback.consequence}")
        return None


def _unsatisfied(requirement: ExecutionRequirement, detail: str) -> Reason:
    return Reason(ReasonKind.EXECUTION_REQUIREMENT_UNSATISFIED, str(requirement), detail)


def measured() -> SGLangFeasibility:
    """The evaluator with exactly what experiments 001, 002 and 005 measured."""
    return SGLangFeasibility(
        seams=MappingProxyType({
            # exp 001: both behaviors held in eager, compiled and replayed runs.
            STEP_OBSERVE: SeamRegion.DENOISING_LOOP,
            STEP_PREDICTION_OVERRIDE: SeamRegion.DENOISING_LOOP,
            STEP_SCHEDULE_MUTATE: SeamRegion.DENOISING_LOOP,
            # exp 002: under breakable CUDA graphs the hooks ran on 1 of 40
            # calls; under whole-model compile an identity override changed
            # FLUX.2's image and left Qwen-Image-2.1's exact.
            TRUNK_OBSERVE: SeamRegion.DIT_CALL,
            TRUNK_OUTPUT_OVERRIDE: SeamRegion.DIT_CALL,
            SIGNAL_OBSERVE: SeamRegion.DIT_CALL,
        }),
        native_fallbacks=(
            # exp 005; multimodal_gen/runtime/layers/quantization/fp8.py:126-130.
            NativeFallback(ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR, "SGLANG_FORCE_FP8_MARLIN", EnvTrigger.BOOL,
                           "weight-only FP8 (Marlin) would run instead of W8A8"),
            # exp 005; srt/layers/quantization/fp8.py:278.
            NativeFallback(ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR, "SGLANG_FP8_IGNORED_LAYERS",
                           EnvTrigger.NONEMPTY, "only part of the quantizable layers would be FP8"),
        ),
    )
