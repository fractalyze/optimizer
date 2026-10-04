"""The techniques with runtime evidence, and our engine-free implementations of them.

Atomic catalog entries only. Engine-native implementations are declared by
their engine's package; combinations are never declared anywhere (ADR 0019).
"""

from __future__ import annotations

from optimizer.core.capabilities import (
    REQUEST_LOCAL_STATE, SIGNAL_OBSERVE, STEP_OBSERVE, STEP_PREDICTION_OVERRIDE, TIMESTEP_STATE, TRUNK_OBSERVE,
    TRUNK_OUTPUT_OVERRIDE,
)
from optimizer.core.specs import Behavior, ExecutionRequirement, ImplementationSpec, TechniqueSpec

PREDICTION_REUSE = "prediction_reuse"
TEACACHE = "teacache"
FP8_W8A8_DYNAMIC_LINEAR = "fp8_w8a8_dynamic_linear"

TECHNIQUES = (
    # exp 001 "override": the step runs its scheduler update with an earlier
    # step's prediction instead of calling the model (ADR 0011).
    TechniqueSpec(PREDICTION_REUSE,
                  "on chosen denoising steps, reuse an earlier step's noise prediction "
                  "instead of calling the model; the scheduler still steps"),
    # exp 002-004: cross-step reuse of the transformer trunk, decided from a
    # per-call signal (ADR 0012).
    TechniqueSpec(TEACACHE,
                  "skip the transformer trunk on steps where its input signal barely "
                  "changed, rebuilding its output from a cached residual"),
    # exp 005 (ADR 0016).
    TechniqueSpec(FP8_W8A8_DYNAMIC_LINEAR,
                  "linear layers compute in FP8: weights quantized once at load, "
                  "activations quantized per call with scales computed at runtime"),
)

IMPLEMENTATIONS = (
    # docs/architecture.md, "Three techniques, resolved"; requirements per
    # ADR 0013 on the step seams.
    ImplementationSpec("fractalyze-prediction-reuse", PREDICTION_REUSE, frozenset({
        STEP_OBSERVE, STEP_PREDICTION_OVERRIDE, REQUEST_LOCAL_STATE}), frozenset({
        ExecutionRequirement(Behavior.RUNS_EVERY_INVOCATION, STEP_OBSERVE),
        ExecutionRequirement(Behavior.OVERRIDE_EXACT, STEP_PREDICTION_OVERRIDE)})),
    # The capabilities exp 002's policy consumes (opt_trunk_probe/policy.py),
    # and the behavior ADR 0013 declares on its trunk capabilities.
    ImplementationSpec("fractalyze-teacache", TEACACHE, frozenset({
        TIMESTEP_STATE, REQUEST_LOCAL_STATE,
        TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, SIGNAL_OBSERVE}), frozenset({
        ExecutionRequirement(Behavior.RUNS_EVERY_INVOCATION, TRUNK_OBSERVE),
        ExecutionRequirement(Behavior.RUNS_EVERY_INVOCATION, TRUNK_OUTPUT_OVERRIDE),
        ExecutionRequirement(Behavior.OVERRIDE_EXACT, TRUNK_OUTPUT_OVERRIDE)})),
)
