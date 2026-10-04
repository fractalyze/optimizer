"""The techniques, implementations and target providers with runtime evidence.

Every entry names the experiment that validated it. A capability is listed
for a provider only where an experiment exercised it there; code reading alone
is not enough (vLLM-Omni's step seams and native FP8 are therefore absent).
"""

from __future__ import annotations

from optimizer.core.registry import ImplementationRegistry, ProviderRegistry, TechniqueRegistry
from optimizer.core.specs import (
    REQUEST_LOCAL_STATE, SIGNAL_OBSERVE, STEP_OBSERVE, STEP_PREDICTION_OVERRIDE, STEP_SCHEDULE_MUTATE,
    TIMESTEP_STATE, TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, ImplementationSpec, ProviderKind, ProviderSpec,
    TechniqueSpec, ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR,
)

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
    # docs/architecture.md, "Three techniques, resolved".
    ImplementationSpec("fractalyze-prediction-reuse", PREDICTION_REUSE, frozenset({
        STEP_OBSERVE, STEP_PREDICTION_OVERRIDE, REQUEST_LOCAL_STATE})),
    # The capabilities exp 002's policy consumes (opt_trunk_probe/policy.py).
    ImplementationSpec("fractalyze-teacache", TEACACHE, frozenset({
        TIMESTEP_STATE, REQUEST_LOCAL_STATE,
        TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, SIGNAL_OBSERVE})),
    ImplementationSpec("sglang-native-fp8-w8a8", FP8_W8A8_DYNAMIC_LINEAR, frozenset({
        ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR})),
)

_TRUNK = frozenset({TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, SIGNAL_OBSERVE})
PROVIDERS = (
    # Step seams: exp 001 (Qwen-Image-2.1, FLUX.2-klein). Timestep and
    # per-owner state: exp 002-003. FP8: exp 005, at SGLang 8ca82118e.
    ProviderSpec("SGLangAdapter", ProviderKind.ENGINE_ADAPTER, "sglang", None, frozenset({
        STEP_OBSERVE, STEP_PREDICTION_OVERRIDE, STEP_SCHEDULE_MUTATE,
        TIMESTEP_STATE, REQUEST_LOCAL_STATE,
        ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR})),
    # exp 002 Bindings (opt_trunk_probe/bindings.py), batched in exp 003.
    ProviderSpec("SGLangQwenImage21Binding", ProviderKind.BINDING, "sglang", "qwen-image-2.1", _TRUNK),
    ProviderSpec("SGLangFlux2Binding", ProviderKind.BINDING, "sglang", "flux.2-klein", _TRUNK),
    # exp 004, at vLLM-Omni 68003cf6a (opt_omni_probe/).
    ProviderSpec("VllmOmniAdapter", ProviderKind.ENGINE_ADAPTER, "vllm-omni", None, frozenset({
        TIMESTEP_STATE, REQUEST_LOCAL_STATE})),
    ProviderSpec("VllmOmniQwenImageBinding", ProviderKind.BINDING, "vllm-omni", "qwen-image-2512", _TRUNK),
)


def techniques() -> TechniqueRegistry:
    return TechniqueRegistry(TECHNIQUES)


def implementations(technique_registry: TechniqueRegistry | None = None) -> ImplementationRegistry:
    return ImplementationRegistry(technique_registry or techniques(), IMPLEMENTATIONS)


def providers() -> ProviderRegistry:
    return ProviderRegistry(PROVIDERS)
