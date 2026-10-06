"""What SGLang contributes to the catalog: its adapter, its Bindings, and its
native implementations, as validated at SGLang 8ca82118e.

A capability is listed for a provider only where an experiment exercised it.
"""

from __future__ import annotations

from optimizer.core.capabilities import (
    ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR, REQUEST_LOCAL_STATE, SIGNAL_OBSERVE, STEP_OBSERVE,
    STEP_PREDICTION_OVERRIDE, STEP_SCHEDULE_MUTATE, TIMESTEP_STATE, TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE,
)
from optimizer.core.resources import LINEAR_LAYERS
from optimizer.core.specs import ImplementationSpec, Lifecycle, ProviderKind, ProviderSpec
from optimizer.techniques import FP8_W8A8_DYNAMIC_LINEAR

ENGINE = "sglang"
QWEN_IMAGE_2_1 = "qwen-image-2.1"
FLUX_2_KLEIN = "flux.2-klein"

PROVIDERS = (
    # Step seams: exp 001 (Qwen-Image-2.1, FLUX.2-klein). Timestep and
    # per-owner state: exp 002-003. FP8: exp 005.
    ProviderSpec("SGLangAdapter", ProviderKind.ENGINE_ADAPTER, ENGINE, None, frozenset({
        STEP_OBSERVE, STEP_PREDICTION_OVERRIDE, STEP_SCHEDULE_MUTATE,
        TIMESTEP_STATE, REQUEST_LOCAL_STATE,
        ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR})),
    # exp 002 Bindings (opt_trunk_probe/bindings.py), batched in exp 003.
    ProviderSpec("SGLangQwenImage21Binding", ProviderKind.BINDING, ENGINE, QWEN_IMAGE_2_1, frozenset({
        TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, SIGNAL_OBSERVE})),
    ProviderSpec("SGLangFlux2Binding", ProviderKind.BINDING, ENGINE, FLUX_2_KLEIN, frozenset({
        TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, SIGNAL_OBSERVE})),
)

IMPLEMENTATIONS = (
    # exp 005: `quantization="fp8"`, online W8A8 (ADR 0016); it decides what
    # every quantizable linear layer computes with.
    ImplementationSpec("sglang-native-fp8-w8a8", FP8_W8A8_DYNAMIC_LINEAR, frozenset({
        ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR}), owns=frozenset({LINEAR_LAYERS}),
        # Weights quantized at load, fixed for the server's life (exp 005).
        lifecycle=frozenset({Lifecycle.MUTATES_MODEL})),
)
