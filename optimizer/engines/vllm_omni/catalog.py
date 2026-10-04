"""What vLLM-Omni contributes to the catalog, as validated at vLLM-Omni 68003cf6a.

Only what experiment 004 ran (opt_omni_probe/): its step seams and native
FP8 were read, not run, so they are absent.
"""

from __future__ import annotations

from optimizer.core.capabilities import (
    REQUEST_LOCAL_STATE, SIGNAL_OBSERVE, TIMESTEP_STATE, TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE,
)
from optimizer.core.specs import ProviderKind, ProviderSpec

ENGINE = "vllm-omni"
QWEN_IMAGE_2512 = "qwen-image-2512"

PROVIDERS = (
    ProviderSpec("VllmOmniAdapter", ProviderKind.ENGINE_ADAPTER, ENGINE, None, frozenset({
        TIMESTEP_STATE, REQUEST_LOCAL_STATE})),
    ProviderSpec("VllmOmniQwenImageBinding", ProviderKind.BINDING, ENGINE, QWEN_IMAGE_2512, frozenset({
        TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, SIGNAL_OBSERVE})),
)
