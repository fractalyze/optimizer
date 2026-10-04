"""The default catalog: every atomic entry, gathered from its owner.

It declares what exists and nothing more. It holds no combinations, no
preference between implementations, and no feasibility or conflict rule;
those are computed by the stages that consume it (ADR 0019).
"""

from __future__ import annotations

from optimizer import techniques as engine_free
from optimizer.core.registry import ImplementationRegistry, ProviderRegistry, TechniqueRegistry
from optimizer.engines.sglang import catalog as sglang
from optimizer.engines.vllm_omni import catalog as vllm_omni

TECHNIQUES = engine_free.TECHNIQUES
IMPLEMENTATIONS = engine_free.IMPLEMENTATIONS + sglang.IMPLEMENTATIONS
PROVIDERS = sglang.PROVIDERS + vllm_omni.PROVIDERS


def techniques() -> TechniqueRegistry:
    return TechniqueRegistry(TECHNIQUES)


def implementations(technique_registry: TechniqueRegistry | None = None) -> ImplementationRegistry:
    return ImplementationRegistry(technique_registry or techniques(), IMPLEMENTATIONS)


def providers() -> ProviderRegistry:
    return ProviderRegistry(PROVIDERS)
