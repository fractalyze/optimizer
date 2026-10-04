"""Capability resolution: can this target supply what this implementation needs?

It answers "supported", never "feasible": the GPU, the applied execution mode
and engine settings are the next stage's question (ADR 0013). It knows no
technique, implementation or engine by name.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Mapping

from optimizer.core.registry import ProviderRegistry
from optimizer.core.specs import ImplementationSpec, Target


class Status(enum.Enum):
    RESOLVED = "RESOLVED"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclasses.dataclass(frozen=True)
class Resolution:
    implementation: str
    target: Target
    status: Status
    providers: Mapping[str, str]
    missing: tuple[str, ...] = ()
    ambiguous: Mapping[str, tuple[str, ...]] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"{self.implementation} on {self.target.engine} + {self.target.model}: {self.status.value}"]
        lines += [f"  {cap} -> {provider}" for cap, provider in self.providers.items()]
        lines += [f"  {cap} -> (none)" for cap in self.missing]
        lines += [f"  {cap} -> {' | '.join(ps)} (ambiguous)" for cap, ps in self.ambiguous.items()]
        return "\n".join(lines)


def resolve(implementation: ImplementationSpec, target: Target, providers: ProviderRegistry) -> Resolution:
    """Matches each required capability to exactly one of the target's providers.

    No capability is ever assigned by precedence: if two providers on one
    target claim it, the result is AMBIGUOUS and the registry must be fixed.
    Missing capabilities take priority over ambiguous ones in the status, since
    no choice of provider could make an unsupported implementation run."""
    active = providers.for_target(target)
    found, missing, ambiguous = {}, [], {}
    for capability in sorted(implementation.requires):
        claims = tuple(p.id for p in active if capability in p.provides)
        if not claims:
            missing.append(capability)
        elif len(claims) > 1:
            ambiguous[capability] = claims
        else:
            found[capability] = claims[0]
    status = Status.UNSUPPORTED if missing else Status.AMBIGUOUS if ambiguous else Status.RESOLVED
    return Resolution(implementation.id, target, status, found, tuple(missing), ambiguous)
