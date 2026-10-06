"""Composition: can several individually feasible implementations coexist, and
in what relative order must they be in place?

It starts where feasibility ends and never repeats an earlier stage: each
candidate arrives with its RESOLVED resolution and its FEASIBLE result. It
reads only the candidates' own declarations, so no combination is ever listed
anywhere; the configuration being asked about is built here, from atomic
entries (ADR 0020). It decides nothing about how or when anything is
installed: that is execution planning.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Sequence

from optimizer.core.feasibility import FeasibilityResult, Status as FeasibilityStatus
from optimizer.core.resolver import Resolution, Status as ResolutionStatus
from optimizer.core.specs import ImplementationSpec


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One implementation with the evidence that it is supported and feasible."""

    implementation: ImplementationSpec
    resolution: Resolution
    feasibility: FeasibilityResult

    def __post_init__(self):
        impl = self.implementation.id
        if self.resolution.status is not ResolutionStatus.RESOLVED or self.resolution.implementation != impl:
            raise ValueError(f"{impl}: a candidate needs its own RESOLVED resolution")
        if self.feasibility.status is not FeasibilityStatus.FEASIBLE or self.feasibility.implementation != impl:
            raise ValueError(f"{impl}: a candidate needs its own FEASIBLE result")


class ReasonKind(enum.Enum):
    EXCLUSIVE_RESOURCE = "exclusive_resource"
    ORDERING_CYCLE = "ordering_cycle"


@dataclasses.dataclass(frozen=True)
class Reason:
    kind: ReasonKind
    subject: str  # the contested resource, or the implementations on a cycle
    detail: str

    def __str__(self) -> str:
        return f"{self.kind.value}: {self.subject}: {self.detail}"


@dataclasses.dataclass(frozen=True)
class Order:
    """`before` must be in place before `after`, because `after` declared it
    comes after the owner of `resource`."""

    before: str
    after: str
    resource: str

    def __str__(self) -> str:
        return f"{self.before} before {self.after} ({self.resource})"


class Status(enum.Enum):
    COMPOSABLE = "COMPOSABLE"
    CONFLICT = "CONFLICT"


@dataclasses.dataclass(frozen=True)
class CompositionResult:
    status: Status
    implementations: tuple[str, ...]
    ordering: tuple[Order, ...] = ()  # a partial order; empty unless COMPOSABLE
    reasons: tuple[Reason, ...] = ()

    def __str__(self) -> str:
        lines = [f"{' + '.join(self.implementations)}: {self.status.value}"]
        lines += [f"  {o}" for o in self.ordering] + [f"  {r}" for r in self.reasons]
        return "\n".join(lines)


def compose(candidates: Sequence[Candidate]) -> CompositionResult:
    """Exclusive claims must not overlap, and the ordering they imply must be
    acyclic. Sharing a required capability is never a conflict. The result
    does not depend on the order of `candidates`."""
    by_id = {c.implementation.id: c.implementation for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("an implementation appears more than once")
    if len({c.resolution.target for c in candidates}) > 1:
        raise ValueError("candidates were resolved for different targets")
    if len({c.feasibility.context for c in candidates}) > 1:
        raise ValueError("candidates were judged feasible in different runtime contexts")
    ids = tuple(sorted(by_id))

    owners: dict[str, list[str]] = {}
    for impl_id in ids:
        for resource in sorted(by_id[impl_id].owns):
            owners.setdefault(resource, []).append(impl_id)
    reasons = [Reason(ReasonKind.EXCLUSIVE_RESOURCE, resource, f"claimed exclusively by {', '.join(claimants)}")
               for resource, claimants in sorted(owners.items()) if len(claimants) > 1]

    ordering = tuple(Order(owner, impl_id, resource)
                     for impl_id in ids for resource in sorted(by_id[impl_id].after)
                     for owner in owners.get(resource, ()))
    cycle = _on_a_cycle(ids, ordering)
    if cycle:
        reasons.append(Reason(ReasonKind.ORDERING_CYCLE, ", ".join(cycle),
                              "their declared ordering cannot all hold at once"))

    if reasons:
        return CompositionResult(Status.CONFLICT, ids, (), tuple(reasons))
    return CompositionResult(Status.COMPOSABLE, ids, ordering)


def _on_a_cycle(ids: tuple[str, ...], ordering: tuple[Order, ...]) -> tuple[str, ...]:
    """The implementations a topological sort cannot place, in id order."""
    incoming = {i: 0 for i in ids}
    for o in ordering:
        incoming[o.after] += 1
    ready = [i for i in ids if incoming[i] == 0]
    while ready:
        node = ready.pop()
        for o in ordering:
            if o.before == node:
                incoming[o.after] -= 1
                if incoming[o.after] == 0:
                    ready.append(o.after)
    return tuple(i for i in ids if incoming[i] > 0)
