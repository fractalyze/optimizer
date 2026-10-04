"""Feasibility: can a supported implementation run correctly in this runtime?

It starts from a RESOLVED capability resolution and never re-resolves it.
What it adds is the runtime, which only the target's engine can interpret, so
the judging is done by that engine's evaluator; this module asks it about
every execution requirement and every required capability, and requires all
answers to pass (ADR 0018).

Feasibility rejects only what is knowable before launch. A fallback the
engine takes for a reason not known here is caught after the run, by
engagement verification, as FAILED_TO_ENGAGE (ADR 0013).
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Mapping, Protocol

from optimizer.core.resolver import Resolution, Status as ResolutionStatus
from optimizer.core.specs import ExecutionRequirement, ImplementationSpec


class CompileScope(enum.Enum):
    NONE = "none"
    REGIONAL = "regional"  # only the repeated blocks are compiled
    WHOLE_MODEL = "whole_model"


@dataclasses.dataclass(frozen=True)
class RuntimeContext:
    """The concrete conditions of one attempted run, apart from its Target.

    Every field is one a validated rule reads, as the engine applied it, not
    as it was requested (ADR 0013). What a value implies is decided only by
    the engine's evaluator.

    `identity_check_passed` is measured, not configured: whether an override
    returning the computed value left the final output bitwise identical, for
    this target in this compile scope. None means it was not measured."""

    compile_scope: CompileScope = CompileScope.NONE
    graph_replay: bool = False
    identity_check_passed: bool | None = None
    engine_env: Mapping[str, str] = dataclasses.field(default_factory=dict)


class ReasonKind(enum.Enum):
    EXECUTION_REQUIREMENT_UNSATISFIED = "execution_requirement_unsatisfied"
    KNOWN_NATIVE_FALLBACK = "known_native_fallback"
    UNSUPPORTED_RUNTIME_CONDITION = "unsupported_runtime_condition"


@dataclasses.dataclass(frozen=True)
class Reason:
    kind: ReasonKind
    subject: str  # the requirement, capability or engine it is about
    detail: str

    def __str__(self) -> str:
        return f"{self.kind.value}: {self.subject}: {self.detail}"


class Status(enum.Enum):
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"


@dataclasses.dataclass(frozen=True)
class FeasibilityResult:
    implementation: str
    status: Status
    reasons: tuple[Reason, ...] = ()

    def __str__(self) -> str:
        return "\n".join([f"{self.implementation}: {self.status.value}"] + [f"  {r}" for r in self.reasons])


class Evaluator(Protocol):
    """One engine's reading of the runtime: None when it delivers the behavior
    or capability faithfully, a Reason when it does not."""

    def requirement(self, requirement: ExecutionRequirement, context: RuntimeContext) -> Reason | None: ...

    def capability(self, capability: str, context: RuntimeContext) -> Reason | None: ...


def check(implementation: ImplementationSpec, resolution: Resolution, context: RuntimeContext,
          evaluators: Mapping[str, Evaluator]) -> FeasibilityResult:
    """All checks must pass. Every failing one is reported, not just the first.

    The implementation is passed alongside its resolution because the
    resolution carries capability ownership, and the spec carries the
    execution requirements; they must describe the same implementation."""
    if resolution.status is not ResolutionStatus.RESOLVED or resolution.implementation != implementation.id:
        raise ValueError(f"feasibility needs a RESOLVED resolution of {implementation.id}, got:\n{resolution}")
    engine = resolution.target.engine
    evaluator = evaluators.get(engine)
    if evaluator is None:
        reason = Reason(ReasonKind.UNSUPPORTED_RUNTIME_CONDITION, engine,
                        "no evaluator for this engine's runtime, so nothing can be guaranteed")
        return FeasibilityResult(implementation.id, Status.INFEASIBLE, (reason,))
    answers = [evaluator.requirement(r, context) for r in sorted(implementation.execution, key=str)]
    answers += [evaluator.capability(c, context) for c in sorted(resolution.providers)]
    reasons = tuple(r for r in answers if r is not None)
    return FeasibilityResult(implementation.id, Status.INFEASIBLE if reasons else Status.FEASIBLE, reasons)
