"""Execution planning: for one composable configuration and its parameter
values, what is fixed when the server is built, and what varies per request?

It starts from a COMPOSABLE result and never repeats an earlier stage. Where
each implementation goes follows from its lifecycle alone: one that mutates
the model contributes to the server, one with per-request state contributes
to the request, and one with both contributes to both. The plan is
engine-neutral; an engine's translator lowers it to that engine's settings
(ADR 0021). Nothing here launches, schedules or reuses a server.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

from optimizer.core.composition import Candidate, CompositionResult, Order, Status as CompositionStatus
from optimizer.core.feasibility import CompileScope, RuntimeContext
from optimizer.core.registry import TechniqueRegistry
from optimizer.core.specs import Lifecycle, ModelRef, Target, TechniqueConfig


@dataclasses.dataclass(frozen=True)
class ServerContribution:
    """An implementation that must be part of the server when it is built,
    with any of its parameters that are fixed with it."""

    implementation: str
    technique: str
    values: tuple[tuple[str, object], ...] = ()


@dataclasses.dataclass(frozen=True)
class RequestContribution:
    """An implementation configured per request, with that request's values."""

    implementation: str
    technique: str
    values: tuple[tuple[str, object], ...] = ()


@dataclasses.dataclass(frozen=True)
class ServerSettings:
    """The facts of a RuntimeContext that decide how a server is built.

    Evidence about a target, such as an identity check's result, is not one
    of them: two servers built the same way are the same server."""

    compile_scope: CompileScope
    graph_replay: bool
    engine_env: tuple[tuple[str, str], ...]

    @classmethod
    def of(cls, context: RuntimeContext) -> "ServerSettings":
        return cls(context.compile_scope, context.graph_replay, context.engine_env)


@dataclasses.dataclass(frozen=True)
class ServerKey:
    """The live-server reuse boundary: plans with equal keys can run on the
    same started server without rebuilding it. Immutable and hashable.

    It names the checkpoint, not the architecture: two checkpoints of one
    class are different servers."""

    engine: str
    model: ModelRef
    settings: ServerSettings
    contributions: tuple[ServerContribution, ...]  # sorted by implementation


@dataclasses.dataclass(frozen=True)
class ExecutionPlan:
    """One dynamically composed configuration, ready to be lowered by an engine."""

    server: ServerKey
    requests: tuple[RequestContribution, ...]  # sorted by implementation
    ordering: tuple[Order, ...]  # the composer's partial order, unchanged

    def __str__(self) -> str:
        key = self.server
        lines = [f"server: {key.engine} + {key.model.checkpoint} {key.settings}"]
        lines += [f"  {c.implementation}{dict(c.values) or ''}" for c in key.contributions]
        lines += ["request:"] + [f"  {r.implementation} {dict(r.values)}" for r in self.requests]
        lines += [f"order: {o}" for o in self.ordering]
        return "\n".join(lines)


def plan(composition: CompositionResult, candidates: Sequence[Candidate], configs: Sequence[TechniqueConfig],
         techniques: TechniqueRegistry) -> ExecutionPlan:
    """Places each candidate by its lifecycle and attaches its technique's values.

    Values go with the request when the implementation has per-request state,
    and with the server otherwise, since then nothing could change them
    without rebuilding it."""
    if composition.status is not CompositionStatus.COMPOSABLE:
        raise ValueError(f"only a COMPOSABLE configuration can be planned, got:\n{composition}")
    by_id = {c.implementation.id: c for c in candidates}
    if tuple(sorted(by_id)) != composition.implementations or len(by_id) != len(candidates):
        raise ValueError("the candidates are not the ones the composer approved")
    targets: set[Target] = {c.resolution.target for c in candidates}
    contexts: set[RuntimeContext] = {c.feasibility.context for c in candidates}
    if len(targets) != 1 or len(contexts) != 1:
        raise ValueError("a plan needs one target and one runtime context")

    values = _validated_values(candidates, configs, techniques)
    server, requests = [], []
    for impl_id in composition.implementations:
        impl = by_id[impl_id].implementation
        mine = values.get(impl.technique, ())
        per_request = Lifecycle.REQUEST_STATE in impl.lifecycle
        if Lifecycle.MUTATES_MODEL in impl.lifecycle or (mine and not per_request):
            server.append(ServerContribution(impl.id, impl.technique, () if per_request else mine))
        if per_request:
            requests.append(RequestContribution(impl.id, impl.technique, mine))

    (target,), (context,) = targets, contexts
    key = ServerKey(target.engine, target.model, ServerSettings.of(context), tuple(server))
    return ExecutionPlan(key, tuple(requests), composition.ordering)


def _validated_values(candidates, configs, techniques) -> dict[str, tuple[tuple[str, object], ...]]:
    """Every configured technique is in the plan, configured once, and valid;
    a technique with parameters must be configured."""
    planned = {c.implementation.technique for c in candidates}
    given = {}
    for config in configs:
        if config.technique not in planned:
            raise ValueError(f"{config.technique!r} is configured but not in the plan")
        if config.technique in given:
            raise ValueError(f"{config.technique!r} is configured twice")
        given[config.technique] = config
    for technique_id in sorted(planned):
        spec = techniques.get(technique_id)
        spec.validate(given.get(technique_id, TechniqueConfig(technique_id)))
    return {t: c.values for t, c in given.items()}
