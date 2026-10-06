"""Core's metadata types: what a technique is, what an implementation needs,
what a target's components provide, and which target is asked.

All of them are plain, frozen metadata. Building one imports no engine and
runs no engine code, so a registry can be loaded, inspected and tested on a
machine without a GPU (ADR 0017).
"""

from __future__ import annotations

import dataclasses
import enum

from optimizer.core.capabilities import check_capability
from optimizer.core.resources import check_resource


class ParamType(enum.Enum):
    """The value types the validated techniques' parameters use."""

    FLOAT = "float"
    INT_TUPLE = "int_tuple"

    def accepts(self, value: object) -> bool:
        if isinstance(value, bool):
            return False
        if self is ParamType.FLOAT:
            return isinstance(value, (int, float))
        return isinstance(value, tuple) and all(isinstance(v, int) and not isinstance(v, bool) for v in value)


@dataclasses.dataclass(frozen=True)
class ParameterSpec:
    """A conceptual parameter: what the technique lets a trial choose. Ranges
    and search priors belong to the search space, not here."""

    name: str
    type: ParamType


@dataclasses.dataclass(frozen=True)
class TechniqueConfig:
    """Parameter values chosen for one trial. Values are kept as sorted pairs
    so a config is hashable and compares by content."""

    technique: str
    values: tuple[tuple[str, object], ...] = ()

    @classmethod
    def of(cls, technique: str, **values: object) -> "TechniqueConfig":
        return cls(technique, tuple(sorted(values.items())))


@dataclasses.dataclass(frozen=True)
class TechniqueSpec:
    """An optimization concept: what is computed, never how an engine does it.

    `summary` states the numerical method, because two methods reached through
    one engine flag are different techniques (ADR 0016). `parameters` are all
    required: no validated technique has an optional one."""

    id: str
    summary: str
    parameters: tuple[ParameterSpec, ...] = ()

    def validate(self, config: TechniqueConfig) -> None:
        if config.technique != self.id:
            raise ValueError(f"a config for {config.technique!r} cannot configure {self.id!r}")
        given = dict(config.values)
        declared = {p.name: p for p in self.parameters}
        unknown = sorted(given.keys() - declared.keys())
        missing = sorted(declared.keys() - given.keys())
        wrong = sorted(n for n, p in declared.items() if n in given and not p.type.accepts(given[n]))
        if unknown or missing or wrong:
            raise ValueError(f"{self.id}: unknown {unknown}, missing {missing}, wrong type {wrong}")


class Lifecycle(enum.Enum):
    """The two lifecycle constraints of ADR 0009 that decide where an
    implementation is set up. ADR 0009's third, `dynamic_in_forward`, is now
    expressed by execution requirements (ADR 0013)."""

    MUTATES_MODEL = "mutates_model"  # changes the served model, so is fixed when the server is built
    REQUEST_STATE = "request_state"  # has per-request setup, so is configured per request


class Behavior(enum.Enum):
    """The two execution requirements experiment 002's failures call for
    (ADR 0013). Each names behavior, never an engine setting."""

    RUNS_EVERY_INVOCATION = "runs_every_invocation"  # broken by graph replay
    OVERRIDE_EXACT = "override_exact"  # broken by whole-model compile on FLUX.2


@dataclasses.dataclass(frozen=True)
class ExecutionRequirement:
    behavior: Behavior
    capability: str

    def __str__(self) -> str:
        return f"{self.behavior.value}({self.capability})"


@dataclasses.dataclass(frozen=True)
class ImplementationSpec:
    """One concrete realization of a technique, and what it needs.

    There is deliberately no engine or model field: whether it can run on a
    target is computed by resolving `requires`, not listed by hand
    (docs/architecture.md, "Implementation contract"). Native and generic
    implementations differ only in what they require. `execution` says what
    behavior it needs at some of those capabilities; the target's feasibility
    evaluator decides whether the runtime gives it.

    For composition: `owns` names resources it controls exclusively, so no
    other implementation in the same configuration may claim them; `after`
    names resources whose owners must be in place before it. Both are about
    this implementation alone, never about another one by name.

    For planning: `lifecycle` says whether it is fixed when the server is
    built, configured per request, or both."""

    id: str
    technique: str
    requires: frozenset[str]
    execution: frozenset[ExecutionRequirement] = frozenset()
    owns: frozenset[str] = frozenset()
    after: frozenset[str] = frozenset()
    lifecycle: frozenset[Lifecycle] = frozenset()

    def __post_init__(self):
        for capability in self.requires:
            check_capability(capability)
        for requirement in self.execution:
            if requirement.capability not in self.requires:
                raise ValueError(f"{self.id}: {requirement} is on a capability it does not require")
        for resource in self.owns | self.after:
            check_resource(resource)
        if self.owns & self.after:
            raise ValueError(f"{self.id}: cannot come after the owner of a resource it owns itself")


class ProviderKind(enum.Enum):
    ENGINE_ADAPTER = "engine_adapter"  # one engine, every model
    BINDING = "binding"  # one engine x one model


@dataclasses.dataclass(frozen=True)
class ProviderSpec:
    """A target component and the capabilities it makes available.

    An adapter has no `model`. A Binding names the model it binds; its
    capabilities run through its engine's adapter, but it is what makes them
    available on that model, so resolution names the Binding."""

    id: str
    kind: ProviderKind
    engine: str
    model: str | None
    provides: frozenset[str]

    def __post_init__(self):
        if (self.kind is ProviderKind.BINDING) != (self.model is not None):
            raise ValueError(f"{self.id}: a Binding names a model and an adapter does not")
        for capability in self.provides:
            check_capability(capability)


@dataclasses.dataclass(frozen=True)
class Target:
    """The engine and model a resolution is asked about.

    Only what selects the providers. GPU, compile and graph mode, and engine
    settings decide feasibility, which is a later stage (ADR 0013)."""

    engine: str
    model: str
