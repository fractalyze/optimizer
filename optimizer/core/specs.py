"""The descriptors Core resolves: what a technique is, what an implementation
needs, what a target's components provide, and which target is asked.

All of them are plain, frozen metadata. Building one imports no engine and
runs no engine code, so a registry can be loaded, inspected and tested on a
machine without a GPU (ADR 0017).
"""

from __future__ import annotations

import dataclasses
import enum

# Capability identifiers: semantic operations, never hooks or seams. The names
# and meanings are defined in docs/architecture.md ("Capabilities in V1").
# Only capabilities with runtime evidence are listed; add one here only
# together with the experiment that validates it.
STEP_OBSERVE = "step_observe"
STEP_PREDICTION_OVERRIDE = "step_prediction_override"
STEP_SCHEDULE_MUTATE = "step_schedule_mutate"
TIMESTEP_STATE = "timestep_state"
REQUEST_LOCAL_STATE = "request_local_state"
TRUNK_OBSERVE = "trunk_observe"
TRUNK_OUTPUT_OVERRIDE = "trunk_output_override"
SIGNAL_OBSERVE = "signal_observe"

CAPABILITIES = frozenset({
    STEP_OBSERVE, STEP_PREDICTION_OVERRIDE, STEP_SCHEDULE_MUTATE,
    TIMESTEP_STATE, REQUEST_LOCAL_STATE,
    TRUNK_OBSERVE, TRUNK_OUTPUT_OVERRIDE, SIGNAL_OBSERVE,
})

# Engine-native features: the engine implements the method whole (ADR 0016).
# Any id under this prefix is accepted. FP8's id happens to match its
# technique's; that is not a rule, and a native implementation may need
# several such capabilities.
ENGINE_FEATURE_PREFIX = "engine_feature."
ENGINE_FEATURE_FP8_W8A8_DYNAMIC_LINEAR = ENGINE_FEATURE_PREFIX + "fp8_w8a8_dynamic_linear"


def check_capability(capability: str) -> None:
    """Rejects a misspelled or unvalidated capability at registration, where
    it would otherwise surface later as a confusing UNSUPPORTED."""
    if capability in CAPABILITIES:
        return
    if capability.startswith(ENGINE_FEATURE_PREFIX) and len(capability) > len(ENGINE_FEATURE_PREFIX):
        return
    raise ValueError(f"unknown capability {capability!r}")


@dataclasses.dataclass(frozen=True)
class TechniqueSpec:
    """An optimization concept: what is computed, never how an engine does it.

    `summary` states the numerical method, because two methods reached through
    one engine flag are different techniques (ADR 0016)."""

    id: str
    summary: str


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
    evaluator decides whether the runtime gives it."""

    id: str
    technique: str
    requires: frozenset[str]
    execution: frozenset[ExecutionRequirement] = frozenset()

    def __post_init__(self):
        for capability in self.requires:
            check_capability(capability)
        for requirement in self.execution:
            if requirement.capability not in self.requires:
                raise ValueError(f"{self.id}: {requirement} is on a capability it does not require")


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
