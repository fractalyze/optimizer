"""Explicit, ordered registries of techniques, implementations and providers.

Entries are added by calling `add`; nothing is discovered. Order is
registration order, so every listing is deterministic.
"""

from __future__ import annotations

from typing import Generic, Iterable, TypeVar

from optimizer.core.specs import ImplementationSpec, ProviderKind, ProviderSpec, Target, TechniqueSpec

T = TypeVar("T", TechniqueSpec, ImplementationSpec, ProviderSpec)


class _Registry(Generic[T]):
    def __init__(self, entries: Iterable[T] = ()):
        self._entries: dict[str, T] = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry: T) -> None:
        if entry.id in self._entries:
            raise ValueError(f"duplicate id {entry.id!r}")
        self._entries[entry.id] = entry

    def get(self, id_: str) -> T:
        try:
            return self._entries[id_]
        except KeyError:
            raise KeyError(f"no entry {id_!r}") from None

    def list(self) -> tuple[T, ...]:
        return tuple(self._entries.values())


class TechniqueRegistry(_Registry[TechniqueSpec]):
    pass


class ImplementationRegistry(_Registry[ImplementationSpec]):
    def __init__(self, techniques: TechniqueRegistry, entries: Iterable[ImplementationSpec] = ()):
        self._techniques = techniques
        super().__init__(entries)

    def add(self, entry: ImplementationSpec) -> None:
        self._techniques.get(entry.technique)  # must exist
        super().add(entry)

    def for_technique(self, technique_id: str) -> tuple[ImplementationSpec, ...]:
        """Every implementation of the technique, in registration order. No
        ranking: which one to prefer is decided after feasibility."""
        self._techniques.get(technique_id)
        return tuple(i for i in self.list() if i.technique == technique_id)


class ProviderRegistry(_Registry[ProviderSpec]):
    def for_target(self, target: Target) -> tuple[ProviderSpec, ...]:
        """The target's engine adapter(s), then the Binding(s) for its architecture.

        An architecture with no Binding still gets its engine's adapter, so
        engine-common capabilities resolve on any model."""
        same_engine = [p for p in self.list() if p.engine == target.engine]
        adapters = [p for p in same_engine if p.kind is ProviderKind.ENGINE_ADAPTER]
        bindings = [p for p in same_engine if p.kind is ProviderKind.BINDING and p.architecture == target.architecture]
        return tuple(adapters + bindings)
