"""SGLang's plan translator: lowers an engine-neutral ExecutionPlan to SGLang
server settings and per-request policies, as validated at SGLang 8ca82118e.

It returns configuration only; launching, sending requests and reading
evidence are the execution stage's. Its knowledge is explicit data built by
`validated()`, each value from the experiment that used it. The plugins are
the experiments' probes (`experiments/00*/pyproject.toml`) until production
plugins replace them.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Mapping

from optimizer.core.feasibility import CompileScope
from optimizer.core.plan import ExecutionPlan
from optimizer.engines.sglang.catalog import ENGINE

STEP_PLUGIN = "opt_step_probe"  # exp 001: the adapter's step seams
TRUNK_PLUGIN = "opt_trunk_probe"  # exp 002: trunk hooks through the model's Binding


@dataclasses.dataclass(frozen=True)
class ServerLowering:
    server_kwargs: tuple[tuple[str, object], ...] = ()
    plugins: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class SGLangServerConfig:
    """What `DiffGenerator.from_pretrained` and the worker environment need.
    `model_path` is the user's checkpoint, unchanged; `SGLANG_PLUGINS` is
    derived from `plugins` by whoever launches."""

    model_path: str
    server_kwargs: tuple[tuple[str, object], ...]
    env: tuple[tuple[str, str], ...]
    plugins: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class SGLangRequestConfig:
    """Per plugin, the policy for one request. SGLang has no request field for
    plugin settings, so the experiments carried these in a control table keyed
    by request id; how execution delivers them is the next stage's choice."""

    policies: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]


@dataclasses.dataclass(frozen=True)
class SGLangPlanTranslator:
    baseline_plugins: tuple[str, ...]  # present in every server; inert until a request asks
    server: Mapping[str, ServerLowering]  # by implementation id
    request_plugin: Mapping[str, str]  # by implementation id: the plugin reading its policy

    def lower(self, plan: ExecutionPlan) -> tuple[SGLangServerConfig, SGLangRequestConfig]:
        key = plan.server
        if key.engine != ENGINE:
            raise ValueError(f"cannot lower a plan for {key.engine!r} to SGLang")
        kwargs = list(_execution_mode_kwargs(key.settings.compile_scope, key.settings.graph_replay))
        if key.model.revision is not None:
            kwargs.append(("revision", key.model.revision))
        plugins = set(self.baseline_plugins)
        for contribution in key.contributions:
            lowering = self._lookup(self.server, contribution.implementation)
            if contribution.values:
                raise ValueError(f"{contribution.implementation}: no server-time parameters are lowered")
            kwargs += lowering.server_kwargs
            plugins |= set(lowering.plugins)
        policies = tuple(
            (self._lookup(self.request_plugin, r.implementation),
             (("implementation", r.implementation),) + r.values)
            for r in plan.requests)
        server = SGLangServerConfig(key.model.checkpoint, tuple(sorted(kwargs)), key.settings.engine_env,
                                    tuple(sorted(plugins)))
        return server, SGLangRequestConfig(policies)

    @staticmethod
    def _lookup(table, implementation):
        try:
            return table[implementation]
        except KeyError:
            raise ValueError(f"SGLang has no lowering for {implementation!r}") from None


def _execution_mode_kwargs(scope: CompileScope, graph_replay: bool) -> tuple[tuple[str, object], ...]:
    """The server arguments the experiments used for each mode
    (experiments 001 and 002 READMEs)."""
    kwargs = []
    if scope is not CompileScope.NONE:
        kwargs.append(("enable_torch_compile", True))
    if scope is CompileScope.REGIONAL:
        kwargs.append(("regional_compile", True))
    if graph_replay:
        kwargs.append(("enable_breakable_cuda_graph", True))
    return tuple(kwargs)


def validated() -> SGLangPlanTranslator:
    return SGLangPlanTranslator(
        # exp 001: with no policy for a request the step plugin only observes,
        # so it is loaded always and request-only step techniques need no
        # server change.
        baseline_plugins=(STEP_PLUGIN,),
        server=MappingProxyType({
            # exp 005: native FP8 W8A8 is one server argument.
            "sglang-native-fp8-w8a8": ServerLowering(server_kwargs=(("quantization", "fp8"),)),
            # exp 002: trunk hooks are a plugin loaded before the model is built.
            "fractalyze-teacache": ServerLowering(plugins=(TRUNK_PLUGIN,)),
        }),
        request_plugin=MappingProxyType({
            "fractalyze-teacache": TRUNK_PLUGIN,
            "fractalyze-prediction-reuse": STEP_PLUGIN,
        }),
    )
