"""SPIKE: the smallest optimizer path for an engine-native technique (exp 005).

    Technique -> Implementation -> required capability -> EngineAdapter
      -> feasibility -> native configuration -> run -> engagement -> status

Nothing here is a framework. It exists to see whether the architecture's
concepts hold one engine-native technique without leaking engine details
upward: the Technique names a numerical method, the Implementation names the
capability it needs, and only the EngineAdapter knows a flag, a GPU check or a
class name.

Pure Python; no SGLang or torch import, so it is tested on a CPU.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Mapping

VALID, UNSUPPORTED, INFEASIBLE, RUNTIME_ERROR, FAILED_TO_ENGAGE = (
    "VALID", "UNSUPPORTED", "INFEASIBLE", "RUNTIME_ERROR", "FAILED_TO_ENGAGE")


@dataclasses.dataclass(frozen=True)
class Technique:
    """A numerical method, independent of engine. Parameters here are
    conceptual: what is quantized, not how a kernel does it."""

    id: str
    summary: str
    params: Mapping[str, object]


# Experiment 005's taxonomy: methods that differ in what is computed in FP8
# are different techniques, even when an engine selects them with one flag.
TECHNIQUES = {
    t.id: t for t in (
        Technique("fp8_w8a8_dynamic_linear",
                  "linear layers compute in FP8: weights quantized once at load, "
                  "activations quantized per call with scales computed at runtime",
                  {"scope": "transformer"}),
        Technique("fp8_weight_only_linear",
                  "weights stored in FP8, dequantized; activations and math stay 16-bit",
                  {"scope": "transformer"}),
    )
}


@dataclasses.dataclass(frozen=True)
class Evidence:
    """What the adapter observed in the live engine, in engine-neutral terms.

    `quantizable`: linear layers the engine built to accept quantization.
    The other counts split them by what they actually became. `runtime_*`
    count what executed during the measured requests."""

    quantizable: int
    fp8_w8a8: int
    fp8_weight_only: int
    unquantized: int
    not_quantizable: int
    runtime_w8a8_gemms: int
    runtime_weight_only_gemms: int
    runtime_activation_quants: int


@dataclasses.dataclass(frozen=True)
class Implementation:
    id: str
    technique: str
    kind: str  # "native" | "generic"
    engine: str
    requires: frozenset[str]  # capabilities
    lifecycle: frozenset[str]  # mutates_model, dynamic_in_forward, request_state
    execution: frozenset[str]  # execution requirements (ADR 0013)
    owns: frozenset[str]
    engaged: Callable[[Evidence], tuple[bool, str]]


def _w8a8_engaged(e: Evidence) -> tuple[bool, str]:
    if e.quantizable == 0:
        return False, "the engine built no quantizable linear layer"
    if e.fp8_weight_only or e.runtime_weight_only_gemms:
        return False, (f"weight-only FP8 ran instead ({e.fp8_weight_only} layers, "
                       f"{e.runtime_weight_only_gemms} GEMMs): a different technique")
    if e.fp8_w8a8 != e.quantizable:
        return False, f"partial: {e.fp8_w8a8} of {e.quantizable} quantizable layers are FP8 W8A8"
    if e.runtime_w8a8_gemms == 0 or e.runtime_activation_quants == 0:
        return False, "FP8 layers present, but no FP8 GEMM or activation quantization executed"
    return True, (f"{e.fp8_w8a8}/{e.quantizable} layers FP8 W8A8; "
                  f"{e.runtime_w8a8_gemms} FP8 GEMMs and {e.runtime_activation_quants} "
                  f"activation quantizations executed")


IMPLEMENTATIONS = (
    Implementation(
        id="sglang-native-fp8-w8a8", technique="fp8_w8a8_dynamic_linear", kind="native",
        engine="sglang", requires=frozenset({"engine_feature.fp8_w8a8_dynamic_linear"}),
        lifecycle=frozenset({"mutates_model"}), execution=frozenset(),
        owns=frozenset({"linear_layers"}), engaged=_w8a8_engaged),
    # Not run here: shows a second engine's native candidate for the same
    # technique is just another row (vLLM-Omni's online fp8, per-tensor weights).
    Implementation(
        id="vllm-omni-native-fp8-w8a8", technique="fp8_w8a8_dynamic_linear", kind="native",
        engine="vllm-omni", requires=frozenset({"engine_feature.fp8_w8a8_dynamic_linear"}),
        lifecycle=frozenset({"mutates_model"}), execution=frozenset(),
        owns=frozenset({"linear_layers"}), engaged=_w8a8_engaged),
)


@dataclasses.dataclass(frozen=True)
class Target:
    """What the adapter read back from the machine and engine, not requested."""

    engine: str
    gpu_capability: tuple[int, int]
    cuda_version: tuple[int, int]
    capabilities: frozenset[str]  # what this engine build can provide
    env_overrides: Mapping[str, str]  # engine env vars that change native behavior


class SGLangAdapter:
    """The only place that knows SGLang's flag, its GPU rules and its
    silent fallbacks (traced at 8ca82118e; see README)."""

    engine = "sglang"
    # Engine settings that would make `quantization="fp8"` run a different
    # numerical method without an error.
    _FORCES_WEIGHT_ONLY = ("SGLANG_FORCE_FP8_MARLIN",)
    _DROPS_LAYERS = ("SGLANG_FP8_IGNORED_LAYERS",)

    def supported(self, impl: Implementation, target: Target) -> tuple[bool, str]:
        missing = sorted(impl.requires - target.capabilities)
        return (not missing, f"missing {missing}" if missing else "all capabilities resolve")

    def feasible(self, impl: Implementation, target: Target) -> tuple[bool, str]:
        # SGLang applies no capability check to the DiT: below the native
        # FP8 GEMM it silently switches to Marlin weight-only FP8.
        if target.gpu_capability < (8, 9):
            return False, f"sm_{''.join(map(str, target.gpu_capability))} has no FP8 GEMM; SGLang would run weight-only"
        if target.gpu_capability == (8, 9) and target.cuda_version < (12, 4):
            return False, "sm_89 needs CUDA 12.4 for SGLang's FP8 GEMM"
        for var in self._FORCES_WEIGHT_ONLY + self._DROPS_LAYERS:
            if target.env_overrides.get(var):
                return False, f"{var} is set and changes the method or its coverage"
        return True, "native W8A8 FP8 GEMM available"

    def configure(self, impl: Implementation, params: Mapping[str, object]) -> dict:
        """Capability -> SGLang's own settings. The Technique never sees these."""
        if "engine_feature.fp8_w8a8_dynamic_linear" in impl.requires:
            if params.get("scope", "transformer") != "transformer":
                raise ValueError("SGLang's global flag quantizes the transformer only")
            return {"quantization": "fp8"}
        return {}


def resolve(technique_id: str, target: Target) -> list[Implementation]:
    """Candidates for a technique on this target's engine, in registry order."""
    return [i for i in IMPLEMENTATIONS if i.technique == technique_id and i.engine == target.engine]


def plan(technique_id: str, target: Target, adapter: SGLangAdapter, *, skip_feasibility=False):
    """Returns (implementation, settings, status, reason); settings is None
    unless the implementation should be launched."""
    technique = TECHNIQUES[technique_id]
    candidates = resolve(technique_id, target)
    if not candidates:
        return None, None, UNSUPPORTED, f"no implementation of {technique_id} for {target.engine}"
    impl = candidates[0]
    ok, why = adapter.supported(impl, target)
    if not ok:
        return impl, None, UNSUPPORTED, why
    ok, why = adapter.feasible(impl, target)
    if not ok and not skip_feasibility:
        return impl, None, INFEASIBLE, why
    return impl, adapter.configure(impl, technique.params), None, why


def verdict(impl: Implementation, evidence: Evidence) -> tuple[str, str]:
    engaged, why = impl.engaged(evidence)
    return (VALID if engaged else FAILED_TO_ENGAGE), why
