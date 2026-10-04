"""Engine packages: each declares its catalog entries and owns its runtime policy."""

from __future__ import annotations

from optimizer.core.feasibility import Evaluator
from optimizer.engines.sglang import catalog as sglang_catalog
from optimizer.engines.sglang import feasibility as sglang_feasibility


def evaluators() -> dict[str, Evaluator]:
    """Feasibility evaluators by engine. vLLM-Omni has none yet: experiment 004
    ran it eager only, too little to judge any other mode."""
    return {sglang_catalog.ENGINE: sglang_feasibility.measured()}
