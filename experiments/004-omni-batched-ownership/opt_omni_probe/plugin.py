"""SPIKE: vLLM-Omni general-plugin entry point for experiment 004.

vLLM-Omni loads the `vllm_omni.general_plugins` group in every process, and in
the diffusion worker before the model is built (`diffusion_worker.py:1634`), so
class methods patched here are what the layerwise-offload hooks later wrap.
The probe stays inert unless OPT_OMNI_PROBE=1, which is how a run without the
probe is made in the same environment.
"""

from __future__ import annotations

import os

_done = False


def register():
    global _done
    if _done or os.environ.get("OPT_OMNI_PROBE") != "1":
        return
    _done = True
    from opt_omni_probe import adapter
    from opt_omni_probe.bindings import BINDINGS

    for path, hook in adapter.runner_hooks().items():
        adapter.wrap(path, hook)
    for binding in BINDINGS:
        adapter.wrap(binding.dit, adapter.around_dit(binding))
        for kind, target in binding.blocks.items():
            adapter.wrap(target, adapter.around_block(binding, kind))
        adapter.wrap(binding.exit, adapter.around_exit(binding))
    adapter._log(dict(event="registered", bindings=[b.name for b in BINDINGS]))
