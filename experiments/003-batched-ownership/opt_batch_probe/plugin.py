"""SPIKE: SGLang plugin entry point for experiment 003.

Registers the batch-aware adapter's trunk hooks at the seams experiment 002's
Bindings name. Install only one of the two probes' plugins per run (the
SGLANG_PLUGINS allow-list in run.py does that); both hook the same seams.
"""

from __future__ import annotations

from opt_batch_probe import adapter
from opt_trunk_probe.bindings import BINDINGS


def register():
    from sglang.multimodal_gen.runtime.platforms.plugins import HookType, plugin_hook

    for binding in BINDINGS:
        plugin_hook(binding.dit, HookType.AROUND)(adapter.around_dit(binding))
        for kind, target in binding.blocks.items():
            plugin_hook(target, HookType.AROUND)(adapter.around_block(binding, kind))
        plugin_hook(binding.exit, HookType.AROUND)(adapter.around_exit(binding))
    adapter._log(dict(event="registered", bindings=[b.name for b in BINDINGS]))
