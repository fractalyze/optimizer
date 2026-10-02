"""SPIKE: SGLang plugin entry point for experiment 002.

Registers the adapter's trunk hooks at the seams each Binding names. Hooks
for every Binding are registered; only the model that is actually loaded ever
reaches its own.
"""

from __future__ import annotations

from opt_trunk_probe import adapter
from opt_trunk_probe.bindings import BINDINGS


def register():
    from sglang.multimodal_gen.runtime.platforms.plugins import HookType, plugin_hook

    for binding in BINDINGS:
        plugin_hook(binding.dit, HookType.AROUND)(adapter.around_dit(binding))
        for kind, target in binding.blocks.items():
            plugin_hook(target, HookType.AROUND)(adapter.around_block(binding, kind))
        plugin_hook(binding.exit, HookType.AROUND)(adapter.around_exit(binding))
    adapter._log(dict(event="registered", bindings=[b.name for b in BINDINGS]))
