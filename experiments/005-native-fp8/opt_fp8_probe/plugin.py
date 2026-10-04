"""SPIKE: SGLang plugin entry point for experiment 005's engagement evidence.

Registers observation hooks on engine-level classes and functions only. It can
be allow-listed next to experiment 002's trunk probe (SGLANG_PLUGINS takes a
comma-separated list) to test FP8 with TeaCache-style control.
"""

from __future__ import annotations

from opt_fp8_probe import evidence


def register():
    from sglang.multimodal_gen.runtime.platforms.plugins import HookType, plugin_hook

    mm = evidence.MM
    plugin_hook(f"{mm}.loader.fsdp_load.process_model_weights_after_loading",
                HookType.AROUND)(evidence.after_weights_processed)
    plugin_hook(f"{mm}.layers.quantization.fp8.Fp8LinearMethod.apply",
                HookType.AROUND)(evidence.around_fp8_apply)
    for name, target in evidence.KERNELS.items():
        plugin_hook(target, HookType.AROUND)(evidence.counting(name))
    plugin_hook(f"{mm}.pipelines_core.stages.denoising.DenoisingStage.forward",
                HookType.AROUND)(evidence.around_denoising)
    evidence._log(dict(event="registered"))
