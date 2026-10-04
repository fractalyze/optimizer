"""SPIKE: SGLangAdapter's engagement evidence for native FP8 (exp 005).

Observation only: nothing here changes what the engine computes. Every hook
sits on an engine-level class or function, the same for every model, so no
Binding is involved. Paths are at SGLang 8ca82118e.

  inventory  after `process_model_weights_after_loading(model)`
             (runtime/loader/fsdp_load.py:485): classify every linear layer
             by what it became, in engine-neutral terms
  runtime    `Fp8LinearMethod.apply` (runtime/layers/quantization/fp8.py:385)
             and the GEMM / activation-quantization kernels it dispatches to
             (srt/layers/quantization/fp8_utils.py)
  request    around `DenoisingStage.forward`: reset counters and peak memory,
             log what the request executed
"""

from __future__ import annotations

import collections
import json
import os
import time

_counts = collections.Counter()

MM = "sglang.multimodal_gen.runtime"
UTILS = "sglang.srt.layers.quantization.fp8_utils"
KERNELS = {  # counted name -> function the engine may dispatch to
    "gemm_cutlass": f"{UTILS}.fp8_scaled_mm",
    "gemm_triton": f"{UTILS}.triton_scaled_mm",
    "gemm_dequant_fallback": f"{UTILS}._apply_fallback_scaled_mm",
    "gemm_marlin_weight_only": f"{MM}.layers.quantization.fp8.apply_fp8_marlin_linear",
    "act_quant_per_token": f"{UTILS}.sglang_per_token_quant_fp8",
}


def _log(record):
    path = os.environ.get("OPT_FP8_LOG")
    if path:
        record.update(pid=os.getpid(), wall=time.time())
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def classify(module):
    """What one linear layer became: (category, params)."""
    import torch
    from sglang.multimodal_gen.runtime.layers.linear import LinearBase
    from sglang.srt.layers.linear import LinearBase as SrtLinearBase

    weight = getattr(module, "weight", None)
    params = 0 if weight is None else int(weight.numel())
    if isinstance(module, (LinearBase, SrtLinearBase)):
        if getattr(module, "quant_config", None) is None:
            return "not_quantizable", params
        method = type(module.quant_method).__name__
        if method == "Fp8LinearMethod":
            if getattr(module.quant_method, "use_marlin", False):
                return "fp8_weight_only", params
            if weight is not None and weight.dtype == torch.float8_e4m3fn:
                return "fp8_w8a8", params
            return "fp8_method_but_16bit_weight", params
        if method.startswith("Unquantized"):
            return "unquantized", params
        return f"other:{method}", params
    if isinstance(module, torch.nn.Linear):
        return "not_quantizable", params
    return None, 0


def after_weights_processed(original, model, *args, **kwargs):
    result = original(model, *args, **kwargs)
    layers, params = collections.Counter(), collections.Counter()
    scales = collections.Counter()
    for module in model.modules():
        category, n = classify(module)
        if category:
            layers[category] += 1
            params[category] += n
            scale = getattr(module, "weight_scale", None)
            if category == "fp8_w8a8" and scale is not None:
                scales["per_tensor" if scale.numel() == 1 else "per_channel"] += 1
    _log(dict(event="inventory", model=type(model).__name__, layers=dict(layers),
              params=dict(params), weight_scales=dict(scales)))
    return result


def around_fp8_apply(original, method, layer, *args, **kwargs):
    _counts["apply_weight_only" if getattr(method, "use_marlin", False) else "apply_w8a8"] += 1
    return original(method, layer, *args, **kwargs)


def counting(name):
    def hook(original, *args, **kwargs):
        _counts[name] += 1
        return original(*args, **kwargs)

    return hook


def around_denoising(original, stage, batch, *args, **kwargs):
    import torch

    _counts.clear()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = original(stage, batch, *args, **kwargs)
    torch.cuda.synchronize()
    _log(dict(event="request", request_id=getattr(batch, "request_id", None),
              warmup=bool(getattr(batch, "is_warmup", False)),
              denoise_s=round(time.perf_counter() - t0, 4),
              peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
              counts=dict(_counts)))
    return out
