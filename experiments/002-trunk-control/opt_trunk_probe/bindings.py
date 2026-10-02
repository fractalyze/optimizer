"""SPIKE: (SGLang × model) Bindings for the transformer trunk.

A Binding answers, for one model as SGLang builds it:
  - which forward is one trunk invocation (the DiT call);
  - where the trunk starts: the first block call, and the image-stream tensor
    entering it;
  - where it ends: the output norm, whose input is the trunk's image-stream
    result;
  - what a block returns when it is told not to compute (its identity);
  - the cache signal: the first block's modulated image input;
  - whether this invocation may be overridden at all.

Everything model-specific in experiment 002 lives in this file, and nothing
else does: payload forms, request state, compile and graph behavior and
engagement counting are the adapter's. At the trunk boundary both models carry
the same thing, image tokens in and image tokens out, which is why the payload
forms need no Binding.

Code locations are at SGLang 8ca82118e, python/sglang/multimodal_gen/runtime/.
"""

from __future__ import annotations

import torch.nn.functional as F

_DITS = "sglang.multimodal_gen.runtime.models.dits"


def _modulated(x, eps, scale, shift=None):
    """LayerNorm then modulate, in fp32 and outside the model's fused kernels,
    so computing the signal never touches the model's own kernel state."""
    out = F.layer_norm(x.float(), (x.shape[-1],), eps=eps) * (1 + scale.float())
    return out if shift is None else out + shift.float()


class SGLangQwenImage21Binding:
    """models/dits/qwen_image21.py: QwenImage21Transformer2DModel.forward
    (:507). Trunk = the `transformer_blocks` loop (:557) over the image stream
    only; text is a prefix KV cache the blocks fill on the first call (prefill)
    and only read afterwards. Exit = `norm_out(images, temb)` (:567)."""

    name = "sglang-qwen-image-2.1"
    dit = f"{_DITS}.qwen_image21.QwenImage21Transformer2DModel.forward"
    blocks = {"block": f"{_DITS}.qwen_image21.QwenImage21TransformerBlock.forward"}
    exit = f"{_DITS}.qwen_image21.QwenImage21OutputNorm.forward"

    @staticmethod
    def exit_module(model):
        return model.norm_out

    @staticmethod
    def overridable(dit_kwargs):
        # The prefill call writes the prefix KV cache every later step reads.
        # Skipping it would leave the cache empty, so it must always compute.
        caches = dit_kwargs.get("prefix_caches")
        return caches is None or all(cache[0] for cache in caches)

    @staticmethod
    def entry(kind, args, kwargs):
        return args[0]  # hidden_states: (B, image tokens, hidden)

    @staticmethod
    def signal(block, kind, args, kwargs):
        # Blocks share one modulation (:501): (scale1, gate1, scale2, gate2);
        # the attention input is norm1(x) * (1 + scale1), no shift (:157).
        return _modulated(args[0], block.img_norm1.eps, args[1][0])

    @staticmethod
    def identity(kind, args, kwargs):
        return args[0]


class SGLangFlux2Binding:
    """models/dits/flux_2.py: Flux2Transformer2DModel.forward (:1630). Trunk =
    the double-stream loop (:1724), the one-time text+image join (:1743), the
    single-stream loop (:1746) and the slice back to image tokens (:1761).
    Blocks may return a pending gated residual (a tuple) that the next block
    or the join materializes. Exit = `norm_out(hidden_states, temb)` (:1764), a
    diffusers AdaLayerNormContinuous shared with other models, so the hook acts
    only on this model's own instance."""

    name = "sglang-flux.2"
    dit = f"{_DITS}.flux_2.Flux2Transformer2DModel.forward"
    blocks = {
        "double": f"{_DITS}.flux_2.Flux2TransformerBlock.forward",
        "single": f"{_DITS}.flux_2.Flux2SingleTransformerBlock.forward",
    }
    exit = "diffusers.models.normalization.AdaLayerNormContinuous.forward"

    @staticmethod
    def exit_module(model):
        return model.norm_out

    @staticmethod
    def overridable(dit_kwargs):
        return True  # no state crosses a FLUX.2 trunk invocation

    @staticmethod
    def entry(kind, args, kwargs):
        return kwargs["hidden_states"]  # image stream after x_embedder

    @staticmethod
    def signal(block, kind, args, kwargs):
        # temb_mod_params_img = ((shift, scale, gate) for attention, (...) for MLP)
        shift, scale, _ = kwargs["temb_mod_params_img"][0]
        return _modulated(kwargs["hidden_states"], block.norm1.eps, scale, shift)

    @staticmethod
    def identity(kind, args, kwargs):
        if kind == "double":
            return kwargs["encoder_hidden_states"], kwargs["hidden_states"]
        return kwargs["hidden_states"]


BINDINGS = (SGLangQwenImage21Binding, SGLangFlux2Binding)
