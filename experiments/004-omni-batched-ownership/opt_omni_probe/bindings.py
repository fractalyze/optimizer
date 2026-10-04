"""SPIKE: the (vLLM-Omni × Qwen-Image) Binding for the transformer trunk.

Same questions as experiment 002's SGLang Bindings, answered for vLLM-Omni's
own Qwen-Image transformer. Qwen-Image-2512 resolves to it through
`model_index.json` `_class_name` = "QwenImagePipeline"
(`vllm_omni/diffusion/data.py:1586`, `registry.py:38`).

Code locations are at vLLM-Omni 68003cf6a,
vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py.
"""

from __future__ import annotations

import torch.nn.functional as F

_M = "vllm_omni.diffusion.models.qwen_image.qwen_image_transformer"


def _modulated(x, eps, scale, shift):
    """LayerNorm then modulate, in fp32 and outside the model's kernels."""
    out = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
    return out * (1 + scale.float()) + shift.float()


class VllmOmniQwenImageBinding:
    """QwenImageTransformer2DModel.forward (:1145). Trunk = the
    `transformer_blocks` loop (:1264): dual-stream blocks that return
    (encoder_hidden_states, hidden_states) (:968). Exit = `norm_out(hidden_states,
    temb)` (:1279), a diffusers AdaLayerNormContinuous shared with other models,
    so the hook acts only on this model's own instance."""

    name = "vllm-omni-qwen-image"
    dit = f"{_M}.QwenImageTransformer2DModel.forward"
    blocks = {"block": f"{_M}.QwenImageTransformerBlock.forward"}
    exit = "diffusers.models.normalization.AdaLayerNormContinuous.forward"

    @staticmethod
    def exit_module(model):
        return model.norm_out

    @staticmethod
    def overridable(dit_kwargs):
        return True  # no state crosses a Qwen-Image trunk invocation here

    @staticmethod
    def entry(kind, args, kwargs):
        return kwargs["hidden_states"]  # image tokens after img_in: (rows, tokens, dim)

    @staticmethod
    def signal(block, kind, args, kwargs):
        # The block's image modulation is img_mod(temb), first half = norm1's
        # (shift, scale, gate) (:876, :884, :858); img_norm1 is layernorm·(1+scale)+shift.
        mod1 = block.img_mod(kwargs["temb"]).chunk(2, dim=-1)[0]
        shift, scale, _ = mod1.chunk(3, dim=-1)
        return _modulated(kwargs["hidden_states"], block.img_norm1.eps,
                          scale.unsqueeze(1), shift.unsqueeze(1))

    @staticmethod
    def identity(kind, args, kwargs):
        return kwargs["encoder_hidden_states"], kwargs["hidden_states"]


BINDINGS = (VllmOmniQwenImageBinding,)
