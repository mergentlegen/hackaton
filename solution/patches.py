"""Patches that send Soup's Qwen2 training step through our kernels (see kernels.py)."""

import torch

from .kernels import fused_cross_entropy, silu_mul


def _is_4bit_linear(module):
    try:
        import bitsandbytes as bnb
    except ImportError:  # pragma: no cover - bitsandbytes is part of the task environment
        return False
    return isinstance(module, bnb.nn.Linear4bit)


def patch_qwen2_mlp():
    """Qwen2MLP in fp16 end to end.

    The reference casts the fp32 stream to fp16 inside every 4-bit linear and casts each result
    back to fp32, so it keeps three fp32 (tokens, 8960) tensors per layer. Here the input is cast
    once, the three linears run on fp16 (same values), SwiGLU keeps only its two fp16 inputs, and
    the output is cast back to the stream's dtype, so the layer returns the same numbers.
    """
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        gate_proj, up_proj, down_proj = self.gate_proj, self.up_proj, self.down_proj
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if (act != "silu" or not isinstance(x, torch.Tensor) or not x.is_cuda
                or not all(_is_4bit_linear(m) for m in (gate_proj, up_proj, down_proj))):
            return original(self, x)
        compute = getattr(gate_proj, "compute_dtype", None) or torch.float16
        x16 = x.to(compute)
        hidden = silu_mul(gate_proj(x16), up_proj(x16))
        return down_proj(hidden).to(x.dtype)

    modeling_qwen2.Qwen2MLP.forward = forward


def patch_causal_lm_loss():
    """Cross-entropy straight from the logits, without the fp32 copies of transformers' loss."""
    from transformers.loss import loss_utils

    original = loss_utils.ForCausalLMLoss

    def fused_loss(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100,
                   shift_labels=None, **kwargs):
        if not isinstance(logits, torch.Tensor) or not logits.is_cuda or ignore_index != -100:
            return original(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch,
                            ignore_index=ignore_index, shift_labels=shift_labels, **kwargs)
        if shift_labels is None:
            labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
        targets = shift_labels.to(logits.device).reshape(-1)
        if num_items_in_batch is None:
            denom = (targets != ignore_index).sum().to(torch.float32)
        else:
            denom = torch.as_tensor(num_items_in_batch, device=logits.device,
                                    dtype=torch.float32)
        return fused_cross_entropy(logits.reshape(-1, vocab_size), targets, denom)

    loss_utils.ForCausalLMLoss = fused_loss
    for key, value in list(loss_utils.LOSS_MAPPING.items()):
        if value is original:
            loss_utils.LOSS_MAPPING[key] = fused_loss
