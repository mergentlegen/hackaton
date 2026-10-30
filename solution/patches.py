"""Patches that send Soup's Qwen2 training step through our kernels (see kernels.py)."""

import torch

from .kernels import fused_cross_entropy, silu_mul
from .attention import pack_rows, qkv_rotate
from .logit_stats import entropy_from_logits, row_cross_entropy
from .rmsnorm import rms_norm


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


def _autocast_fp16():
    try:
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
    except (TypeError, RuntimeError):  # older torch without the device argument
        return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16


def patch_qwen2_attention():
    """Qwen2Attention with the rotation, the fp16 casts and the GQA key/value expansion done by
    the kernels of attention.py. Anything else (a cache, a sliding window, no mask, another
    attention implementation, fp16 stream) goes through the original code."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2Attention.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                **kwargs):
        packed = kwargs.pop("packed", None)
        if packed is None:  # not the packed training step (packed.py): the original code
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values, **kwargs)
        b_idx, t_idx, batch, seq = packed
        cos, sin = position_embeddings
        config = self.config
        query, key, value = qkv_rotate(self.q_proj(hidden_states), self.k_proj(hidden_states),
                                       self.v_proj(hidden_states), cos, sin, b_idx, t_idx, batch,
                                       seq, config.num_attention_heads,
                                       config.num_key_value_heads)
        attn = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, scale=self.scaling,
            is_causal=False)
        return self.o_proj(pack_rows(attn, b_idx, t_idx)), None

    modeling_qwen2.Qwen2Attention.forward = forward


def patch_qwen2_rmsnorm():
    """Qwen2RMSNorm as one Triton pass each way (the stream is fp32 in Soup's 4-bit setup)."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        weight = self.weight
        if (hidden_states.is_cuda and hidden_states.dtype == torch.float32
                and weight.dtype == torch.float32 and weight.is_cuda):
            # in the fp16 autocast step the consumers cast to fp16 anyway: hand it over as fp16
            return rms_norm(hidden_states, weight, self.variance_epsilon,
                            to_half=_autocast_fp16())
        return original(self, hidden_states)

    modeling_qwen2.Qwen2RMSNorm.forward = forward


def patch_logit_metrics():
    """TRL's per-token entropy and Soup's per-row loss are read from the fp16 logits; run them as
    single passes over the logits (same quantities, see logit_stats.py)."""
    try:
        from trl.trainer import sft_trainer
        from trl.trainer import utils as trl_utils
    except ImportError:
        pass
    else:
        original = trl_utils.entropy_from_logits

        def entropy(logits, chunk_size=128):
            if not isinstance(logits, torch.Tensor) or not logits.is_cuda:
                return original(logits, chunk_size)
            return entropy_from_logits(logits, chunk_size)

        trl_utils.entropy_from_logits = entropy
        if hasattr(sft_trainer, "entropy_from_logits"):
            sft_trainer.entropy_from_logits = entropy
    try:
        from soup_cli.trainer import rewind_hf
    except ImportError:
        return
    original_rows = rewind_hf.row_losses

    def row_losses(logits, labels):
        if not isinstance(logits, torch.Tensor) or not logits.is_cuda or logits.dim() != 3:
            return original_rows(logits, labels)
        with torch.no_grad():
            return row_cross_entropy(logits, labels)

    rewind_hf.row_losses = row_losses


def patch_trainer_flops():
    """transformers' Trainer calls `floating_point_ops` after every step, and that walks all ~930
    modules of the PEFT model twice (`num_parameters(exclude_embeddings=True)`) to get a number
    that never changes. Same number, computed once per trainer (~15 ms of every step)."""
    import transformers

    original = transformers.Trainer.floating_point_ops

    def floating_point_ops(self, inputs):
        model = self.model
        name = getattr(model, "main_input_name", "input_ids")
        if name not in inputs or not hasattr(model, "num_parameters"):
            return original(self, inputs)
        params = self.__dict__.get("_non_embedding_parameters")
        if params is None:
            params = model.num_parameters(exclude_embeddings=True)
            self.__dict__["_non_embedding_parameters"] = params
        return 6 * inputs[name].numel() * params

    transformers.Trainer.floating_point_ops = floating_point_ops
