"""Padding-free training step for Qwen2.

Soup trains on batches of 4 sequences padded to the longest one. On this data 43% of the
positions are padding (mean length 207 tokens, longest of four ~360), and every layer computes
them: projections, LoRA, norms, MLP, lm_head, loss, and the memory for all of it. The task
explicitly allows removing padding inside a batch (never mixing rows across batches).

`Qwen2ForCausalLM.forward` is replaced for the training step: the N real tokens are gathered
into an (N, hidden) matrix and go through the unchanged decoder layers (they are shape agnostic
except attention, which gets the token -> (row, position) map and the sequence lengths, see
attention.py). The loss is the same cross-entropy on the same
rows (padding rows had label -100). The logits TRL and Soup read for their metrics are handed
over in the padded (B, T, vocab) layout, detached, with zeros at the padding positions, which
those metrics ignore (attention mask / -100 labels).

Anything that is not that exact training step (evaluation, a cache, labels missing, another
attention implementation, ...) goes through the original forward.
"""

import torch

from .kernels import fused_cross_entropy
from .offload import OFFLOADER
from .packed_logits import PackedLogits


def _autocast_fp16():
    try:
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
    except (TypeError, RuntimeError):  # older torch without the device argument
        return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16


def _fp16_embeddings(model, lm_head):
    """The embedding table (233M weights, tied with lm_head) is fp32 in Soup's 4-bit setup
    because peft's kbit preparation casts every non-quantized weight up; but it was loaded from a
    fp16 checkpoint, so every value is exactly representable in fp16. Storing it as fp16 loses
    nothing (checked, per chunk) and drops 467 MiB, plus the fp16 copy autocast made of it for the
    lm_head matmul (445 MiB). Only a frozen fp32 table is touched."""
    embed = model.embed_tokens
    if getattr(embed, "_fp16_checked", False):
        return
    embed._fp16_checked = True
    _flatten_quant_states(model)
    for module in (embed, lm_head):
        weight = module.weight
        if weight.dtype != torch.float32 or weight.requires_grad:
            continue
        half = weight.data.to(torch.float16)
        if all(torch.equal(half[i:i + 16384].float(), weight.data[i:i + 16384])
               for i in range(0, half.shape[0], 16384)):
            weight.data = half


def _flatten_quant_states(model):
    """bitsandbytes' 4-bit weights use double quantization: the per-block scales (`absmax`) are
    themselves stored 8-bit and decoded (a dequantize kernel plus an add kernel) inside every
    single matmul, forward and backward, 2 x 196 times per step, always to the same fp32 values.
    Decode them once; bitsandbytes' own CPU path does exactly this permanent conversion
    (`quant_state.nested = False`), and its non-nested GPU path then reads them directly."""
    try:
        import bitsandbytes as bnb
        from bitsandbytes import functional as bnb_functional
    except ImportError:
        return
    for module in model.modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        state = getattr(module.weight, "quant_state", None)
        if state is None or not getattr(state, "nested", False):
            continue
        absmax = bnb_functional.dequantize_blockwise(state.absmax, state.state2)
        absmax += state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()
        state.absmax = absmax
        state.nested = False


def patch_causal_lm_packed():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2ForCausalLM.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        model, config = self.model, self.config
        eligible = (
            self.training and torch.is_grad_enabled() and labels is not None
            and input_ids is not None and inputs_embeds is None and past_key_values is None
            and not use_cache and isinstance(attention_mask, torch.Tensor)
            and attention_mask.dim() == 2 and isinstance(logits_to_keep, int)
            and logits_to_keep == 0 and input_ids.is_cuda
            and getattr(config, "_attn_implementation", None) == "sdpa"
            and not getattr(model, "has_sliding_layers", False)
            and config.attention_dropout == 0.0 and _autocast_fp16()
            and model.norm.weight.dtype == torch.float32
            and not kwargs.get("output_attentions") and not kwargs.get("output_hidden_states")
            and "shift_labels" not in kwargs and "packed" not in kwargs)
        if eligible:
            real = attention_mask.to(torch.bool)
            shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
            # the loss must only have terms at real positions, which is what the packed step
            # computes (with left padding, the padding position before the first token can carry
            # a label; with right padding, as in Soup, it never does)
            eligible = shifted.shape == real.shape and not bool(((shifted != -100) & ~real).any())
        if not eligible:
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)

        _fp16_embeddings(model, self.lm_head)
        batch, seq = input_ids.shape
        device = input_ids.device
        rows = torch.nonzero(real.reshape(-1), as_tuple=True)[0]  # flat (b * seq + t) of real tokens
        b_idx = (rows // seq).to(torch.int32)
        t_idx = (rows % seq).to(torch.int32)

        hidden = model.embed_tokens(input_ids.reshape(-1)[rows]).to(torch.float32)
        if position_ids is None:
            position_ids = torch.arange(seq, device=device).unsqueeze(0)
        cos, sin = model.rotary_emb(hidden, position_ids)
        # With padding on the right, each sequence's real tokens are one contiguous block of the
        # packed rows, and a real query only ever looks at real keys of its own sequence (the
        # padding keys lie in its future): attention is plain causal attention of each block on
        # its own, with no mask and no padding positions at all (attention.varlen_attention).
        # Any other padding layout keeps the padded layout and transformers' (B, 1, T, T) mask.
        lengths = real.sum(dim=1)
        right_padded = torch.equal(real, torch.arange(seq, device=device)[None, :] < lengths[:, None])
        if right_padded:
            mask4, row_lengths = None, lengths.tolist()
        else:
            causal = torch.ones(seq, seq, dtype=torch.bool, device=device).tril_()
            mask4, row_lengths = causal[None, None] & real[:, None, None, :], None
        packed = (b_idx, t_idx, batch, seq, row_lengths)
        # what the layers keep for the backward goes to pinned host memory (offload.py)
        OFFLOADER.begin(device)
        try:
            with torch.autograd.graph.saved_tensors_hooks(OFFLOADER.pack, OFFLOADER.unpack):
                for layer in model.layers[: config.num_hidden_layers]:
                    hidden = layer(hidden, attention_mask=mask4, position_ids=position_ids,
                                   position_embeddings=(cos, sin), packed=packed)
                hidden = model.norm(hidden)
        finally:
            OFFLOADER.end()
        logits = self.lm_head(hidden)  # (N, vocab), fp16 under autocast

        targets = shifted.reshape(-1)[rows]
        items = kwargs.get("num_items_in_batch")
        if items is None:
            denom = (targets != -100).sum().to(torch.float32)
        else:
            denom = torch.as_tensor(items, device=device, dtype=torch.float32)
        # TRL and Soup read the logits for their metrics inside compute_loss, before the backward
        # pass; the backward then writes the loss gradient over them (no second N x vocab buffer)
        loss = fused_cross_entropy(logits, targets, denom, inplace_grad=True)

        return CausalLMOutputWithPast(loss=loss, logits=PackedLogits(logits.detach(), rows, batch,
                                                                       seq))

    modeling_qwen2.Qwen2ForCausalLM.forward = forward
