"""In-batch valid-token compaction (unpadding), T4-optimal GQA SDPA,
fused MLP wiring, and custom low-overhead decoder-layer activation checkpointing."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import torch
from torch.nn import functional as F

_TLS = threading.local()


@dataclass
class PackedBatchContext:
    orig_batch: int
    orig_seq: int
    seqlens: list[int]
    segments: list[tuple[int, int]]
    valid_idx: torch.Tensor | None
    packed_input_ids: torch.Tensor
    packed_position_ids: torch.Tensor | None


def get_active_packed_context() -> PackedBatchContext | None:
    return getattr(_TLS, "packed_ctx", None)


@contextmanager
def packed_context_scope(ctx: PackedBatchContext | None):
    prev = getattr(_TLS, "packed_ctx", None)
    _TLS.packed_ctx = ctx
    try:
        yield
    finally:
        _TLS.packed_ctx = prev


def maybe_build_packed_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None = None,
) -> PackedBatchContext | None:
    """Compact non-pad tokens across a right-padded batch into (1, N_valid) so every linear,
    norm, MLP, RoPE, SDPA and output-head projection skips all padding compute & memory."""
    if attention_mask is None or not input_ids.is_cuda or input_ids.ndim != 2:
        return None
    b, s = input_ids.shape
    if b == 0 or s == 0 or attention_mask.shape != (b, s):
        return None

    mask_bool = attention_mask.ne(0)
    seqlens_gpu = mask_bool.sum(dim=1, dtype=torch.int32)
    pos_s = torch.arange(s, device=input_ids.device, dtype=torch.int32)
    # Verify monotonic right-padding (1,1,...,1,0,...,0) + fetch row lengths in 1 tiny D2H copy
    has_hole = (mask_bool != (pos_s.unsqueeze(0) < seqlens_gpu.unsqueeze(1))).any().to(torch.int32)
    meta = torch.cat([seqlens_gpu, has_hole.unsqueeze(0)]).tolist()
    seqlens, mismatch = meta[:-1], meta[-1]

    if mismatch != 0:
        return None

    total_valid = sum(seqlens)
    if total_valid <= 0:
        return None

    if total_valid == b * s:
        # Zero padding in this batch: keep (B, S) shape, just drop the 4D mask so SDPA uses fast causal kernel!
        return PackedBatchContext(
            orig_batch=b,
            orig_seq=s,
            seqlens=seqlens,
            segments=[],
            valid_idx=None,
            packed_input_ids=input_ids,
            packed_position_ids=position_ids,
        )

    valid_idx = torch.nonzero(mask_bool.reshape(-1), as_tuple=False).squeeze(-1)
    packed_input_ids = input_ids.reshape(-1).index_select(0, valid_idx).unsqueeze(0)

    if position_ids is not None:
        base_pos = position_ids.expand(b, s)
    else:
        base_pos = pos_s.to(torch.long).unsqueeze(0).expand(b, s)
    packed_position_ids = base_pos.reshape(-1).index_select(0, valid_idx).unsqueeze(0)

    segments: list[tuple[int, int]] = []
    cursor = 0
    for length in seqlens:
        if length > 0:
            segments.append((cursor, cursor + length))
            cursor += length

    return PackedBatchContext(
        orig_batch=b,
        orig_seq=s,
        seqlens=seqlens,
        segments=segments,
        valid_idx=valid_idx,
        packed_input_ids=packed_input_ids,
        packed_position_ids=packed_position_ids,
    )


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h_kv, slen, d = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(b, h_kv, n_rep, slen, d)
        .reshape(b, h_kv * n_rep, slen, d)
    )


def _fast_qwen2_attention_forward(
    attn_mod,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    packed_ctx: PackedBatchContext | None,
) -> torch.Tensor:
    from .rope import fused_qk_rope

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_mod.head_dim)

    q = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    q, k = fused_qk_rope(q, k, cos, sin)

    dropout_p = attn_mod.attention_dropout if attn_mod.training else 0.0
    n_rep = attn_mod.num_key_value_groups
    scale = attn_mod.scaling

    if packed_ctx is not None and packed_ctx.valid_idx is not None:
        # Packed (1, H, N_valid, D): run fused EFFICIENT_ATTENTION per segment with repeat_kv
        # (avoids SM75 enable_gqa fallback to unfused math SDPA and avoids O(B*S_max^2) padding FLOPs!)
        out_parts = []
        for s0, s1 in packed_ctx.segments:
            q_seg = q[:, :, s0:s1, :]
            k_seg = _repeat_kv(k[:, :, s0:s1, :], n_rep)
            v_seg = _repeat_kv(v[:, :, s0:s1, :], n_rep)
            seg_out = F.scaled_dot_product_attention(
                q_seg,
                k_seg,
                v_seg,
                attn_mask=None,
                dropout_p=dropout_p,
                scale=scale,
                is_causal=(s1 - s0) > 1,
            )
            out_parts.append(seg_out)
        if len(out_parts) == 1:
            attn_out = out_parts[0]
        else:
            attn_out = torch.cat(out_parts, dim=2)
    else:
        # Full rectangular batch with zero padding (attention_mask is None):
        # Explicitly repeat_kv so Turing SM75 dispatches to fused cutlassF EFFICIENT_ATTENTION
        # instead of enable_gqa=True falling back to slow O(N^2) math SDPA!
        k_exp = _repeat_kv(k, n_rep)
        v_exp = _repeat_kv(v, n_rep)
        is_causal = attention_mask is None and q.shape[2] > 1
        attn_out = F.scaled_dot_product_attention(
            q,
            k_exp,
            v_exp,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            scale=scale,
            is_causal=is_causal,
        )

    attn_out = attn_out.transpose(1, 2).reshape(*input_shape, -1)
    return attn_mod.o_proj(attn_out)


def _fast_qwen2_mlp_forward(mlp_mod, hidden_states: torch.Tensor) -> torch.Tensor:
    from .nf4_linear import TritonNF4SwiGLUMLPFunc, _resolve_bnb_compute_dtype, can_triton_dequant

    gate_mod = mlp_mod.gate_proj
    up_mod = mlp_mod.up_proj
    down_mod = mlp_mod.down_proj

    # Fast fused 4-bit MLP path when gate/up/down are frozen bnb 4-bit linears without bias/LoRA
    if (
        gate_mod.__class__.__name__ == "Linear4bit"
        and up_mod.__class__.__name__ == "Linear4bit"
        and down_mod.__class__.__name__ == "Linear4bit"
        and gate_mod.bias is None
        and up_mod.bias is None
        and down_mod.bias is None
    ):
        from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module

        fix_4bit_weight_quant_state_from_module(gate_mod)
        fix_4bit_weight_quant_state_from_module(up_mod)
        fix_4bit_weight_quant_state_from_module(down_mod)
        g_qs = getattr(gate_mod.weight, "quant_state", None)
        u_qs = getattr(up_mod.weight, "quant_state", None)
        d_qs = getattr(down_mod.weight, "quant_state", None)
        if (
            can_triton_dequant(gate_mod.weight.data, g_qs)
            and can_triton_dequant(up_mod.weight.data, u_qs)
            and can_triton_dequant(down_mod.weight.data, d_qs)
        ):
            comp_dtype = _resolve_bnb_compute_dtype(gate_mod, hidden_states)
            if comp_dtype in (torch.float16, torch.bfloat16, torch.float32):
                return TritonNF4SwiGLUMLPFunc.apply(
                    hidden_states,
                    gate_mod.weight.data,
                    g_qs,
                    up_mod.weight.data,
                    u_qs,
                    down_mod.weight.data,
                    d_qs,
                    comp_dtype,
                )

    return mlp_mod(hidden_states)


def _decoder_layer_inner(
    layer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    packed_ctx: PackedBatchContext | None,
) -> torch.Tensor:
    from .rmsnorm import fused_add_rmsnorm, rmsnorm

    residual = hidden_states
    normed = rmsnorm(
        hidden_states,
        layer.input_layernorm.weight,
        layer.input_layernorm.variance_epsilon,
    )
    attn_delta = _fast_qwen2_attention_forward(
        layer.self_attn,
        normed,
        position_embeddings,
        attention_mask,
        packed_ctx,
    )
    residual, post_normed = fused_add_rmsnorm(
        residual,
        attn_delta,
        layer.post_attention_layernorm.weight,
        layer.post_attention_layernorm.variance_epsilon,
    )
    mlp_delta = _fast_qwen2_mlp_forward(layer.mlp, post_normed)
    if mlp_delta.dtype != residual.dtype:
        mlp_delta = mlp_delta.to(residual.dtype)
    return residual + mlp_delta


class DecoderLayerCheckpointFunc(torch.autograd.Function):
    """Low-overhead custom C++-autograd activation checkpointing per Qwen2DecoderLayer.
    Saves only the layer input [1, N_valid, D] (~2.6 MiB) + exact CUDA RNG state for LoRA dropout."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        layer,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        packed_ctx: PackedBatchContext | None,
        *trainable_params: torch.Tensor,
    ):
        ctx.rng_state = torch.cuda.get_rng_state(hidden_states.device)
        ctx.autocast_enabled = torch.is_autocast_enabled("cuda")
        ctx.autocast_dtype = torch.get_autocast_dtype("cuda")
        ctx.layer = layer
        ctx.attention_mask = attention_mask
        ctx.packed_ctx = packed_ctx
        ctx.save_for_backward(hidden_states.detach(), cos, sin, *trainable_params)
        ctx.set_materialize_grads(False)

        with torch.no_grad():
            return _decoder_layer_inner(
                layer,
                hidden_states.detach(),
                (cos, sin),
                attention_mask,
                packed_ctx,
            )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor | None):
        num_params = len(ctx.saved_tensors) - 3
        if grad_output is None:
            return (None, None, None, None, None, None, *([None] * num_params))

        saved = ctx.saved_tensors
        hidden_saved, cos, sin = saved[0], saved[1], saved[2]
        trainable_params = saved[3:]

        need_h = ctx.needs_input_grad[0]
        h_in = hidden_saved.detach().requires_grad_(need_h)

        dev = h_in.device
        cur_rng = torch.cuda.get_rng_state(dev)
        torch.cuda.set_rng_state(ctx.rng_state, dev)
        try:
            with torch.enable_grad(), torch.amp.autocast(
                "cuda", enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype
            ):
                out = _decoder_layer_inner(
                    ctx.layer,
                    h_in,
                    (cos, sin),
                    ctx.attention_mask,
                    ctx.packed_ctx,
                )
        finally:
            torch.cuda.set_rng_state(cur_rng, dev)

        targets: list[torch.Tensor] = []
        if need_h:
            targets.append(h_in)
        active_param_indices: list[int] = []
        for idx, param in enumerate(trainable_params):
            if ctx.needs_input_grad[6 + idx]:
                targets.append(param)
                active_param_indices.append(idx)

        if not targets:
            return (None, None, None, None, None, None, *([None] * num_params))

        grads = torch.autograd.grad(out, targets, grad_output, allow_unused=True)

        ptr = 0
        d_hidden = None
        if need_h:
            d_hidden = grads[0]
            ptr = 1

        param_grads = [None] * num_params
        for idx in active_param_indices:
            param_grads[idx] = grads[ptr]
            ptr += 1

        return (d_hidden, None, None, None, None, None, *param_grads)


def install_model_layer_optimizations(causal_lm: torch.nn.Module) -> None:
    """Install fast unpadded / T4-SDPA / checkpointed decoder-layer execution on Qwen2."""
    base_model = getattr(causal_lm, "model", None)
    layers = getattr(base_model, "layers", None)
    if base_model is None or layers is None:
        return
    if causal_lm.__class__.__name__ != "Qwen2ForCausalLM":
        return

    for layer in layers:
        if getattr(layer, "_soup_fast_layer_installed", False):
            continue
        orig_layer_fwd = layer.forward
        # Cache trainable Parameter list on the layer once after PEFT setup completes
        trainable_params = tuple(p for p in layer.parameters() if p.requires_grad)

        def _make_layer_fwd(_layer, _orig_fwd, _trainable):
            def _layer_fwd(
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                past_key_values=None,
                use_cache: bool | None = False,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                **kwargs,
            ):
                packed_ctx = get_active_packed_context()
                if (
                    not hidden_states.is_cuda
                    or past_key_values is not None
                    or use_cache
                    or position_embeddings is None
                    or kwargs.get("output_attentions", False)
                    or getattr(_layer, "sliding_window", None) is not None
                    or getattr(_layer.self_attn, "sliding_window", None) is not None
                ):
                    return _orig_fwd(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

                mask_arg = None if packed_ctx is not None else attention_mask
                if (
                    _layer.training
                    and torch.is_grad_enabled()
                    and (hidden_states.requires_grad or len(_trainable) > 0)
                ):
                    cos, sin = position_embeddings
                    return DecoderLayerCheckpointFunc.apply(
                        hidden_states,
                        _layer,
                        cos,
                        sin,
                        mask_arg,
                        packed_ctx,
                        *_trainable,
                    )

                return _decoder_layer_inner(
                    _layer,
                    hidden_states,
                    position_embeddings,
                    mask_arg,
                    packed_ctx,
                )

            return _layer_fwd

        layer.forward = _make_layer_fwd(layer, orig_layer_fwd, trainable_params)
        layer._soup_fast_layer_installed = True
