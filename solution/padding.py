"""In-batch valid-token compaction (unpadding), direct C++ varlen SM75 CUTLASS attention,
fused QKV + packed SwiGLU MLP wiring, and token-aware selective attention checkpointing."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import torch
from torch.nn import functional as F

_TLS = threading.local()

# Cap extra uncheckpointed-attention memory so peak GPU memory stays ~1.7-1.8 GiB while uncheckpointing more layers
_ATTN_UNCKPT_BUDGET_MIB = 140.0
_ATTN_MIB_PER_TOKEN = 0.0105  # ~10.5 KB/token with TritonFusedAttnQKVRoPEFunc (1-byte dropout masks + 2-head KV)


@dataclass
class PackedBatchContext:
    orig_batch: int
    orig_seq: int
    total_tokens: int
    seqlens: list[int]
    segments: list[tuple[int, int]]
    cu_seqlens: torch.Tensor | None
    max_seqlen: int
    valid_idx: torch.Tensor | None
    packed_input_ids: torch.Tensor
    packed_position_ids: torch.Tensor | None
    unckpt_attn_layers: int


def _compute_unckpt_attn_layers(total_tokens: int) -> int:
    if total_tokens <= 0:
        return 28
    per_layer_mib = max(0.05, total_tokens * _ATTN_MIB_PER_TOKEN)
    return max(1, min(28, int(_ATTN_UNCKPT_BUDGET_MIB // per_layer_mib)))


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

    unckpt = _compute_unckpt_attn_layers(total_valid)

    if total_valid == b * s:
        return PackedBatchContext(
            orig_batch=b,
            orig_seq=s,
            total_tokens=total_valid,
            seqlens=seqlens,
            segments=[],
            cu_seqlens=None,
            max_seqlen=s,
            valid_idx=None,
            packed_input_ids=input_ids,
            packed_position_ids=position_ids,
            unckpt_attn_layers=unckpt,
        )

    valid_idx = torch.nonzero(mask_bool.reshape(-1), as_tuple=False).squeeze(-1)
    packed_input_ids = input_ids.reshape(-1).index_select(0, valid_idx).unsqueeze(0)

    if position_ids is not None:
        base_pos = position_ids.expand(b, s)
    else:
        base_pos = pos_s.to(torch.long).unsqueeze(0).expand(b, s)
    packed_position_ids = base_pos.reshape(-1).index_select(0, valid_idx).unsqueeze(0)

    segments: list[tuple[int, int]] = []
    cu_list = [0]
    cursor = 0
    max_seqlen = 0
    for length in seqlens:
        if length > 0:
            segments.append((cursor, cursor + length))
            cursor += length
            cu_list.append(cursor)
            if length > max_seqlen:
                max_seqlen = length

    cu_seqlens = torch.tensor(cu_list, device=input_ids.device, dtype=torch.int32)

    return PackedBatchContext(
        orig_batch=b,
        orig_seq=s,
        total_tokens=total_valid,
        seqlens=seqlens,
        segments=segments,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        valid_idx=valid_idx,
        packed_input_ids=packed_input_ids,
        packed_position_ids=packed_position_ids,
        unckpt_attn_layers=unckpt,
    )


def _expand_kv_bshd(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return kv.contiguous()
    b, s, h_kv, d = kv.shape
    return kv.unsqueeze(3).expand(b, s, h_kv, n_rep, d).reshape(b, s, h_kv * n_rep, d)


def _repeat_kv_bhsd(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h_kv, slen, d = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(b, h_kv, n_rep, slen, d)
        .reshape(b, h_kv * n_rep, slen, d)
    )


class VarlenCutlassAttentionFunc(torch.autograd.Function):
    """Direct C++ varlen/rectangular SM75 CUTLASS memory-efficient attention (`cutlassF`/`cutlassB`).
    Operates natively in contiguous BSHD layout (zero transposes, zero Python segment loops, zero cat)
    and saves compact 2-head K/V tensors (6x less KV activation memory than pre-expanded GQA)."""

    @staticmethod
    def forward(
        ctx,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int,
        scale: float,
        dropout_p: float,
        is_causal: bool,
    ):
        q_c = q_bshd.contiguous()
        n_rep = q_c.shape[2] // k_bshd.shape[2]
        k_exp = _expand_kv_bshd(k_bshd, n_rep)
        v_exp = _expand_kv_bshd(v_bshd, n_rep)
        custom_mask_type = 1 if is_causal else 0

        if cu_seqlens is not None:
            out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
                q_c,
                k_exp,
                v_exp,
                None,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                float(dropout_p),
                custom_mask_type,
                True,
                scale=scale,
            )
        else:
            s_len = q_c.shape[1]
            out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
                q_c,
                k_exp,
                v_exp,
                None,
                None,
                None,
                s_len,
                s_len,
                float(dropout_p),
                custom_mask_type,
                True,
                scale=scale,
            )

        del k_exp, v_exp
        ctx.save_for_backward(q_c, k_bshd.detach(), v_bshd.detach(), out, lse, seed, offset)
        ctx.cu_seqlens = cu_seqlens
        ctx.max_seqlen = max_seqlen
        ctx.scale = scale
        ctx.dropout_p = float(dropout_p)
        ctx.custom_mask_type = custom_mask_type
        ctx.n_rep = n_rep
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor | None):
        if grad_out is None:
            return None, None, None, None, None, None, None, None

        q_c, k_saved, v_saved, out, lse, seed, offset = ctx.saved_tensors
        n_rep = ctx.n_rep
        k_exp = _expand_kv_bshd(k_saved, n_rep)
        v_exp = _expand_kv_bshd(v_saved, n_rep)
        cu_seqlens = ctx.cu_seqlens
        max_seq = ctx.max_seqlen if cu_seqlens is not None else q_c.shape[1]

        dq, dk_exp, dv_exp, _ = torch.ops.aten._efficient_attention_backward(
            grad_out.contiguous(),
            q_c,
            k_exp,
            v_exp,
            None,
            out,
            cu_seqlens,
            cu_seqlens,
            max_seq,
            max_seq,
            lse,
            ctx.dropout_p,
            seed,
            offset,
            ctx.custom_mask_type,
            False,
            scale=ctx.scale,
        )
        del k_exp, v_exp

        if n_rep > 1:
            b, s, h_kv, d = k_saved.shape
            dk = dk_exp.view(b, s, h_kv, n_rep, d).sum(dim=3)
            dv = dv_exp.view(b, s, h_kv, n_rep, d).sum(dim=3)
        else:
            dk = dk_exp
            dv = dv_exp

        return (
            dq if ctx.needs_input_grad[0] else None,
            dk if ctx.needs_input_grad[1] else None,
            dv if ctx.needs_input_grad[2] else None,
            None,
            None,
            None,
            None,
            None,
        )


def _unwrap_4bit_base(proj_mod):
    """Return (base_bnb_mod, is_clean_peft_or_plain) if proj_mod can participate in fused QKV."""
    cls_name = proj_mod.__class__.__name__
    if hasattr(proj_mod, "lora_A") and hasattr(proj_mod, "base_layer"):
        if proj_mod.disable_adapters or proj_mod.merged:
            return None, False
        for active_adapter in proj_mod.active_adapters:
            if active_adapter in proj_mod.lora_A and active_adapter in proj_mod.lora_variant:
                return None, False
        base = proj_mod.base_layer
    else:
        base = proj_mod

    from .nf4_linear import get_valid_4bit_state

    qs = get_valid_4bit_state(base)
    if qs is None:
        return None, False
    if base.bias is not None and base.bias.requires_grad:
        return None, False
    return base, True


def _apply_lora_adapters_2d(proj_mod, x: torch.Tensor, base_slice_2d: torch.Tensor) -> torch.Tensor:
    if not hasattr(proj_mod, "lora_A"):
        return base_slice_2d
    out_dtype = base_slice_2d.dtype
    res_2d = base_slice_2d
    autocast_on = torch.is_autocast_enabled("cuda")

    for active_adapter in proj_mod.active_adapters:
        if active_adapter not in proj_mod.lora_A:
            continue
        lora_A = proj_mod.lora_A[active_adapter]
        lora_B = proj_mod.lora_B[active_adapter]
        dropout = proj_mod.lora_dropout[active_adapter]
        scaling = float(proj_mod.scaling[active_adapter])

        x_in = x if autocast_on else proj_mod._cast_input_dtype(x, lora_A.weight.dtype)
        h = lora_A(dropout(x_in))
        h_2d = h.reshape(-1, h.shape[-1])
        if h_2d.dtype != out_dtype:
            h_2d = h_2d.to(out_dtype)
        wb = lora_B.weight
        if wb.dtype != out_dtype:
            wb = wb.to(out_dtype)
        if lora_B.bias is not None:
            res_2d = res_2d + (lora_B(h).reshape(-1, res_2d.shape[-1]).to(out_dtype) * scaling)
        else:
            # Out-of-place addmm avoids mutating shared qkv autograd buffer while fusing bias+scale!
            res_2d = torch.addmm(res_2d.contiguous(), h_2d, wb.t(), beta=1.0, alpha=scaling)
    return res_2d


def _extract_single_lora(proj_mod):
    """Extract (wa, wb, scaling, drop_p, is_supported) for fused in-place LoRA execution."""
    if not hasattr(proj_mod, "lora_A"):
        return None, None, 0.0, 0.0, True
    if proj_mod.disable_adapters or proj_mod.merged:
        return None, None, 0.0, 0.0, False
    active = [a for a in proj_mod.active_adapters if a in proj_mod.lora_A]
    if len(active) == 0:
        return None, None, 0.0, 0.0, True
    if len(active) > 1:
        return None, None, 0.0, 0.0, False
    a = active[0]
    if a in getattr(proj_mod, "lora_variant", {}):
        return None, None, 0.0, 0.0, False
    lora_A = proj_mod.lora_A[a]
    lora_B = proj_mod.lora_B[a]
    if lora_A.bias is not None or lora_B.bias is not None:
        return None, None, 0.0, 0.0, False
    dropout = proj_mod.lora_dropout[a]
    drop_p = float(getattr(dropout, "p", 0.0)) if getattr(dropout, "training", False) else 0.0
    scaling = float(proj_mod.scaling[a])
    return lora_A.weight, lora_B.weight, scaling, drop_p, True


def _lora_fwd_inplace(
    normed_2d: torch.Tensor,
    slice_2d: torch.Tensor,
    wa: torch.Tensor | None,
    wb: torch.Tensor | None,
    scale: float,
    drop_p: float,
    comp_dtype: torch.dtype,
):
    if wa is None or wb is None:
        return None, None
    if drop_p > 0.0:
        x_drop, mask = torch._VF._fused_dropout(normed_2d, 1.0 - drop_p)
    else:
        x_drop, mask = normed_2d, None
    wa_c = wa if wa.dtype == comp_dtype else wa.to(comp_dtype)
    wb_c = wb if wb.dtype == comp_dtype else wb.to(comp_dtype)
    h = torch.matmul(x_drop, wa_c.t())
    slice_2d.addmm_(h, wb_c.t(), beta=1.0, alpha=scale)
    return h, mask


def _lora_bwd_branch(
    g_slice: torch.Tensor,
    h: torch.Tensor,
    mask: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    scale: float,
    drop_p: float,
    has_branch: bool,
    need_wa: bool,
    need_wb: bool,
    need_dx: bool,
    normed_2d: torch.Tensor | None,
    comp_dtype: torch.dtype,
):
    if not has_branch:
        return None, None, None
    grad_wa = None
    grad_wb = None
    dx_lora = None
    if need_wb:
        grad_wb = torch.matmul(g_slice.t(), h).to(wb.dtype).mul_(scale)
    if need_wa or need_dx:
        wb_c = wb if wb.dtype == comp_dtype else wb.to(comp_dtype)
        dh = torch.matmul(g_slice, wb_c).mul_(scale)
        if need_wa and normed_2d is not None:
            x_drop = (
                torch._masked_scale(normed_2d, mask, 1.0 / (1.0 - drop_p))
                if drop_p > 0.0
                else normed_2d
            )
            grad_wa = torch.matmul(dh.t(), x_drop).to(wa.dtype)
            del x_drop
        if need_dx:
            wa_c = wa if wa.dtype == comp_dtype else wa.to(comp_dtype)
            dx_drop = torch.matmul(dh, wa_c)
            dx_lora = (
                torch._masked_scale(dx_drop, mask, 1.0 / (1.0 - drop_p))
                if drop_p > 0.0
                else dx_drop
            )
    return grad_wa, grad_wb, dx_lora


class TritonFusedAttnQKVRoPEFunc(torch.autograd.Function):
    """Fused Attention Front-End:
    - RMSNorm written directly in compute_dtype (no fp32->fp16 cast copy).
    - 1-launch triple NF4 dequantization + 1 QKV base GEMM into qkv_2d [M, dq+dk+dv].
    - In-place Q/K/V LoRA via addmm_ on strided qkv_2d slices, saving only 1-byte dropout masks
      and rank-r h tensors (normed_2d and dropped inputs are freed immediately!).
    - Stride-native triton_qk_rope_kernel reading directly from strided qkv_2d in forward and
      writing dq, dk directly into strided grad_qkv_2d in backward (zero SliceBackward0!).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        norm_w: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        wa_q: torch.Tensor | None,
        wb_q: torch.Tensor | None,
        wa_k: torch.Tensor | None,
        wb_k: torch.Tensor | None,
        wa_v: torch.Tensor | None,
        wb_v: torch.Tensor | None,
        norm_eps: float,
        qkv_meta: tuple,
        lora_cfg: tuple,
        comp_dtype: torch.dtype,
        head_dim: int,
    ):
        from .nf4_linear import _dequant_triple_to_workspace
        from .rmsnorm import _norm_launch_cfg, triton_rmsnorm_fwd
        from .rope import _normalize_cos_sin, launch_qk_rope_bshd_strided

        orig_shape = x.shape
        b, s, d = orig_shape[0], orig_shape[1], orig_shape[2]
        m = b * s
        q_base, q_qs, k_base, k_qs, v_base, v_qs, dq, dk, dv, qkv_bias = qkv_meta
        scale_q, drop_q, scale_k, drop_k, scale_v, drop_v = lora_cfg
        h_q = dq // head_dim
        h_k = dk // head_dim

        x_2d = x.reshape(m, d).contiguous()
        normed_2d = torch.empty((m, d), dtype=comp_dtype, device=x.device)
        r = torch.empty((m,), dtype=torch.float32, device=x.device)
        if m > 0:
            block, warps = _norm_launch_cfg(d)
            triton_rmsnorm_fwd[(m,)](
                x_2d,
                x_2d,
                normed_2d,
                normed_2d,
                norm_w,
                r,
                x_2d.stride(0),
                d,
                norm_eps,
                0,
                BLOCK_SIZE=block,
                num_warps=warps,
            )

        w_qkv = _dequant_triple_to_workspace(
            q_base.weight.data,
            q_qs,
            k_base.weight.data,
            k_qs,
            v_base.weight.data,
            v_qs,
            comp_dtype,
        )
        b_comp = (
            None
            if qkv_bias is None
            else (qkv_bias if qkv_bias.dtype == comp_dtype else qkv_bias.to(comp_dtype))
        )
        qkv_2d = F.linear(normed_2d, w_qkv, b_comp)

        hq, mask_q = _lora_fwd_inplace(
            normed_2d, qkv_2d[:, :dq], wa_q, wb_q, scale_q, drop_q, comp_dtype
        )
        hk, mask_k = _lora_fwd_inplace(
            normed_2d, qkv_2d[:, dq : dq + dk], wa_k, wb_k, scale_k, drop_k, comp_dtype
        )
        hv, mask_v = _lora_fwd_inplace(
            normed_2d, qkv_2d[:, dq + dk :], wa_v, wb_v, scale_v, drop_v, comp_dtype
        )
        del normed_2d

        q_in = qkv_2d[:, :dq].view(b, s, h_q, head_dim)
        k_in = qkv_2d[:, dq : dq + dk].view(b, s, h_k, head_dim)
        q_rot = torch.empty((b, s, h_q, head_dim), dtype=comp_dtype, device=x.device)
        k_rot = torch.empty((b, s, h_k, head_dim), dtype=comp_dtype, device=x.device)
        cos_3d, sc_b, sc_s = _normalize_cos_sin(cos, s, head_dim)
        sin_3d, ss_b, ss_s = _normalize_cos_sin(sin, s, head_dim)
        launch_qk_rope_bshd_strided(
            q_in, k_in, cos_3d, sin_3d, q_rot, k_rot,
            sc_b, sc_s, ss_b, ss_s, b, s, h_q, h_k, head_dim, 0,
        )
        v_bshd = qkv_2d[:, dq + dk :].contiguous().view(b, s, h_k, head_dim)
        del qkv_2d

        empty = x_2d[:0, :0]
        ctx.save_for_backward(
            x_2d.detach(),
            norm_w.detach(),
            r,
            cos_3d,
            sin_3d,
            hq if hq is not None else empty,
            mask_q if mask_q is not None else empty,
            hk if hk is not None else empty,
            mask_k if mask_k is not None else empty,
            hv if hv is not None else empty,
            mask_v if mask_v is not None else empty,
            wa_q.detach() if wa_q is not None else empty,
            wb_q.detach() if wb_q is not None else empty,
            wa_k.detach() if wa_k is not None else empty,
            wb_k.detach() if wb_k is not None else empty,
            wa_v.detach() if wa_v is not None else empty,
            wb_v.detach() if wb_v is not None else empty,
        )
        ctx.has_lora = (
            wa_q is not None and wb_q is not None,
            wa_k is not None and wb_k is not None,
            wa_v is not None and wb_v is not None,
        )
        ctx.orig_shape = orig_shape
        ctx.norm_eps = norm_eps
        ctx.qkv_meta = qkv_meta
        ctx.lora_cfg = lora_cfg
        ctx.comp_dtype = comp_dtype
        ctx.head_dim = head_dim
        ctx.set_materialize_grads(False)
        return q_rot, k_rot, v_bshd

    @staticmethod
    def backward(
        ctx,
        grad_q_rot: torch.Tensor | None,
        grad_k_rot: torch.Tensor | None,
        grad_v_bshd: torch.Tensor | None,
    ):
        if grad_q_rot is None and grad_k_rot is None and grad_v_bshd is None:
            return (None,) * 15

        from .nf4_linear import _dequant_triple_to_workspace
        from .rmsnorm import _norm_launch_cfg, rmsnorm_bwd_2d, triton_rmsnorm_fwd
        from .rope import _normalize_cos_sin, launch_qk_rope_bshd_strided

        (
            x_2d,
            norm_w,
            r,
            cos_3d,
            sin_3d,
            hq,
            mask_q,
            hk,
            mask_k,
            hv,
            mask_v,
            wa_q,
            wb_q,
            wa_k,
            wb_k,
            wa_v,
            wb_v,
        ) = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        b, s, d = orig_shape[0], orig_shape[1], orig_shape[2]
        m = b * s
        q_base, q_qs, k_base, k_qs, v_base, v_qs, dq, dk, dv, _ = ctx.qkv_meta
        scale_q, drop_q, scale_k, drop_k, scale_v, drop_v = ctx.lora_cfg
        has_q, has_k, has_v = ctx.has_lora
        comp_dtype = ctx.comp_dtype
        head_dim = ctx.head_dim
        h_q = dq // head_dim
        h_k = dk // head_dim

        grad_qkv_2d = torch.empty((m, dq + dk + dv), dtype=comp_dtype, device=x_2d.device)
        dq_view = grad_qkv_2d[:, :dq].view(b, s, h_q, head_dim)
        dk_view = grad_qkv_2d[:, dq : dq + dk].view(b, s, h_k, head_dim)

        if grad_q_rot is not None or grad_k_rot is not None:
            gq = (
                (grad_q_rot if grad_q_rot.dtype == comp_dtype else grad_q_rot.to(comp_dtype)).contiguous()
                if grad_q_rot is not None
                else dq_view
            )
            gk = (
                (grad_k_rot if grad_k_rot.dtype == comp_dtype else grad_k_rot.to(comp_dtype)).contiguous()
                if grad_k_rot is not None
                else dk_view
            )
            _, sc_b, sc_s = _normalize_cos_sin(cos_3d, s, head_dim)
            _, ss_b, ss_s = _normalize_cos_sin(sin_3d, s, head_dim)
            launch_qk_rope_bshd_strided(
                gq, gk, cos_3d, sin_3d, dq_view, dk_view,
                sc_b, sc_s, ss_b, ss_s, b, s, h_q, h_k, head_dim, 1,
            )
            if grad_q_rot is None:
                grad_qkv_2d[:, :dq].zero_()
            if grad_k_rot is None:
                grad_qkv_2d[:, dq : dq + dk].zero_()
        else:
            grad_qkv_2d[:, : dq + dk].zero_()

        if grad_v_bshd is not None:
            gv = grad_v_bshd if grad_v_bshd.dtype == comp_dtype else grad_v_bshd.to(comp_dtype)
            grad_qkv_2d[:, dq + dk :].copy_(gv.reshape(m, dv))
        else:
            grad_qkv_2d[:, dq + dk :].zero_()

        need_x = ctx.needs_input_grad[0]
        need_norm_w = ctx.needs_input_grad[1]
        need_wa_q, need_wb_q = ctx.needs_input_grad[4], ctx.needs_input_grad[5]
        need_wa_k, need_wb_k = ctx.needs_input_grad[6], ctx.needs_input_grad[7]
        need_wa_v, need_wb_v = ctx.needs_input_grad[8], ctx.needs_input_grad[9]
        need_normed = need_x or need_norm_w

        normed_2d = None
        if (has_q and need_wa_q) or (has_k and need_wa_k) or (has_v and need_wa_v):
            normed_2d = torch.empty((m, d), dtype=comp_dtype, device=x_2d.device)
            if m > 0:
                block, warps = _norm_launch_cfg(d)
                triton_rmsnorm_fwd[(m,)](
                    x_2d,
                    x_2d,
                    normed_2d,
                    normed_2d,
                    norm_w,
                    r,
                    x_2d.stride(0),
                    d,
                    ctx.norm_eps,
                    0,
                    BLOCK_SIZE=block,
                    num_warps=warps,
                )

        grad_wa_q, grad_wb_q, dx_q = _lora_bwd_branch(
            grad_qkv_2d[:, :dq],
            hq,
            mask_q,
            wa_q,
            wb_q,
            scale_q,
            drop_q,
            has_q,
            need_wa_q,
            need_wb_q,
            need_normed,
            normed_2d,
            comp_dtype,
        )
        grad_wa_k, grad_wb_k, dx_k = _lora_bwd_branch(
            grad_qkv_2d[:, dq : dq + dk],
            hk,
            mask_k,
            wa_k,
            wb_k,
            scale_k,
            drop_k,
            has_k,
            need_wa_k,
            need_wb_k,
            need_normed,
            normed_2d,
            comp_dtype,
        )
        grad_wa_v, grad_wb_v, dx_v = _lora_bwd_branch(
            grad_qkv_2d[:, dq + dk :],
            hv,
            mask_v,
            wa_v,
            wb_v,
            scale_v,
            drop_v,
            has_v,
            need_wa_v,
            need_wb_v,
            need_normed,
            normed_2d,
            comp_dtype,
        )
        del normed_2d

        dx = None
        grad_norm_w = None
        if need_normed:
            d_normed = None
            for dx_part in (dx_q, dx_k, dx_v):
                if dx_part is not None:
                    d_normed = dx_part if d_normed is None else d_normed.add_(dx_part)

            w_qkv = _dequant_triple_to_workspace(
                q_base.weight.data,
                q_qs,
                k_base.weight.data,
                k_qs,
                v_base.weight.data,
                v_qs,
                comp_dtype,
            )
            if d_normed is not None:
                d_normed = torch.addmm(d_normed, grad_qkv_2d, w_qkv)
            else:
                d_normed = torch.matmul(grad_qkv_2d, w_qkv)
            del grad_qkv_2d

            if need_norm_w:
                grad_norm_w = (d_normed.float() * (x_2d.float() * r.unsqueeze(1))).sum(dim=0).to(norm_w.dtype)
            if need_x:
                dx = rmsnorm_bwd_2d(d_normed, x_2d, norm_w, r).view(orig_shape)
            del d_normed

        return (
            dx,
            grad_norm_w,
            None,
            None,
            grad_wa_q,
            grad_wb_q,
            grad_wa_k,
            grad_wb_k,
            grad_wa_v,
            grad_wb_v,
            None,
            None,
            None,
            None,
            None,
        )


def _get_fused_qkv_meta(attn_mod):
    cached = getattr(attn_mod, "_soup_qkv_meta", None)
    if cached is not None:
        return cached

    q_base, q_ok = _unwrap_4bit_base(attn_mod.q_proj)
    k_base, k_ok = _unwrap_4bit_base(attn_mod.k_proj)
    v_base, v_ok = _unwrap_4bit_base(attn_mod.v_proj)
    if not (q_ok and k_ok and v_ok):
        attn_mod._soup_qkv_meta = False
        return False

    q_qs = q_base._soup_cached_qs
    k_qs = k_base._soup_cached_qs
    v_qs = v_base._soup_cached_qs
    if not (q_qs.shape[1] == k_qs.shape[1] == v_qs.shape[1]):
        attn_mod._soup_qkv_meta = False
        return False

    dq, dk, dv = q_qs.shape[0], k_qs.shape[0], v_qs.shape[0]
    if q_base.bias is None and k_base.bias is None and v_base.bias is None:
        qkv_bias = None
    else:
        ref_b = q_base.bias if q_base.bias is not None else (k_base.bias if k_base.bias is not None else v_base.bias)
        bq = q_base.bias if q_base.bias is not None else torch.zeros((dq,), dtype=ref_b.dtype, device=ref_b.device)
        bk = k_base.bias if k_base.bias is not None else torch.zeros((dk,), dtype=ref_b.dtype, device=ref_b.device)
        bv = v_base.bias if v_base.bias is not None else torch.zeros((dv,), dtype=ref_b.dtype, device=ref_b.device)
        qkv_bias = torch.cat([bq.detach(), bk.detach(), bv.detach()], dim=0)

    meta = (q_base, q_qs, k_base, k_qs, v_base, v_qs, dq, dk, dv, qkv_bias)
    attn_mod._soup_qkv_meta = meta
    return meta


def _fast_qwen2_attention_forward(
    attn_mod,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    packed_ctx: PackedBatchContext | None,
) -> torch.Tensor:
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    cos, sin = position_embeddings
    dropout_p = attn_mod.attention_dropout if attn_mod.training else 0.0
    scale = attn_mod.scaling

    qkv_meta = _get_fused_qkv_meta(attn_mod)
    if qkv_meta:
        from .nf4_linear import TritonFusedQKVBaseFunc, _resolve_bnb_compute_dtype

        q_base, q_qs, k_base, k_qs, v_base, v_qs, dq, dk, dv, qkv_bias = qkv_meta
        comp_dtype = _resolve_bnb_compute_dtype(q_base, hidden_states)
        if comp_dtype in (torch.float16, torch.bfloat16, torch.float32) and hidden_states.shape[-1] == q_qs.shape[1]:
            qkv = TritonFusedQKVBaseFunc.apply(
                hidden_states,
                q_base.weight.data,
                q_qs,
                k_base.weight.data,
                k_qs,
                v_base.weight.data,
                v_qs,
                qkv_bias,
                comp_dtype,
            )
            qkv_2d = qkv.reshape(-1, dq + dk + dv)
            q_2d = _apply_lora_adapters_2d(attn_mod.q_proj, hidden_states, qkv_2d[:, :dq])
            k_2d = _apply_lora_adapters_2d(attn_mod.k_proj, hidden_states, qkv_2d[:, dq : dq + dk])
            v_2d = _apply_lora_adapters_2d(attn_mod.v_proj, hidden_states, qkv_2d[:, dq + dk :])
            q_bshd = q_2d.view(*input_shape, -1, head_dim)
            k_bshd = k_2d.view(*input_shape, -1, head_dim)
            v_bshd = v_2d.view(*input_shape, -1, head_dim)
        else:
            qkv_meta = False

    if not qkv_meta:
        hidden_shape = (*input_shape, -1, head_dim)
        q_bshd = attn_mod.q_proj(hidden_states).view(hidden_shape)
        k_bshd = attn_mod.k_proj(hidden_states).view(hidden_shape)
        v_bshd = attn_mod.v_proj(hidden_states).view(hidden_shape)

    # Fast zero-transpose BSHD + direct C++ CUTLASS SM75 attention when eligible (fp16/bf16, no explicit mask)
    can_direct_cutlass = (
        attention_mask is None
        and q_bshd.dtype in (torch.float16, torch.bfloat16)
        and (head_dim % 8 == 0)
    )
    if can_direct_cutlass:
        from .rope import fused_qk_rope_bshd

        q_rot, k_rot = fused_qk_rope_bshd(q_bshd, k_bshd, cos, sin)
        if packed_ctx is not None and packed_ctx.cu_seqlens is not None:
            attn_out = VarlenCutlassAttentionFunc.apply(
                q_rot,
                k_rot,
                v_bshd,
                packed_ctx.cu_seqlens,
                packed_ctx.max_seqlen,
                scale,
                dropout_p,
                packed_ctx.max_seqlen > 1,
            )
        else:
            s_len = q_rot.shape[1]
            attn_out = VarlenCutlassAttentionFunc.apply(
                q_rot,
                k_rot,
                v_bshd,
                None,
                s_len,
                scale,
                dropout_p,
                s_len > 1,
            )
        return attn_mod.o_proj(attn_out.reshape(*input_shape, -1))

    # Fallback path for custom 4D attention masks or FP32 inputs
    from .rope import fused_qk_rope

    q = q_bshd.transpose(1, 2)
    k = k_bshd.transpose(1, 2)
    v = v_bshd.transpose(1, 2)
    q, k = fused_qk_rope(q, k, cos, sin)
    n_rep = attn_mod.num_key_value_groups

    if packed_ctx is not None and packed_ctx.valid_idx is not None:
        out_parts = []
        for s0, s1 in packed_ctx.segments:
            q_seg = q[:, :, s0:s1, :]
            k_seg = _repeat_kv_bhsd(k[:, :, s0:s1, :], n_rep)
            v_seg = _repeat_kv_bhsd(v[:, :, s0:s1, :], n_rep)
            out_parts.append(
                F.scaled_dot_product_attention(
                    q_seg,
                    k_seg,
                    v_seg,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    scale=scale,
                    is_causal=(s1 - s0) > 1,
                )
            )
        attn_out = out_parts[0] if len(out_parts) == 1 else torch.cat(out_parts, dim=2)
    else:
        k_exp = _repeat_kv_bhsd(k, n_rep)
        v_exp = _repeat_kv_bhsd(v, n_rep)
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


def _fast_qwen2_mlp_forward(
    mlp_mod,
    hidden_states: torch.Tensor,
    recompute_mode: int = 0,
    s_saved: torch.Tensor | None = None,
    norm_w: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    add_residual: bool = False,
) -> torch.Tensor:
    from .nf4_linear import TritonNF4SwiGLUMLPFunc, _resolve_bnb_compute_dtype, get_valid_4bit_state

    gate_mod = mlp_mod.gate_proj
    up_mod = mlp_mod.up_proj
    down_mod = mlp_mod.down_proj

    if gate_mod.bias is None and up_mod.bias is None and down_mod.bias is None:
        g_qs = get_valid_4bit_state(gate_mod)
        u_qs = get_valid_4bit_state(up_mod)
        d_qs = get_valid_4bit_state(down_mod)
        if g_qs is not None and u_qs is not None and d_qs is not None:
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
                    recompute_mode,
                    s_saved,
                    norm_w,
                    norm_eps,
                    add_residual,
                )

    out = mlp_mod(hidden_states)
    if add_residual and s_saved is not None:
        if out.dtype != s_saved.dtype:
            out = out.to(s_saved.dtype)
        return s_saved + out
    return out


def _attention_subblock(
    layer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    packed_ctx: PackedBatchContext | None,
) -> torch.Tensor:
    attn_mod = layer.self_attn
    head_dim = attn_mod.head_dim
    qkv_meta = _get_fused_qkv_meta(attn_mod)

    if (
        qkv_meta
        and attention_mask is None
        and hidden_states.ndim == 3
        and (head_dim % 8 == 0)
        and (head_dim // 2) <= 256
    ):
        from .nf4_linear import _resolve_bnb_compute_dtype

        q_base, q_qs = qkv_meta[0], qkv_meta[1]
        comp_dtype = _resolve_bnb_compute_dtype(q_base, hidden_states)
        if comp_dtype in (torch.float16, torch.bfloat16) and hidden_states.shape[-1] == q_qs.shape[1]:
            wa_q, wb_q, scale_q, drop_q, ok_q = _extract_single_lora(attn_mod.q_proj)
            wa_k, wb_k, scale_k, drop_k, ok_k = _extract_single_lora(attn_mod.k_proj)
            wa_v, wb_v, scale_v, drop_v, ok_v = _extract_single_lora(attn_mod.v_proj)
            if ok_q and ok_k and ok_v:
                cos, sin = position_embeddings
                q_rot, k_rot, v_bshd = TritonFusedAttnQKVRoPEFunc.apply(
                    hidden_states,
                    layer.input_layernorm.weight,
                    cos,
                    sin,
                    wa_q,
                    wb_q,
                    wa_k,
                    wb_k,
                    wa_v,
                    wb_v,
                    layer.input_layernorm.variance_epsilon,
                    qkv_meta,
                    (scale_q, drop_q, scale_k, drop_k, scale_v, drop_v),
                    comp_dtype,
                    head_dim,
                )
                dropout_p = attn_mod.attention_dropout if attn_mod.training else 0.0
                scale = attn_mod.scaling
                if packed_ctx is not None and packed_ctx.cu_seqlens is not None:
                    attn_out = VarlenCutlassAttentionFunc.apply(
                        q_rot,
                        k_rot,
                        v_bshd,
                        packed_ctx.cu_seqlens,
                        packed_ctx.max_seqlen,
                        scale,
                        dropout_p,
                        packed_ctx.max_seqlen > 1,
                    )
                else:
                    s_len = q_rot.shape[1]
                    attn_out = VarlenCutlassAttentionFunc.apply(
                        q_rot,
                        k_rot,
                        v_bshd,
                        None,
                        s_len,
                        scale,
                        dropout_p,
                        s_len > 1,
                    )
                return attn_mod.o_proj(attn_out.reshape(*hidden_states.shape[:-1], -1))

    from .rmsnorm import rmsnorm

    normed = rmsnorm(
        hidden_states,
        layer.input_layernorm.weight,
        layer.input_layernorm.variance_epsilon,
    )
    return _fast_qwen2_attention_forward(
        attn_mod,
        normed,
        position_embeddings,
        attention_mask,
        packed_ctx,
    )


class AttentionSublayerCheckpointFunc(torch.autograd.Function):
    """Low-overhead custom C++-autograd activation checkpointing ONLY for the Attention sub-block
    (11.9% of layer weights), leaving MLP outside so down_proj forward & w_gate_up dequant are never repeated."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        layer,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        packed_ctx: PackedBatchContext | None,
        *trainable_attn_params: torch.Tensor,
    ):
        ctx.rng_state = torch.cuda.get_rng_state(hidden_states.device)
        ctx.autocast_enabled = torch.is_autocast_enabled("cuda")
        ctx.autocast_dtype = torch.get_autocast_dtype("cuda")
        ctx.layer = layer
        ctx.attention_mask = attention_mask
        ctx.packed_ctx = packed_ctx
        ctx.save_for_backward(hidden_states.detach(), cos, sin, *trainable_attn_params)
        ctx.set_materialize_grads(False)

        with torch.no_grad():
            return _attention_subblock(
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
                out = _attention_subblock(
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
    """Install fast unpadded / direct-CUTLASS / zero-redundant-recompute Qwen2 decoder layers."""
    base_model = getattr(causal_lm, "model", None)
    layers = getattr(base_model, "layers", None)
    if base_model is None or layers is None:
        return
    if causal_lm.__class__.__name__ != "Qwen2ForCausalLM":
        return

    from .rmsnorm import fused_add_rmsnorm

    for layer_idx, layer in enumerate(layers):
        if getattr(layer, "_soup_fast_layer_installed", False):
            continue
        orig_layer_fwd = layer.forward
        trainable_attn_params = tuple(
            p
            for p in list(layer.input_layernorm.parameters()) + list(layer.self_attn.parameters())
            if p.requires_grad
        )
        has_trainable_mlp = any(p.requires_grad for p in layer.mlp.parameters())

        def _make_layer_fwd(_idx, _layer, _orig_fwd, _attn_trainable, _mlp_trainable):
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
                grad_active = torch.is_grad_enabled() and (
                    hidden_states.requires_grad or bool(_attn_trainable) or _mlp_trainable
                )

                if not grad_active:
                    attn_delta = _attention_subblock(
                        _layer, hidden_states, position_embeddings, mask_arg, packed_ctx
                    )
                    residual, post_normed = fused_add_rmsnorm(
                        hidden_states,
                        attn_delta,
                        _layer.post_attention_layernorm.weight,
                        _layer.post_attention_layernorm.variance_epsilon,
                    )
                    return _fast_qwen2_mlp_forward(
                        _layer.mlp,
                        post_normed,
                        recompute_mode=0,
                        s_saved=residual,
                        add_residual=True,
                    )

                # Token-aware dynamic checkpointing for Attention sub-block
                if packed_ctx is not None:
                    unckpt_limit = packed_ctx.unckpt_attn_layers
                else:
                    n_toks = hidden_states.shape[0] * hidden_states.shape[1]
                    unckpt_limit = _compute_unckpt_attn_layers(n_toks)

                cos, sin = position_embeddings
                if _idx < unckpt_limit:
                    attn_delta = _attention_subblock(
                        _layer, hidden_states, position_embeddings, mask_arg, packed_ctx
                    )
                else:
                    attn_delta = AttentionSublayerCheckpointFunc.apply(
                        hidden_states,
                        _layer,
                        cos,
                        sin,
                        mask_arg,
                        packed_ctx,
                        *_attn_trainable,
                    )

                w_post = _layer.post_attention_layernorm.weight
                eps_post = _layer.post_attention_layernorm.variance_epsilon
                residual, post_normed = fused_add_rmsnorm(
                    hidden_states,
                    attn_delta,
                    w_post,
                    eps_post,
                )

                # MLP uses recompute_mode=2 + fused residual add (zero AddBackward0 node,
                # zero extra saved tensors, zero down_proj recompute, 1-launch w_gate_up dequant!)
                recomp_mode = 0 if _mlp_trainable else 2
                out = _fast_qwen2_mlp_forward(
                    _layer.mlp,
                    post_normed,
                    recompute_mode=recomp_mode,
                    s_saved=residual,
                    norm_w=w_post,
                    norm_eps=eps_post,
                    add_residual=True,
                )
                del post_normed, residual
                return out

            return _layer_fwd

        layer.forward = _make_layer_fwd(
            layer_idx, layer, orig_layer_fwd, trainable_attn_params, has_trainable_mlp
        )
        layer._soup_fast_layer_installed = True
