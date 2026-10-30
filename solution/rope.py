"""Zero-copy stride-native Fused Q+K Rotary Position Embedding (RoPE) in Triton."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def triton_qk_rope_fwd(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_cos_b,
    stride_cos_s,
    stride_sin_b,
    stride_sin_s,
    seq_len,
    h_q,
    half_dim,
    HAS_K: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_idx = pid_bs // seq_len
    s_idx = pid_bs % seq_len

    cols = tl.arange(0, BLOCK_HALF)
    mask = cols < half_dim

    cos_row = cos_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s
    sin_row = sin_ptr + b_idx * stride_sin_b + s_idx * stride_sin_s

    c1 = tl.load(cos_row + cols, mask=mask, other=1.0).to(tl.float32)
    c2 = tl.load(cos_row + half_dim + cols, mask=mask, other=1.0).to(tl.float32)
    s1 = tl.load(sin_row + cols, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(sin_row + half_dim + cols, mask=mask, other=0.0).to(tl.float32)

    if pid_h < h_q:
        off = b_idx * stride_qb + pid_h * stride_qh + s_idx * stride_qs
        x1 = tl.load(q_ptr + off + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(q_ptr + off + half_dim + cols, mask=mask, other=0.0).to(tl.float32)
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
        tl.store(q_out_ptr + off + cols, y1.to(q_out_ptr.dtype.element_ty), mask=mask)
        tl.store(q_out_ptr + off + half_dim + cols, y2.to(q_out_ptr.dtype.element_ty), mask=mask)
    elif HAS_K:
        kh = pid_h - h_q
        off = b_idx * stride_kb + kh * stride_kh + s_idx * stride_ks
        x1 = tl.load(k_ptr + off + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(k_ptr + off + half_dim + cols, mask=mask, other=0.0).to(tl.float32)
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
        tl.store(k_out_ptr + off + cols, y1.to(k_out_ptr.dtype.element_ty), mask=mask)
        tl.store(k_out_ptr + off + half_dim + cols, y2.to(k_out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def triton_qk_rope_bwd(
    gq_ptr,
    gk_ptr,
    cos_ptr,
    sin_ptr,
    dq_ptr,
    dk_ptr,
    stride_gqb,
    stride_gqh,
    stride_gqs,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_gkb,
    stride_gkh,
    stride_gks,
    stride_dkb,
    stride_dkh,
    stride_dks,
    stride_cos_b,
    stride_cos_s,
    stride_sin_b,
    stride_sin_s,
    seq_len,
    h_q,
    half_dim,
    HAS_GQ: tl.constexpr,
    HAS_GK: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_idx = pid_bs // seq_len
    s_idx = pid_bs % seq_len

    cols = tl.arange(0, BLOCK_HALF)
    mask = cols < half_dim

    cos_row = cos_ptr + b_idx * stride_cos_b + s_idx * stride_cos_s
    sin_row = sin_ptr + b_idx * stride_sin_b + s_idx * stride_sin_s

    c1 = tl.load(cos_row + cols, mask=mask, other=1.0).to(tl.float32)
    c2 = tl.load(cos_row + half_dim + cols, mask=mask, other=1.0).to(tl.float32)
    s1 = tl.load(sin_row + cols, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(sin_row + half_dim + cols, mask=mask, other=0.0).to(tl.float32)

    if pid_h < h_q:
        if HAS_GQ:
            goff = b_idx * stride_gqb + pid_h * stride_gqh + s_idx * stride_gqs
            doff = b_idx * stride_dqb + pid_h * stride_dqh + s_idx * stride_dqs
            g1 = tl.load(gq_ptr + goff + cols, mask=mask, other=0.0).to(tl.float32)
            g2 = tl.load(gq_ptr + goff + half_dim + cols, mask=mask, other=0.0).to(tl.float32)
            dx1 = g1 * c1 + g2 * s2
            dx2 = g2 * c2 - g1 * s1
            tl.store(dq_ptr + doff + cols, dx1.to(dq_ptr.dtype.element_ty), mask=mask)
            tl.store(dq_ptr + doff + half_dim + cols, dx2.to(dq_ptr.dtype.element_ty), mask=mask)
    elif HAS_GK:
        kh = pid_h - h_q
        goff = b_idx * stride_gkb + kh * stride_gkh + s_idx * stride_gks
        doff = b_idx * stride_dkb + kh * stride_dkh + s_idx * stride_dks
        g1 = tl.load(gk_ptr + goff + cols, mask=mask, other=0.0).to(tl.float32)
        g2 = tl.load(gk_ptr + goff + half_dim + cols, mask=mask, other=0.0).to(tl.float32)
        dx1 = g1 * c1 + g2 * s2
        dx2 = g2 * c2 - g1 * s1
        tl.store(dk_ptr + doff + cols, dx1.to(dk_ptr.dtype.element_ty), mask=mask)
        tl.store(dk_ptr + doff + half_dim + cols, dx2.to(dk_ptr.dtype.element_ty), mask=mask)


def _normalize_cos_sin(t: torch.Tensor, seq_len: int, d: int) -> tuple[torch.Tensor, int, int]:
    """Return a 3D view (B_or_1, S, D) with stride(-1) == 1 and (stride_b, stride_s)."""
    if t.ndim == 4:
        # Either (B, 1, S, D) or (B, S, 1, D)
        if t.shape[1] == 1:
            t3 = t.squeeze(1)
        elif t.shape[2] == 1:
            t3 = t.squeeze(2)
        else:
            t3 = t[:, 0]
    elif t.ndim == 3:
        t3 = t
    elif t.ndim == 2:
        t3 = t.unsqueeze(0)
    else:
        t3 = t.reshape(-1, seq_len, d)

    if t3.stride(-1) != 1:
        t3 = t3.contiguous()
    stride_b = 0 if t3.shape[0] == 1 else t3.stride(0)
    stride_s = t3.stride(1)
    return t3, stride_b, stride_s


def _ensure_last_contig(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


class TritonFusedQKRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q = _ensure_last_contig(q)
        k = _ensure_last_contig(k)
        b, h_q, s, d = q.shape
        h_k = k.shape[1]
        half_dim = d // 2

        q_out = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
        k_out = torch.empty_strided(k.shape, k.stride(), device=k.device, dtype=k.dtype)

        cos_3d, sc_b, sc_s = _normalize_cos_sin(cos, s, d)
        sin_3d, ss_b, ss_s = _normalize_cos_sin(sin, s, d)

        if b * s > 0:
            block_half = triton.next_power_of_2(half_dim)
            warps = 2 if block_half <= 64 else 4
            triton_qk_rope_fwd[(b * s, h_q + h_k)](
                q,
                k,
                cos_3d,
                sin_3d,
                q_out,
                k_out,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                sc_b,
                sc_s,
                ss_b,
                ss_s,
                s,
                h_q,
                half_dim,
                HAS_K=True,
                BLOCK_HALF=block_half,
                num_warps=warps,
            )

        ctx.save_for_backward(cos_3d, sin_3d)
        ctx.meta = (
            b,
            h_q,
            h_k,
            s,
            d,
            q.stride(),
            k.stride(),
            q.dtype,
            k.dtype,
            sc_b,
            sc_s,
            ss_b,
            ss_s,
        )
        ctx.set_materialize_grads(False)
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor | None, grad_k: torch.Tensor | None):
        cos_3d, sin_3d = ctx.saved_tensors
        (
            b,
            h_q,
            h_k,
            s,
            d,
            q_stride,
            k_stride,
            q_dtype,
            k_dtype,
            sc_b,
            sc_s,
            ss_b,
            ss_s,
        ) = ctx.meta
        half_dim = d // 2

        need_q = ctx.needs_input_grad[0] and grad_q is not None
        need_k = ctx.needs_input_grad[1] and grad_k is not None
        if not need_q and not need_k:
            return None, None, None, None

        gq = _ensure_last_contig(grad_q) if need_q else cos_3d
        gk = _ensure_last_contig(grad_k) if need_k else cos_3d

        dq = torch.empty_strided((b, h_q, s, d), q_stride, device=cos_3d.device, dtype=q_dtype) if need_q else cos_3d
        dk = torch.empty_strided((b, h_k, s, d), k_stride, device=cos_3d.device, dtype=k_dtype) if need_k else cos_3d

        if b * s > 0:
            block_half = triton.next_power_of_2(half_dim)
            warps = 2 if block_half <= 64 else 4
            triton_qk_rope_bwd[(b * s, h_q + (h_k if need_k else 0))](
                gq,
                gk,
                cos_3d,
                sin_3d,
                dq,
                dk,
                gq.stride(0) if need_q else 0,
                gq.stride(1) if need_q else 0,
                gq.stride(2) if need_q else 0,
                dq.stride(0) if need_q else 0,
                dq.stride(1) if need_q else 0,
                dq.stride(2) if need_q else 0,
                gk.stride(0) if need_k else 0,
                gk.stride(1) if need_k else 0,
                gk.stride(2) if need_k else 0,
                dk.stride(0) if need_k else 0,
                dk.stride(1) if need_k else 0,
                dk.stride(2) if need_k else 0,
                sc_b,
                sc_s,
                ss_b,
                ss_s,
                s,
                h_q,
                half_dim,
                HAS_GQ=need_q,
                HAS_GK=need_k,
                BLOCK_HALF=block_half,
                num_warps=warps,
            )

        return (dq if need_q else None), (dk if need_k else None), None, None


class TritonSingleRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = _ensure_last_contig(x)
        b, h, s, d = x.shape
        half_dim = d // 2
        out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
        cos_3d, sc_b, sc_s = _normalize_cos_sin(cos, s, d)
        sin_3d, ss_b, ss_s = _normalize_cos_sin(sin, s, d)

        if b * s > 0:
            block_half = triton.next_power_of_2(half_dim)
            warps = 2 if block_half <= 64 else 4
            triton_qk_rope_fwd[(b * s, h)](
                x,
                x,
                cos_3d,
                sin_3d,
                out,
                out,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                0,
                0,
                0,
                sc_b,
                sc_s,
                ss_b,
                ss_s,
                s,
                h,
                half_dim,
                HAS_K=False,
                BLOCK_HALF=block_half,
                num_warps=warps,
            )

        ctx.save_for_backward(cos_3d, sin_3d)
        ctx.meta = (b, h, s, d, x.stride(), x.dtype, sc_b, sc_s, ss_b, ss_s)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        cos_3d, sin_3d = ctx.saved_tensors
        b, h, s, d, x_stride, x_dtype, sc_b, sc_s, ss_b, ss_s = ctx.meta
        half_dim = d // 2
        gq = _ensure_last_contig(grad_out)
        dx = torch.empty_strided((b, h, s, d), x_stride, device=gq.device, dtype=x_dtype)

        if b * s > 0:
            block_half = triton.next_power_of_2(half_dim)
            warps = 2 if block_half <= 64 else 4
            triton_qk_rope_bwd[(b * s, h)](
                gq,
                gq,
                cos_3d,
                sin_3d,
                dx,
                dx,
                gq.stride(0),
                gq.stride(1),
                gq.stride(2),
                dx.stride(0),
                dx.stride(1),
                dx.stride(2),
                0,
                0,
                0,
                0,
                0,
                0,
                sc_b,
                sc_s,
                ss_b,
                ss_s,
                s,
                h,
                half_dim,
                HAS_GQ=True,
                HAS_GK=False,
                BLOCK_HALF=block_half,
                num_warps=warps,
            )
        return dx, None, None


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return TritonSingleRoPEFunc.apply(x, cos, sin)


def fused_qk_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return TritonFusedQKRoPEFunc.apply(q, k, cos, sin)


def patch_qwen2_rope() -> None:
    """Patch apply_rotary_pos_emb in transformers.models.qwen2.modeling_qwen2."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except ImportError:
        return

    if getattr(modeling_qwen2, "_soup_rope_patched", False):
        return

    original_apply = modeling_qwen2.apply_rotary_pos_emb

    def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if not q.is_cuda or unsqueeze_dim != 1 or q.ndim != 4 or (q.shape[-1] % 2 != 0):
            return original_apply(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return TritonFusedQKRoPEFunc.apply(q, k, cos, sin)

    modeling_qwen2.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
    modeling_qwen2._soup_rope_patched = True
