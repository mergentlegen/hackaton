"""Zero-copy stride-native Fused Q+K Rotary Position Embedding (RoPE) in Triton (single-kernel compilation)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "stride_cos_b",
        "stride_sin_b",
        "seq_len",
        "h_q",
        "has_q",
        "has_k",
        "is_bwd",
    ]
)
def triton_qk_rope_kernel(
    q_in_ptr,
    k_in_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    stride_qib,
    stride_qih,
    stride_qis,
    stride_qob,
    stride_qoh,
    stride_qos,
    stride_kib,
    stride_kih,
    stride_kis,
    stride_kob,
    stride_koh,
    stride_kos,
    stride_cos_b,
    stride_cos_s,
    stride_sin_b,
    stride_sin_s,
    seq_len,
    h_q,
    half_dim,
    has_q,
    has_k,
    is_bwd,
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
        if has_q != 0:
            ioff = b_idx * stride_qib + pid_h * stride_qih + s_idx * stride_qis
            ooff = b_idx * stride_qob + pid_h * stride_qoh + s_idx * stride_qos
            x1 = tl.load(q_in_ptr + ioff + cols, mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(q_in_ptr + ioff + half_dim + cols, mask=mask, other=0.0).to(tl.float32)
            if is_bwd == 0:
                y1 = x1 * c1 - x2 * s1
                y2 = x2 * c2 + x1 * s2
            else:
                y1 = x1 * c1 + x2 * s2
                y2 = x2 * c2 - x1 * s1
            tl.store(q_out_ptr + ooff + cols, y1.to(q_out_ptr.dtype.element_ty), mask=mask)
            tl.store(q_out_ptr + ooff + half_dim + cols, y2.to(q_out_ptr.dtype.element_ty), mask=mask)
    elif has_k != 0:
        kh = pid_h - h_q
        ioff = b_idx * stride_kib + kh * stride_kih + s_idx * stride_kis
        ooff = b_idx * stride_kob + kh * stride_koh + s_idx * stride_kos
        x1 = tl.load(k_in_ptr + ioff + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(k_in_ptr + ioff + half_dim + cols, mask=mask, other=0.0).to(tl.float32)
        if is_bwd == 0:
            y1 = x1 * c1 - x2 * s1
            y2 = x2 * c2 + x1 * s2
        else:
            y1 = x1 * c1 + x2 * s2
            y2 = x2 * c2 - x1 * s1
        tl.store(k_out_ptr + ooff + cols, y1.to(k_out_ptr.dtype.element_ty), mask=mask)
        tl.store(k_out_ptr + ooff + half_dim + cols, y2.to(k_out_ptr.dtype.element_ty), mask=mask)


def _normalize_cos_sin(t: torch.Tensor, seq_len: int, d: int) -> tuple[torch.Tensor, int, int]:
    """Return a 3D view (B_or_1, S, D) with stride(-1) == 1 and (stride_b, stride_s)."""
    if t.ndim == 4:
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
    """BHSD-convention (B, H, S, D) Fused Q+K RoPE."""

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
            triton_qk_rope_kernel[(b * s, h_q + h_k)](
                q,
                k,
                cos_3d,
                sin_3d,
                q_out,
                k_out,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q_out.stride(0),
                q_out.stride(1),
                q_out.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k_out.stride(0),
                k_out.stride(1),
                k_out.stride(2),
                sc_b,
                sc_s,
                ss_b,
                ss_s,
                s,
                h_q,
                half_dim,
                1,
                1,
                0,
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
            # Pass valid 16-aligned strides even when a branch is inactive (has_q/has_k == 0)
            # so Triton 3.6.0 never re-specializes on zero vs non-zero stride arguments.
            sq0 = gq.stride(0) if need_q else gk.stride(0)
            sq1 = gq.stride(1) if need_q else gk.stride(1)
            sq2 = gq.stride(2) if need_q else gk.stride(2)
            dq0 = dq.stride(0) if need_q else dk.stride(0)
            dq1 = dq.stride(1) if need_q else dk.stride(1)
            dq2 = dq.stride(2) if need_q else dk.stride(2)
            sk0 = gk.stride(0) if need_k else sq0
            sk1 = gk.stride(1) if need_k else sq1
            sk2 = gk.stride(2) if need_k else sq2
            dk0 = dk.stride(0) if need_k else dq0
            dk1 = dk.stride(1) if need_k else dq1
            dk2 = dk.stride(2) if need_k else dq2
            triton_qk_rope_kernel[(b * s, h_q + (h_k if need_k else 0))](
                gq,
                gk,
                cos_3d,
                sin_3d,
                dq,
                dk,
                sq0,
                sq1,
                sq2,
                dq0,
                dq1,
                dq2,
                sk0,
                sk1,
                sk2,
                dk0,
                dk1,
                dk2,
                sc_b,
                sc_s,
                ss_b,
                ss_s,
                s,
                h_q,
                half_dim,
                1 if need_q else 0,
                1 if need_k else 0,
                1,
                BLOCK_HALF=block_half,
                num_warps=warps,
            )

        return (dq if need_q else None), (dk if need_k else None), None, None


class TritonFusedQKRoPEBSHDFunc(torch.autograd.Function):
    """Zero-transpose BSHD-convention (B, S, H, D) Fused Q+K RoPE outputting contiguous (B, S, H, D)."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q = _ensure_last_contig(q)
        k = _ensure_last_contig(k)
        b, s, h_q, d = q.shape
        h_k = k.shape[2]
        half_dim = d // 2

        q_out = torch.empty((b, s, h_q, d), device=q.device, dtype=q.dtype)
        k_out = torch.empty((b, s, h_k, d), device=k.device, dtype=k.dtype)

        cos_3d, sc_b, sc_s = _normalize_cos_sin(cos, s, d)
        sin_3d, ss_b, ss_s = _normalize_cos_sin(sin, s, d)

        if b * s > 0:
            block_half = triton.next_power_of_2(half_dim)
            warps = 2 if block_half <= 64 else 4
            # In BSHD: stride_b = stride(0), stride_h = stride(2), stride_s = stride(1)
            triton_qk_rope_kernel[(b * s, h_q + h_k)](
                q,
                k,
                cos_3d,
                sin_3d,
                q_out,
                k_out,
                q.stride(0),
                q.stride(2),
                q.stride(1),
                q_out.stride(0),
                q_out.stride(2),
                q_out.stride(1),
                k.stride(0),
                k.stride(2),
                k.stride(1),
                k_out.stride(0),
                k_out.stride(2),
                k_out.stride(1),
                sc_b,
                sc_s,
                ss_b,
                ss_s,
                s,
                h_q,
                half_dim,
                1,
                1,
                0,
                BLOCK_HALF=block_half,
                num_warps=warps,
            )

        ctx.save_for_backward(cos_3d, sin_3d)
        ctx.meta = (b, s, h_q, h_k, d, q.dtype, k.dtype, sc_b, sc_s, ss_b, ss_s)
        ctx.set_materialize_grads(False)
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q: torch.Tensor | None, grad_k: torch.Tensor | None):
        cos_3d, sin_3d = ctx.saved_tensors
        b, s, h_q, h_k, d, q_dtype, k_dtype, sc_b, sc_s, ss_b, ss_s = ctx.meta
        half_dim = d // 2

        need_q = ctx.needs_input_grad[0] and grad_q is not None
        need_k = ctx.needs_input_grad[1] and grad_k is not None
        if not need_q and not need_k:
            return None, None, None, None

        gq = _ensure_last_contig(grad_q) if need_q else cos_3d
        gk = _ensure_last_contig(grad_k) if need_k else cos_3d

        dq = torch.empty((b, s, h_q, d), device=cos_3d.device, dtype=q_dtype) if need_q else cos_3d
        dk = torch.empty((b, s, h_k, d), device=cos_3d.device, dtype=k_dtype) if need_k else cos_3d

        if b * s > 0:
            block_half = triton.next_power_of_2(half_dim)
            warps = 2 if block_half <= 64 else 4
            sq0 = gq.stride(0) if need_q else gk.stride(0)
            sq2 = gq.stride(2) if need_q else gk.stride(2)
            sq1 = gq.stride(1) if need_q else gk.stride(1)
            dq0 = dq.stride(0) if need_q else dk.stride(0)
            dq2 = dq.stride(2) if need_q else dk.stride(2)
            dq1 = dq.stride(1) if need_q else dk.stride(1)
            sk0 = gk.stride(0) if need_k else sq0
            sk2 = gk.stride(2) if need_k else sq2
            sk1 = gk.stride(1) if need_k else sq1
            dk0 = dk.stride(0) if need_k else dq0
            dk2 = dk.stride(2) if need_k else dq2
            dk1 = dk.stride(1) if need_k else dq1
            triton_qk_rope_kernel[(b * s, h_q + (h_k if need_k else 0))](
                gq,
                gk,
                cos_3d,
                sin_3d,
                dq,
                dk,
                sq0,
                sq2,
                sq1,
                dq0,
                dq2,
                dq1,
                sk0,
                sk2,
                sk1,
                dk0,
                dk2,
                dk1,
                sc_b,
                sc_s,
                ss_b,
                ss_s,
                s,
                h_q,
                half_dim,
                1 if need_q else 0,
                1 if need_k else 0,
                1,
                BLOCK_HALF=block_half,
                num_warps=warps,
            )

        return (dq if need_q else None), (dk if need_k else None), None, None


def launch_qk_rope_bshd_strided(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    cos_3d: torch.Tensor,
    sin_3d: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    sc_b: int,
    sc_s: int,
    ss_b: int,
    ss_s: int,
    b: int,
    s: int,
    h_q: int,
    h_k: int,
    d: int,
    is_bwd: int,
) -> None:
    """Launch BSHD-indexed RoPE directly on strided QKV views (zero copy/slice overhead)."""
    if b * s <= 0:
        return
    half_dim = d // 2
    block_half = triton.next_power_of_2(half_dim)
    warps = 2 if block_half <= 64 else 4
    triton_qk_rope_kernel[(b * s, h_q + h_k)](
        q_in,
        k_in,
        cos_3d,
        sin_3d,
        q_out,
        k_out,
        q_in.stride(0),
        q_in.stride(2),
        q_in.stride(1),
        q_out.stride(0),
        q_out.stride(2),
        q_out.stride(1),
        k_in.stride(0),
        k_in.stride(2),
        k_in.stride(1),
        k_out.stride(0),
        k_out.stride(2),
        k_out.stride(1),
        sc_b,
        sc_s,
        ss_b,
        ss_s,
        s,
        h_q,
        half_dim,
        1,
        1,
        is_bwd,
        BLOCK_HALF=block_half,
        num_warps=warps,
    )


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
            triton_qk_rope_kernel[(b * s, h)](
                x,
                x,
                cos_3d,
                sin_3d,
                out,
                out,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                x.stride(0),
                x.stride(1),
                x.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                sc_b,
                sc_s,
                ss_b,
                ss_s,
                s,
                h,
                half_dim,
                1,
                0,
                0,
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
            triton_qk_rope_kernel[(b * s, h)](
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
                gq.stride(0),
                gq.stride(1),
                gq.stride(2),
                dx.stride(0),
                dx.stride(1),
                dx.stride(2),
                sc_b,
                sc_s,
                ss_b,
                ss_s,
                s,
                h,
                half_dim,
                1,
                0,
                1,
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


def fused_qk_rope_bshd(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return TritonFusedQKRoPEBSHDFunc.apply(q, k, cos, sin)


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
