"""Fused NF4/FP4 Double-Quant Dequantization + Packed SwiGLU MLP + Fused QKV Base Projection in Triton."""

from __future__ import annotations

import torch
from torch.nn import functional as F
import triton
import triton.language as tl


def _get_dequant_workspace(numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    from .cross_entropy import get_shared_workspace

    return get_shared_workspace(numel, dtype, device)


@triton.jit(do_not_specialize=["is_nested"])
def triton_nf4_dequant_kernel(
    qweight_ptr,
    absmax_ptr,
    s2_absmax_ptr,
    s2_code_ptr,
    offset_ptr,
    code_ptr,
    out_ptr,
    n_bytes,
    n_elements,
    is_nested,
    HALF_BLOCKSIZE: tl.constexpr,
    S2_BLOCKSIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """Ultra-fast single-matrix 4-bit NF4/FP4 double-quant dequantization kernel."""
    pid = tl.program_id(0)
    byte_offs = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    byte_mask = byte_offs < n_bytes

    packed = tl.load(qweight_ptr + byte_offs, mask=byte_mask, other=0).to(tl.int32)
    hi_idx = (packed >> 4) & 0x0F
    lo_idx = packed & 0x0F

    blk_idx = byte_offs // HALF_BLOCKSIZE
    if is_nested != 0:
        qabs = tl.load(absmax_ptr + blk_idx, mask=byte_mask, other=0).to(tl.int32)
        s2_idx = blk_idx // S2_BLOCKSIZE
        s2_abs = tl.load(s2_absmax_ptr + s2_idx, mask=byte_mask, other=0.0).to(tl.float32)
        s2_val = tl.load(s2_code_ptr + qabs, mask=byte_mask, other=0.0).to(tl.float32)
        off_val = tl.load(offset_ptr).to(tl.float32)
        scale = s2_val * s2_abs + off_val
    else:
        scale = tl.load(absmax_ptr + blk_idx, mask=byte_mask, other=0.0).to(tl.float32)

    val_hi = tl.load(code_ptr + hi_idx) * scale
    val_lo = tl.load(code_ptr + lo_idx) * scale

    pair = tl.join(val_hi, val_lo)
    vals = tl.reshape(pair, (BLOCK_BYTES * 2,))

    out_offs = pid * (BLOCK_BYTES * 2) + tl.arange(0, BLOCK_BYTES * 2)
    out_mask = out_offs < n_elements
    tl.store(out_ptr + out_offs, vals.to(out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit(do_not_specialize=["g1", "nb1", "ne1", "nb2", "ne2", "is_nested"])
def triton_nf4_dequant_pair_kernel(
    qw1_ptr,
    abs1_ptr,
    s2a1_ptr,
    s2c1_ptr,
    off1_ptr,
    qw2_ptr,
    abs2_ptr,
    s2a2_ptr,
    s2c2_ptr,
    off2_ptr,
    code_ptr,
    out_ptr,
    nb1,
    ne1,
    nb2,
    ne2,
    g1,
    is_nested,
    HALF_BLOCKSIZE: tl.constexpr,
    S2_BLOCKSIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """Ultra-fast 2-matrix (gate+up) single-launch NF4/FP4 double-quant dequantization kernel."""
    pid = tl.program_id(0)
    if pid < g1:
        lpid = pid
        qweight_ptr = qw1_ptr
        absmax_ptr = abs1_ptr
        s2_absmax_ptr = s2a1_ptr
        s2_code_ptr = s2c1_ptr
        offset_ptr = off1_ptr
        n_bytes = nb1
        n_elements = ne1
        dst_ptr = out_ptr
    else:
        lpid = pid - g1
        qweight_ptr = qw2_ptr
        absmax_ptr = abs2_ptr
        s2_absmax_ptr = s2a2_ptr
        s2_code_ptr = s2c2_ptr
        offset_ptr = off2_ptr
        n_bytes = nb2
        n_elements = ne2
        dst_ptr = out_ptr + ne1

    byte_offs = lpid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    byte_mask = byte_offs < n_bytes

    packed = tl.load(qweight_ptr + byte_offs, mask=byte_mask, other=0).to(tl.int32)
    hi_idx = (packed >> 4) & 0x0F
    lo_idx = packed & 0x0F

    blk_idx = byte_offs // HALF_BLOCKSIZE
    if is_nested != 0:
        qabs = tl.load(absmax_ptr + blk_idx, mask=byte_mask, other=0).to(tl.int32)
        s2_idx = blk_idx // S2_BLOCKSIZE
        s2_abs = tl.load(s2_absmax_ptr + s2_idx, mask=byte_mask, other=0.0).to(tl.float32)
        s2_val = tl.load(s2_code_ptr + qabs, mask=byte_mask, other=0.0).to(tl.float32)
        off_val = tl.load(offset_ptr).to(tl.float32)
        scale = s2_val * s2_abs + off_val
    else:
        scale = tl.load(absmax_ptr + blk_idx, mask=byte_mask, other=0.0).to(tl.float32)

    val_hi = tl.load(code_ptr + hi_idx) * scale
    val_lo = tl.load(code_ptr + lo_idx) * scale

    pair = tl.join(val_hi, val_lo)
    vals = tl.reshape(pair, (BLOCK_BYTES * 2,))

    out_offs = lpid * (BLOCK_BYTES * 2) + tl.arange(0, BLOCK_BYTES * 2)
    out_mask = out_offs < n_elements
    tl.store(dst_ptr + out_offs, vals.to(out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit(
    do_not_specialize=[
        "g1",
        "g12",
        "nb1",
        "ne1",
        "nb2",
        "ne2",
        "nb3",
        "ne3",
        "ne1",
        "ne12",
        "is_nested",
    ]
)
def triton_nf4_dequant_triple_kernel(
    qw1_ptr,
    abs1_ptr,
    s2a1_ptr,
    s2c1_ptr,
    off1_ptr,
    qw2_ptr,
    abs2_ptr,
    s2a2_ptr,
    s2c2_ptr,
    off2_ptr,
    qw3_ptr,
    abs3_ptr,
    s2a3_ptr,
    s2c3_ptr,
    off3_ptr,
    code_ptr,
    out_ptr,
    nb1,
    ne1,
    nb2,
    ne2,
    nb3,
    ne3,
    g1,
    g12,
    ne1_offset,
    ne12_offset,
    is_nested,
    HALF_BLOCKSIZE: tl.constexpr,
    S2_BLOCKSIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """Ultra-fast 3-matrix (Q+K+V) single-launch NF4/FP4 double-quant dequantization kernel."""
    pid = tl.program_id(0)
    if pid < g1:
        lpid = pid
        qweight_ptr = qw1_ptr
        absmax_ptr = abs1_ptr
        s2_absmax_ptr = s2a1_ptr
        s2_code_ptr = s2c1_ptr
        offset_ptr = off1_ptr
        n_bytes = nb1
        n_elements = ne1
        dst_ptr = out_ptr
    elif pid < g12:
        lpid = pid - g1
        qweight_ptr = qw2_ptr
        absmax_ptr = abs2_ptr
        s2_absmax_ptr = s2a2_ptr
        s2_code_ptr = s2c2_ptr
        offset_ptr = off2_ptr
        n_bytes = nb2
        n_elements = ne2
        dst_ptr = out_ptr + ne1_offset
    else:
        lpid = pid - g12
        qweight_ptr = qw3_ptr
        absmax_ptr = abs3_ptr
        s2_absmax_ptr = s2a3_ptr
        s2_code_ptr = s2c3_ptr
        offset_ptr = off3_ptr
        n_bytes = nb3
        n_elements = ne3
        dst_ptr = out_ptr + ne12_offset

    byte_offs = lpid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    byte_mask = byte_offs < n_bytes

    packed = tl.load(qweight_ptr + byte_offs, mask=byte_mask, other=0).to(tl.int32)
    hi_idx = (packed >> 4) & 0x0F
    lo_idx = packed & 0x0F

    blk_idx = byte_offs // HALF_BLOCKSIZE
    if is_nested != 0:
        qabs = tl.load(absmax_ptr + blk_idx, mask=byte_mask, other=0).to(tl.int32)
        s2_idx = blk_idx // S2_BLOCKSIZE
        s2_abs = tl.load(s2_absmax_ptr + s2_idx, mask=byte_mask, other=0.0).to(tl.float32)
        s2_val = tl.load(s2_code_ptr + qabs, mask=byte_mask, other=0.0).to(tl.float32)
        off_val = tl.load(offset_ptr).to(tl.float32)
        scale = s2_val * s2_abs + off_val
    else:
        scale = tl.load(absmax_ptr + blk_idx, mask=byte_mask, other=0.0).to(tl.float32)

    val_hi = tl.load(code_ptr + hi_idx) * scale
    val_lo = tl.load(code_ptr + lo_idx) * scale

    pair = tl.join(val_hi, val_lo)
    vals = tl.reshape(pair, (BLOCK_BYTES * 2,))

    out_offs = lpid * (BLOCK_BYTES * 2) + tl.arange(0, BLOCK_BYTES * 2)
    out_mask = out_offs < n_elements
    tl.store(dst_ptr + out_offs, vals.to(out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit(do_not_specialize=["is_bwd"])
def triton_swiglu_packed_kernel(
    gate_up_ptr,
    aux_ptr,
    stride_gu_m,
    stride_aux_m,
    i_dim,
    is_bwd,
    BLOCK_I: tl.constexpr,
):
    """Unified 2D packed [M, 2*I] SwiGLU forward and in-place backward kernel."""
    row = tl.program_id(0)
    col_blk = tl.program_id(1)
    cols = col_blk * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < i_dim

    gu_row = gate_up_ptr + row * stride_gu_m
    aux_row = aux_ptr + row * stride_aux_m
    g = tl.load(gu_row + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_row + i_dim + cols, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig

    if is_bwd == 0:
        act = silu * u
        tl.store(aux_row + cols, act.to(aux_ptr.dtype.element_ty), mask=mask)
    else:
        dy = tl.load(aux_row + cols, mask=mask, other=0.0).to(tl.float32)
        du = dy * silu
        dg = dy * u * sig * (1.0 + g * (1.0 - sig))
        out_ty = gate_up_ptr.dtype.element_ty
        tl.store(gu_row + cols, dg.to(out_ty), mask=mask)
        tl.store(gu_row + i_dim + cols, du.to(out_ty), mask=mask)


def can_triton_dequant(qweight: torch.Tensor, quant_state) -> bool:
    if quant_state is None or not qweight.is_cuda or qweight.dtype != torch.uint8:
        return False
    if quant_state.quant_type not in ("nf4", "fp4"):
        return False
    if quant_state.blocksize is None or quant_state.blocksize < 2 or (quant_state.blocksize % 2 != 0):
        return False
    if quant_state.code is None:
        return False
    if quant_state.nested:
        s2 = quant_state.state2
        if s2 is None or s2.absmax is None or s2.code is None or quant_state.offset is None:
            return False
    return True


def _extract_nf4_slot(qweight: torch.Tensor, quant_state, block_bytes: int = 1024):
    q_flat = qweight.reshape(-1)
    nb = q_flat.numel()
    ne = int(quant_state.shape.numel())
    g = triton.cdiv(nb, block_bytes)
    if quant_state.nested:
        s2 = quant_state.state2
        return (q_flat, quant_state.absmax, s2.absmax, s2.code, quant_state.offset, nb, ne, g, int(s2.blocksize), 1)
    dummy = quant_state.code
    return (q_flat, quant_state.absmax, dummy, dummy, dummy, nb, ne, g, 256, 0)


def _compatible_quant_states(qs_a, qs_b) -> bool:
    if (
        qs_a.nested != qs_b.nested
        or qs_a.blocksize != qs_b.blocksize
        or qs_a.quant_type != qs_b.quant_type
    ):
        return False
    if qs_a.nested and int(qs_a.state2.blocksize) != int(qs_b.state2.blocksize):
        return False
    return True


def dequantize_nf4_triton(
    qweight: torch.Tensor,
    quant_state,
    dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize a bitsandbytes 4-bit (NF4/FP4, single or double-quant) tensor in Triton."""
    target_dtype = dtype if dtype is not None else quant_state.dtype
    shape = quant_state.shape
    n_elements = int(shape.numel())

    if out is None:
        out_flat = torch.empty((n_elements,), dtype=target_dtype, device=qweight.device)
    else:
        out_flat = out.reshape(-1)[:n_elements]

    BLOCK_BYTES = 1024
    qw1, abs1, s2a1, s2c1, off1, nb1, ne1, g1, s2_bs, is_nested = _extract_nf4_slot(
        qweight, quant_state, BLOCK_BYTES
    )
    half_bs = int(quant_state.blocksize) // 2

    triton_nf4_dequant_kernel[(g1,)](
        qw1,
        abs1,
        s2a1,
        s2c1,
        off1,
        quant_state.code,
        out_flat,
        nb1,
        ne1,
        is_nested,
        HALF_BLOCKSIZE=half_bs,
        S2_BLOCKSIZE=s2_bs,
        BLOCK_BYTES=BLOCK_BYTES,
        num_warps=4,
    )

    out_2d = out_flat.view(shape)
    if qweight.ndim == 2 and qweight.shape[0] == 1 and qweight.shape[1] > 1:
        return out_2d.t()
    return out_2d


def _dequant_to_workspace(qweight: torch.Tensor, quant_state, dtype: torch.dtype) -> torch.Tensor:
    numel = int(quant_state.shape.numel())
    ws = _get_dequant_workspace(numel, dtype, qweight.device)
    return dequantize_nf4_triton(qweight.view(-1, 1), quant_state, dtype=dtype, out=ws)


def _dequant_pair_to_workspace(
    qw1: torch.Tensor,
    qs1,
    qw2: torch.Tensor,
    qs2,
    dtype: torch.dtype,
) -> torch.Tensor:
    n1 = int(qs1.shape.numel())
    n2 = int(qs2.shape.numel())
    ws = _get_dequant_workspace(n1 + n2, dtype, qw1.device)
    if _compatible_quant_states(qs1, qs2):
        BLOCK_BYTES = 1024
        q1, abs1, s2a1, s2c1, off1, nb1, ne1, g1, s2_bs, is_nested = _extract_nf4_slot(
            qw1, qs1, BLOCK_BYTES
        )
        q2, abs2, s2a2, s2c2, off2, nb2, ne2, g2, _, _ = _extract_nf4_slot(
            qw2, qs2, BLOCK_BYTES
        )
        g12 = g1 + g2
        half_bs = int(qs1.blocksize) // 2
        triton_nf4_dequant_pair_kernel[(g12,)](
            q1,
            abs1,
            s2a1,
            s2c1,
            off1,
            q2,
            abs2,
            s2a2,
            s2c2,
            off2,
            qs1.code,
            ws,
            nb1,
            ne1,
            nb2,
            ne2,
            g1,
            is_nested,
            HALF_BLOCKSIZE=half_bs,
            S2_BLOCKSIZE=s2_bs,
            BLOCK_BYTES=BLOCK_BYTES,
            num_warps=4,
        )
    else:
        dequantize_nf4_triton(qw1.view(-1, 1), qs1, dtype=dtype, out=ws[:n1])
        dequantize_nf4_triton(qw2.view(-1, 1), qs2, dtype=dtype, out=ws[n1 : n1 + n2])
    return ws.view(qs1.shape[0] + qs2.shape[0], qs1.shape[1])


def _dequant_triple_to_workspace(
    qw1: torch.Tensor,
    qs1,
    qw2: torch.Tensor,
    qs2,
    qw3: torch.Tensor,
    qs3,
    dtype: torch.dtype,
) -> torch.Tensor:
    n1 = int(qs1.shape.numel())
    n2 = int(qs2.shape.numel())
    n3 = int(qs3.shape.numel())
    ws = _get_dequant_workspace(n1 + n2 + n3, dtype, qw1.device)
    if _compatible_quant_states(qs1, qs2) and _compatible_quant_states(qs1, qs3):
        BLOCK_BYTES = 1024
        q1, abs1, s2a1, s2c1, off1, nb1, ne1, g1, s2_bs, is_nested = _extract_nf4_slot(
            qw1, qs1, BLOCK_BYTES
        )
        q2, abs2, s2a2, s2c2, off2, nb2, ne2, g2, _, _ = _extract_nf4_slot(
            qw2, qs2, BLOCK_BYTES
        )
        q3, abs3, s2a3, s2c3, off3, nb3, ne3, g3, _, _ = _extract_nf4_slot(
            qw3, qs3, BLOCK_BYTES
        )
        g12 = g1 + g2
        g123 = g12 + g3
        half_bs = int(qs1.blocksize) // 2
        triton_nf4_dequant_triple_kernel[(g123,)](
            q1,
            abs1,
            s2a1,
            s2c1,
            off1,
            q2,
            abs2,
            s2a2,
            s2c2,
            off2,
            q3,
            abs3,
            s2a3,
            s2c3,
            off3,
            qs1.code,
            ws,
            nb1,
            ne1,
            nb2,
            ne2,
            nb3,
            ne3,
            g1,
            g12,
            ne1,
            ne1 + ne2,
            is_nested,
            HALF_BLOCKSIZE=half_bs,
            S2_BLOCKSIZE=s2_bs,
            BLOCK_BYTES=BLOCK_BYTES,
            num_warps=4,
        )
    else:
        dequantize_nf4_triton(qw1.view(-1, 1), qs1, dtype=dtype, out=ws[:n1])
        dequantize_nf4_triton(qw2.view(-1, 1), qs2, dtype=dtype, out=ws[n1 : n1 + n2])
        dequantize_nf4_triton(qw3.view(-1, 1), qs3, dtype=dtype, out=ws[n1 + n2 : n1 + n2 + n3])
    return ws.view(qs1.shape[0] + qs2.shape[0] + qs3.shape[0], qs1.shape[1])


class TritonNF4LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        qweight: torch.Tensor,
        bias: torch.Tensor | None,
        quant_state,
        compute_dtype: torch.dtype,
    ):
        orig_x_dtype = x.dtype
        if x.numel() == 0:
            ctx.is_empty = True
            ctx.orig_shape = x.shape
            ctx.orig_x_dtype = orig_x_dtype
            ctx.bias_dtype = bias.dtype if bias is not None else None
            out_features = quant_state.shape[0]
            return torch.empty((*x.shape[:-1], out_features), dtype=orig_x_dtype, device=x.device)

        ctx.is_empty = False
        x_comp = x if orig_x_dtype == compute_dtype else x.to(compute_dtype)
        b_comp = (
            None
            if bias is None
            else (bias if bias.dtype == compute_dtype else bias.to(compute_dtype))
        )

        w_dq = _dequant_to_workspace(qweight, quant_state, compute_dtype)
        out = F.linear(x_comp, w_dq, b_comp)
        if out.dtype != orig_x_dtype:
            out = out.to(orig_x_dtype)

        ctx.qweight = qweight
        ctx.quant_state = quant_state
        ctx.compute_dtype = compute_dtype
        ctx.orig_x_dtype = orig_x_dtype
        ctx.bias_dtype = bias.dtype if (bias is not None and bias.requires_grad) else None
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor | None):
        if grad_output is None:
            return None, None, None, None, None
        if ctx.is_empty:
            dx = torch.zeros(ctx.orig_shape, dtype=ctx.orig_x_dtype, device=grad_output.device)
            db = (
                torch.zeros((grad_output.shape[-1],), dtype=ctx.bias_dtype, device=grad_output.device)
                if ctx.needs_input_grad[2] and ctx.bias_dtype is not None
                else None
            )
            return dx, None, db, None, None

        grad_bias = None
        if ctx.needs_input_grad[2] and ctx.bias_dtype is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0, dtype=ctx.bias_dtype)

        grad_x = None
        if ctx.needs_input_grad[0]:
            compute_dtype = ctx.compute_dtype
            g_comp = (
                grad_output
                if grad_output.dtype == compute_dtype
                else grad_output.to(compute_dtype)
            )
            w_dq = _dequant_to_workspace(ctx.qweight, ctx.quant_state, compute_dtype)
            g_2d = g_comp.reshape(-1, g_comp.shape[-1])
            grad_x = torch.matmul(g_2d, w_dq).view(*g_comp.shape[:-1], w_dq.shape[1])
            if grad_x.dtype != ctx.orig_x_dtype:
                grad_x = grad_x.to(ctx.orig_x_dtype)

        return grad_x, None, grad_bias, None, None


class TritonFusedQKVBaseFunc(torch.autograd.Function):
    """Fused 4-bit Q+K+V base projection: 1 GEMM in forward, 1 GEMM in backward."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_qw: torch.Tensor,
        q_qs,
        k_qw: torch.Tensor,
        k_qs,
        v_qw: torch.Tensor,
        v_qs,
        qkv_bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ):
        orig_dtype = x.dtype
        orig_shape = x.shape
        out_features = q_qs.shape[0] + k_qs.shape[0] + v_qs.shape[0]
        if x.numel() == 0:
            ctx.is_empty = True
            ctx.orig_shape = orig_shape
            ctx.orig_dtype = orig_dtype
            return torch.empty((*orig_shape[:-1], out_features), dtype=orig_dtype, device=x.device)

        ctx.is_empty = False
        x_comp = x if orig_dtype == compute_dtype else x.to(compute_dtype)
        w_qkv = _dequant_triple_to_workspace(q_qw, q_qs, k_qw, k_qs, v_qw, v_qs, compute_dtype)
        b_comp = (
            None
            if qkv_bias is None
            else (qkv_bias if qkv_bias.dtype == compute_dtype else qkv_bias.to(compute_dtype))
        )
        out = F.linear(x_comp, w_qkv, b_comp)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)

        ctx.weights = (q_qw, q_qs, k_qw, k_qs, v_qw, v_qs)
        ctx.compute_dtype = compute_dtype
        ctx.orig_dtype = orig_dtype
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor | None):
        if grad_out is None:
            return None, None, None, None, None, None, None, None, None
        if ctx.is_empty:
            return (
                torch.zeros(ctx.orig_shape, dtype=ctx.orig_dtype, device=grad_out.device),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None, None, None, None

        q_qw, q_qs, k_qw, k_qs, v_qw, v_qs = ctx.weights
        compute_dtype = ctx.compute_dtype
        g_comp = grad_out if grad_out.dtype == compute_dtype else grad_out.to(compute_dtype)
        w_qkv = _dequant_triple_to_workspace(q_qw, q_qs, k_qw, k_qs, v_qw, v_qs, compute_dtype)
        g_2d = g_comp.reshape(-1, g_comp.shape[-1])
        dx = torch.matmul(g_2d, w_qkv).view(*g_comp.shape[:-1], w_qkv.shape[1])
        if dx.dtype != ctx.orig_dtype:
            dx = dx.to(ctx.orig_dtype)
        return dx, None, None, None, None, None, None, None, None


class TritonNF4SwiGLUMLPFunc(torch.autograd.Function):
    """Fused MLP (packed [gate|up] GEMM + Triton SwiGLU + down_proj) with:
    - Single shared workspace (~52.5 MiB) for stacked [gate; up] weights
    - In-place packed SwiGLU backward (zero transient d_gate / d_up allocations)
    - Zero down_proj forward recomputation & single w_gate_up dequantization in backward
    - recompute_mode:
        0 -> save gate_up [M, 2*I] (fastest, zero recomputation)
        1 -> save x_2d [M, D] (recomputes only gate_up in backward)
        2 -> save s_saved [M, D] shared with post_attention_layernorm (zero extra saved memory!)
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_qw: torch.Tensor,
        gate_qs,
        up_qw: torch.Tensor,
        up_qs,
        down_qw: torch.Tensor,
        down_qs,
        compute_dtype: torch.dtype,
        recompute_mode: int = 0,
        s_saved: torch.Tensor | None = None,
        norm_w: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ):
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_2d = (x if orig_dtype == compute_dtype else x.to(compute_dtype)).reshape(-1, orig_shape[-1]).contiguous()
        m = x_2d.shape[0]
        if m == 0:
            ctx.is_empty = True
            ctx.orig_shape = orig_shape
            ctx.orig_dtype = orig_dtype
            return torch.empty(orig_shape, dtype=orig_dtype, device=x.device)

        ctx.is_empty = False
        i_dim = gate_qs.shape[0]

        w_gate_up = _dequant_pair_to_workspace(gate_qw, gate_qs, up_qw, up_qs, compute_dtype)
        gate_up = torch.matmul(x_2d, w_gate_up.t())

        act = torch.empty((m, i_dim), dtype=compute_dtype, device=x.device)
        BLOCK_I = 1024
        triton_swiglu_packed_kernel[(m, triton.cdiv(i_dim, BLOCK_I))](
            gate_up,
            act,
            gate_up.stride(0),
            act.stride(0),
            i_dim,
            0,
            BLOCK_I=BLOCK_I,
            num_warps=4,
        )

        w_down = _dequant_to_workspace(down_qw, down_qs, compute_dtype)
        out_2d = torch.matmul(act, w_down.t())
        del act
        out = out_2d.view(orig_shape)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)

        if recompute_mode == 2 and s_saved is not None and norm_w is not None:
            del gate_up
            ctx.save_for_backward(s_saved.detach(), norm_w.detach())
            ctx.recompute_mode = 2
            ctx.norm_eps = norm_eps
        elif recompute_mode >= 1:
            del gate_up
            ctx.save_for_backward(x_2d.detach())
            ctx.recompute_mode = 1
        else:
            ctx.save_for_backward(gate_up)
            ctx.recompute_mode = 0

        ctx.weights = (gate_qw, gate_qs, up_qw, up_qs, down_qw, down_qs)
        ctx.compute_dtype = compute_dtype
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = orig_dtype
        ctx.i_dim = i_dim
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor | None):
        none_tail = (None,) * 11
        if grad_out is None:
            return (None, *none_tail)
        if ctx.is_empty:
            return (
                torch.zeros(ctx.orig_shape, dtype=ctx.orig_dtype, device=grad_out.device),
                *none_tail,
            )
        if not ctx.needs_input_grad[0]:
            return (None, *none_tail)

        gate_qw, gate_qs, up_qw, up_qs, down_qw, down_qs = ctx.weights
        compute_dtype = ctx.compute_dtype
        i_dim = ctx.i_dim
        mode = ctx.recompute_mode

        g_2d = (
            grad_out if grad_out.dtype == compute_dtype else grad_out.to(compute_dtype)
        ).reshape(-1, ctx.orig_shape[-1]).contiguous()
        m = g_2d.shape[0]

        # Step 1: d_act = g_2d @ w_down (down_proj forward is NEVER recomputed!)
        w_down = _dequant_to_workspace(down_qw, down_qs, compute_dtype)
        d_act = torch.matmul(g_2d, w_down)

        # Step 2: Dequantize [gate; up] ONCE into shared workspace; reuse for BOTH recompute AND dx!
        w_gate_up = _dequant_pair_to_workspace(gate_qw, gate_qs, up_qw, up_qs, compute_dtype)

        if mode == 2:
            from .rmsnorm import rmsnorm_nograd_2d

            s_saved, norm_w = ctx.saved_tensors
            s_2d = s_saved.reshape(m, -1).contiguous()
            x_rec = rmsnorm_nograd_2d(s_2d, norm_w, ctx.norm_eps)
            if x_rec.dtype != compute_dtype:
                x_rec = x_rec.to(compute_dtype)
            gate_up = torch.matmul(x_rec, w_gate_up.t())
            del x_rec
        elif mode == 1:
            (x_saved,) = ctx.saved_tensors
            gate_up = torch.matmul(x_saved, w_gate_up.t())
        else:
            (gate_up,) = ctx.saved_tensors

        # Step 3: In-place SwiGLU backward overwrites gate_up with [d_gate | d_up]
        BLOCK_I = 1024
        triton_swiglu_packed_kernel[(m, triton.cdiv(i_dim, BLOCK_I))](
            gate_up,
            d_act,
            gate_up.stride(0),
            d_act.stride(0),
            i_dim,
            1,
            BLOCK_I=BLOCK_I,
            num_warps=4,
        )
        del d_act

        # Step 4: Single GEMM dx = [d_gate | d_up] @ [w_gate; w_up] reusing w_gate_up still in ws!
        dx_2d = torch.matmul(gate_up, w_gate_up)
        del gate_up

        dx = dx_2d.view(ctx.orig_shape)
        if dx.dtype != ctx.orig_dtype:
            dx = dx.to(ctx.orig_dtype)
        return (dx, *none_tail)


def _resolve_bnb_compute_dtype(module, x: torch.Tensor) -> torch.dtype:
    if not getattr(module, "compute_type_is_set", False):
        module.set_compute_type(x)
        module.compute_type_is_set = True
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    if module.compute_dtype is not None:
        return module.compute_dtype
    return x.dtype


def get_valid_4bit_state(mod):
    """Cached check returning quant_state if mod is a standard frozen 4-bit Linear4bit."""
    cached = getattr(mod, "_soup_cached_qs", None)
    if cached is not None:
        # Verify weight wasn't replaced dynamically
        if getattr(mod.weight, "quant_state", None) is cached:
            return cached
    if mod.__class__.__name__ != "Linear4bit":
        return None
    from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module

    fix_4bit_weight_quant_state_from_module(mod)
    qs = getattr(mod.weight, "quant_state", None)
    if can_triton_dequant(mod.weight.data, qs) and len(qs.shape) == 2:
        mod._soup_cached_qs = qs
        return qs
    return None


def patch_bnb_nf4_linear() -> None:
    """Patch bitsandbytes Linear4bit and PEFT lora.bnb.Linear4bit to use Triton NF4 dequant
    + cuBLAS Tensor Core GEMM + clone-free fused addmm LoRA."""
    try:
        from bitsandbytes.nn.modules import Linear4bit as BnbLinear4bit
    except ImportError:
        return

    if not getattr(BnbLinear4bit, "_soup_triton_nf4_patched", False):
        orig_bnb_fwd = BnbLinear4bit.forward

        def _fast_bnb_linear4bit_forward(self, x: torch.Tensor):
            if not x.is_cuda:
                return orig_bnb_fwd(self, x)
            qs = get_valid_4bit_state(self)
            if qs is None or x.shape[-1] != qs.shape[1]:
                return orig_bnb_fwd(self, x)
            comp_dtype = _resolve_bnb_compute_dtype(self, x)
            if comp_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                return orig_bnb_fwd(self, x)
            return TritonNF4LinearFunc.apply(x, self.weight.data, self.bias, qs, comp_dtype)

        BnbLinear4bit.forward = _fast_bnb_linear4bit_forward
        BnbLinear4bit._soup_triton_nf4_patched = True

    try:
        from peft.tuners.lora.bnb import Linear4bit as PeftLoraLinear4bit
    except ImportError:
        return

    if not getattr(PeftLoraLinear4bit, "_soup_triton_lora_patched", False):
        orig_peft_fwd = PeftLoraLinear4bit.forward

        def _fast_peft_lora_4bit_forward(self, x: torch.Tensor, *args, **kwargs):
            if (
                not x.is_cuda
                or args
                or kwargs
                or self.disable_adapters
                or self.merged
            ):
                return orig_peft_fwd(self, x, *args, **kwargs)

            for active_adapter in self.active_adapters:
                if active_adapter in self.lora_A and active_adapter in self.lora_variant:
                    return orig_peft_fwd(self, x, *args, **kwargs)

            result = self.base_layer(x)
            out_dtype = result.dtype
            orig_shape = result.shape
            res_2d = result.reshape(-1, orig_shape[-1])
            autocast_on = torch.is_autocast_enabled("cuda")

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A:
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = float(self.scaling[active_adapter])

                x_in = x if autocast_on else self._cast_input_dtype(x, lora_A.weight.dtype)
                h = lora_A(dropout(x_in))
                h_2d = h.reshape(-1, h.shape[-1])
                if h_2d.dtype != out_dtype:
                    h_2d = h_2d.to(out_dtype)
                wb = lora_B.weight
                if wb.dtype != out_dtype:
                    wb = wb.to(out_dtype)
                if lora_B.bias is not None:
                    res_2d = res_2d + (lora_B(h).reshape(-1, orig_shape[-1]).to(out_dtype) * scaling)
                else:
                    res_2d = torch.addmm(res_2d, h_2d, wb.t(), beta=1.0, alpha=scaling)

            return res_2d.view(orig_shape)

        PeftLoraLinear4bit.forward = _fast_peft_lora_4bit_forward
        PeftLoraLinear4bit._soup_triton_lora_patched = True
