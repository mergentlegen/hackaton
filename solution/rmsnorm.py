"""Fused RMSNorm and Fused Residual-Add + RMSNorm in Triton (unified 2-kernel compilation)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["has_res"])
def triton_rmsnorm_fwd(
    x_ptr,
    delta_ptr,
    s_out_ptr,
    y_out_ptr,
    w_ptr,
    r_ptr,
    stride_m,
    d_dim,
    eps,
    has_res,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < d_dim
    base = row_idx * stride_m + cols

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    if has_res != 0:
        delta = tl.load(delta_ptr + base, mask=mask, other=0.0).to(tl.float32)
        s_typed = (x + delta).to(s_out_ptr.dtype.element_ty)
        tl.store(s_out_ptr + base, s_typed, mask=mask)
        x = s_typed.to(tl.float32)

    var = tl.sum(x * x, axis=0) / d_dim
    r = tl.rsqrt(var + eps)
    tl.store(r_ptr + row_idx, r)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * r) * w
    tl.store(y_out_ptr + base, y.to(y_out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["has_gs"])
def triton_rmsnorm_bwd(
    gs_ptr,
    gy_ptr,
    s_ptr,
    w_ptr,
    r_ptr,
    din_ptr,
    stride_m,
    d_dim,
    has_gs,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < d_dim
    base = row_idx * stride_m + cols

    gy = tl.load(gy_ptr + base, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row_idx).to(tl.float32)

    sw = gy * w
    dot = tl.sum(sw * s, axis=0)
    din = r * (sw - s * (r * r * dot / d_dim))
    if has_gs != 0:
        gs = tl.load(gs_ptr + base, mask=mask, other=0.0).to(tl.float32)
        din = din + gs

    tl.store(din_ptr + base, din.to(din_ptr.dtype.element_ty), mask=mask)


def _norm_launch_cfg(d: int) -> tuple[int, int]:
    block = triton.next_power_of_2(d)
    warps = 8 if block >= 2048 else (4 if block >= 512 else 2)
    return block, warps


def rmsnorm_nograd_2d(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    r_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fast no-autograd 2D RMSNorm used during selective MLP recomputation."""
    m, d = x_2d.shape
    y_2d = torch.empty_like(x_2d)
    if m == 0:
        return y_2d
    if r_buf is None:
        r_buf = torch.empty((m,), dtype=torch.float32, device=x_2d.device)
    block, warps = _norm_launch_cfg(d)
    triton_rmsnorm_fwd[(m,)](
        x_2d,
        x_2d,
        y_2d,
        y_2d,
        weight,
        r_buf,
        x_2d.stride(0),
        d,
        eps,
        0,
        BLOCK_SIZE=block,
        num_warps=warps,
    )
    return y_2d


def rmsnorm_fwd_2d_with_r(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D RMSNorm forward returning (y_2d, r_fp32) for fused attention blocks."""
    m, d = x_2d.shape
    y_2d = torch.empty_like(x_2d)
    r = torch.empty((m,), dtype=torch.float32, device=x_2d.device)
    if m > 0:
        block, warps = _norm_launch_cfg(d)
        triton_rmsnorm_fwd[(m,)](
            x_2d,
            x_2d,
            y_2d,
            y_2d,
            weight,
            r,
            x_2d.stride(0),
            d,
            eps,
            0,
            BLOCK_SIZE=block,
            num_warps=warps,
        )
    return y_2d, r


def rmsnorm_bwd_2d(
    grad_2d: torch.Tensor,
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """2D RMSNorm backward returning dx_2d for fused attention blocks."""
    m, d = x_2d.shape
    dx_2d = torch.empty_like(x_2d)
    if m > 0:
        block, warps = _norm_launch_cfg(d)
        triton_rmsnorm_bwd[(m,)](
            grad_2d,
            grad_2d,
            x_2d,
            weight,
            r,
            dx_2d,
            x_2d.stride(0),
            d,
            0,
            BLOCK_SIZE=block,
            num_warps=warps,
        )
    return dx_2d


class TritonRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        orig_shape = x.shape
        d = orig_shape[-1]
        x_2d = x.reshape(-1, d).contiguous()
        m = x_2d.shape[0]
        y_2d = torch.empty_like(x_2d)
        if m == 0:
            r = torch.empty((0,), dtype=torch.float32, device=x.device)
            ctx.save_for_backward(x_2d, weight, r)
            ctx.orig_shape = orig_shape
            ctx.weight_requires_grad = weight.requires_grad
            return y_2d.reshape(orig_shape)

        r = torch.empty((m,), dtype=torch.float32, device=x.device)
        block, warps = _norm_launch_cfg(d)
        triton_rmsnorm_fwd[(m,)](
            x_2d,
            x_2d,
            y_2d,
            y_2d,
            weight,
            r,
            x_2d.stride(0),
            d,
            eps,
            0,
            BLOCK_SIZE=block,
            num_warps=warps,
        )
        ctx.save_for_backward(x_2d, weight, r)
        ctx.orig_shape = orig_shape
        ctx.weight_requires_grad = weight.requires_grad
        return y_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, weight, r = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        d = orig_shape[-1]
        grad_2d = grad_output.reshape(-1, d).contiguous()
        m = grad_2d.shape[0]
        dx_2d = torch.empty_like(x_2d)
        if m > 0:
            block, warps = _norm_launch_cfg(d)
            triton_rmsnorm_bwd[(m,)](
                grad_2d,
                grad_2d,
                x_2d,
                weight,
                r,
                dx_2d,
                x_2d.stride(0),
                d,
                0,
                BLOCK_SIZE=block,
                num_warps=warps,
            )
        if ctx.weight_requires_grad:
            dw = (grad_2d.float() * (x_2d.float() * r.unsqueeze(1))).sum(dim=0).to(weight.dtype)
        else:
            dw = None
        return dx_2d.reshape(orig_shape), dw, None


class TritonFusedAddRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual: torch.Tensor, delta: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        orig_shape = residual.shape
        d = orig_shape[-1]
        res_2d = residual.reshape(-1, d).contiguous()
        if delta.dtype != res_2d.dtype:
            del_2d = delta.to(res_2d.dtype).reshape(-1, d).contiguous()
        else:
            del_2d = delta.reshape(-1, d).contiguous()
        m = res_2d.shape[0]
        s_2d = torch.empty_like(res_2d)
        y_2d = torch.empty_like(res_2d)
        r = torch.empty((m,), dtype=torch.float32, device=residual.device)

        if m > 0:
            block, warps = _norm_launch_cfg(d)
            triton_rmsnorm_fwd[(m,)](
                res_2d,
                del_2d,
                s_2d,
                y_2d,
                weight,
                r,
                res_2d.stride(0),
                d,
                eps,
                1,
                BLOCK_SIZE=block,
                num_warps=warps,
            )
        ctx.save_for_backward(s_2d, weight, r)
        ctx.orig_shape = orig_shape
        ctx.delta_dtype = delta.dtype
        ctx.weight_requires_grad = weight.requires_grad
        ctx.set_materialize_grads(False)
        return s_2d.reshape(orig_shape), y_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_s: torch.Tensor | None, grad_y: torch.Tensor | None):
        s_2d, weight, r = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        d = orig_shape[-1]
        m = s_2d.shape[0]

        if grad_y is None:
            if grad_s is None:
                return None, None, None, None
            d_res = grad_s
            d_del = grad_s.to(ctx.delta_dtype) if ctx.delta_dtype != grad_s.dtype else grad_s
            return d_res, d_del, None, None

        gy_2d = grad_y.reshape(-1, d).contiguous()
        has_gs = 1 if grad_s is not None else 0
        gs_2d = grad_s.reshape(-1, d).contiguous() if has_gs else gy_2d
        din_2d = torch.empty_like(s_2d)

        if m > 0:
            block, warps = _norm_launch_cfg(d)
            triton_rmsnorm_bwd[(m,)](
                gs_2d,
                gy_2d,
                s_2d,
                weight,
                r,
                din_2d,
                s_2d.stride(0),
                d,
                has_gs,
                BLOCK_SIZE=block,
                num_warps=warps,
            )

        if ctx.weight_requires_grad:
            dw = (gy_2d.float() * (s_2d.float() * r.unsqueeze(1))).sum(dim=0).to(weight.dtype)
        else:
            dw = None

        din = din_2d.reshape(orig_shape)
        d_del = din.to(ctx.delta_dtype) if ctx.delta_dtype != din.dtype else din
        return din, d_del, dw, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return TritonRMSNormFunc.apply(x, weight, eps)


def fused_add_rmsnorm(
    residual: torch.Tensor, delta: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    return TritonFusedAddRMSNormFunc.apply(residual, delta, weight, eps)


def patch_qwen2_rmsnorm() -> None:
    """Patch Qwen2RMSNorm to use the fused Triton kernel."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except ImportError:
        return

    if getattr(modeling_qwen2.Qwen2RMSNorm, "_soup_triton_patched", False):
        return

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hidden_states.is_cuda or hidden_states.shape[-1] > 8192:
            return original(self, hidden_states)
        return TritonRMSNormFunc.apply(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward
    modeling_qwen2.Qwen2RMSNorm._soup_triton_patched = True
