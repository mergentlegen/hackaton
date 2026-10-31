"""RMSNorm on the fp32 residual stream, one Triton pass each way.

In Soup's 4-bit setup the embeddings and norm weights are fp32, so the whole stream is fp32
(tools/probe.py). The reference runs about six elementwise passes forward and as many backward
over the (tokens, 1536) tensor; here each direction is one pass, and the only thing kept for the
backward besides the input is one fp32 number per row (1 / rms). One kernel, two branches
(mode 0 forward, mode 1 backward), so it compiles once.
"""

import torch
import triton
import triton.language as tl

from .offload import OFFLOADER


@triton.jit(do_not_specialize=["rows", "hidden", "mode", "_i3", "_i4"])
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, dy_ptr, dx_ptr, _p6, _p7, _p8, _p9, rows, hidden,
                   mode, _i3, _i4, eps, BLOCK: tl.constexpr, _c1: tl.constexpr,
                   _c2: tl.constexpr, _c3: tl.constexpr):
    # (the argument layout shared by all our kernels: see kernels.py)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col = offs < hidden
    base = row.to(tl.int64) * hidden
    x = tl.load(x_ptr + base + offs, mask=col, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=col, other=0.0).to(tl.float32)
    if mode == 0:
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / hidden + eps)
        tl.store(y_ptr + base + offs, (w * (x * rstd)).to(y_ptr.dtype.element_ty), mask=col)
        tl.store(rstd_ptr + row, rstd)
    else:
        rstd = tl.load(rstd_ptr + row)
        dy = tl.load(dy_ptr + base + offs, mask=col, other=0.0).to(tl.float32)
        xhat = x * rstd
        g = dy * w
        mean_gx = tl.sum(g * xhat, axis=0) / hidden
        tl.store(dx_ptr + base + offs, (rstd * (g - xhat * mean_gx)).to(dx_ptr.dtype.element_ty),
                 mask=col)


class RMSNorm(torch.autograd.Function):
    """weight * (x * rsqrt(mean(x^2) + eps)) for an fp32 `x`; the result has the dtype of
    weight * x (fp32 here)."""

    @staticmethod
    def forward(ctx, x, weight, eps, to_half=False, offload=False):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        weight = weight.contiguous()
        rows, hidden = x2.shape
        # to_half: hand the result over already rounded to fp16. Every consumer in the autocast
        # step (q/k/v projections, MLP, lm_head) casts the fp32 result to fp16 anyway, so the
        # values are the same and the cast kernels (and their backward) disappear.
        dtype = torch.float16 if to_half else torch.promote_types(weight.dtype, x.dtype)
        out = torch.empty(x2.shape, dtype=dtype, device=x.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        # dy / dx arguments are unused in the forward branch; same dtypes as in the backward
        # call so that both directions are one compiled kernel
        rmsnorm_kernel[(rows,)](x2, weight, out, rstd, out, x2, *[x2] * 4, rows, hidden, 0, 0, 0,
                                eps, BLOCK=triton.next_power_of_2(hidden), _c1=0, _c2=0, _c3=0,
                                num_warps=8)
        if offload:  # the fp32 input goes to pinned host memory (offload.py)
            ctx.item = OFFLOADER.save(x2)
            ctx.save_for_backward(weight, rstd)
        else:
            ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        if hasattr(ctx, "item"):
            (x2,), (weight, rstd) = OFFLOADER.load(ctx.item), ctx.saved_tensors
        else:
            x2, weight, rstd = ctx.saved_tensors
        rows, hidden = x2.shape
        dy2 = dy.reshape(rows, hidden).contiguous()
        dx = torch.empty_like(x2)
        rmsnorm_kernel[(rows,)](x2, weight, dy2, rstd, dy2, dx, *[dx] * 4, rows, hidden, 1, 0, 0,
                                0.0, BLOCK=triton.next_power_of_2(hidden), _c1=0, _c2=0, _c3=0,
                                num_warps=8)
        dweight = None
        if ctx.needs_input_grad[1]:  # not the case in the task: the norm weights are frozen
            dweight = (dy2.float() * (x2.float() * rstd[:, None])).sum(dim=0).to(weight.dtype)
        return dx.reshape(ctx.shape), dweight, None, None, None


def rms_norm(x, weight, eps, to_half=False):
    offload = (x.requires_grad or weight.requires_grad) and OFFLOADER.wants(x)
    return RMSNorm.apply(x, weight, eps, to_half, offload)
