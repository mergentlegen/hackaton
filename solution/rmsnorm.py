"""RMSNorm on the fp32 residual stream, one Triton pass each way.

In Soup's 4-bit setup the embeddings and norm weights are fp32, so the whole stream is fp32
(tools/probe.py). The reference runs about six elementwise passes forward and as many backward
over the (tokens, 1536) tensor; here each direction is one pass, and the only thing kept for the
backward besides the input is one fp32 number per row (1 / rms).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_forward_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, hidden, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col = offs < hidden
    base = row.to(tl.int64) * hidden
    x = tl.load(x_ptr + base + offs, mask=col, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / hidden + eps)
    w = tl.load(w_ptr + offs, mask=col, other=0.0).to(tl.float32)
    tl.store(y_ptr + base + offs, (w * (x * rstd)).to(y_ptr.dtype.element_ty), mask=col)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_backward_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_ptr, rows, hidden,
                            BLOCK: tl.constexpr, ROWS: tl.constexpr, NEED_DW: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col = offs < hidden
    w = tl.load(w_ptr + offs, mask=col, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK], tl.float32)
    for i in range(ROWS):
        row = pid * ROWS + i
        live = col & (row < rows)
        base = row.to(tl.int64) * hidden
        x = tl.load(x_ptr + base + offs, mask=live, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + base + offs, mask=live, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row, mask=row < rows, other=0.0)
        xhat = x * rstd
        g = dy * w
        mean_gx = tl.sum(g * xhat, axis=0) / hidden
        dx = rstd * (g - xhat * mean_gx)
        tl.store(dx_ptr + base + offs, dx.to(dx_ptr.dtype.element_ty), mask=live)
        if NEED_DW:
            dw_acc += dy * xhat
    if NEED_DW:
        tl.store(dw_ptr + pid * hidden + offs, dw_acc, mask=col)


class RMSNorm(torch.autograd.Function):
    """weight * (x * rsqrt(mean(x^2) + eps)) for an fp32 `x`; the result has the dtype of
    weight * x (fp32 here)."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        weight = weight.contiguous()
        rows, hidden = x2.shape
        out = torch.empty(x2.shape, dtype=torch.promote_types(weight.dtype, x.dtype),
                          device=x.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        rmsnorm_forward_kernel[(rows,)](x2, weight, out, rstd, hidden, eps,
                                        BLOCK=triton.next_power_of_2(hidden), num_warps=8)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        rows, hidden = x2.shape
        dy2 = dy.reshape(rows, hidden).contiguous()
        dx = torch.empty_like(x2)
        need_dw = ctx.needs_input_grad[1]
        per_program = 4 if need_dw else 1
        programs = triton.cdiv(rows, per_program)
        partial = (torch.empty(programs, hidden, dtype=torch.float32, device=x2.device)
                   if need_dw else dx)
        rmsnorm_backward_kernel[(programs,)](dy2, x2, weight, rstd, dx, partial, rows, hidden,
                                             BLOCK=triton.next_power_of_2(hidden),
                                             ROWS=per_program, NEED_DW=need_dw, num_warps=8)
        dweight = partial.sum(dim=0).to(weight.dtype) if need_dw else None
        return dx.reshape(ctx.shape), dweight, None


def rms_norm(x, weight, eps):
    return RMSNorm.apply(x, weight, eps)
