"""Our own Triton kernels and the autograd functions around them.

Why these ops (measured with tools/probe.py on a T4, batch of 1336 tokens):
- Soup runs Qwen2 with fp32 weights for the embeddings and norms, so the whole residual stream and
  the MLP intermediates are fp32 although every matmul runs in fp16 (the values are fp16-exact
  inside an fp32 container). The three (tokens, 8960) fp32 tensors each MLP layer keeps for the
  backward are 57% of everything saved for the backward.
- The loss upcasts the (tokens, 151936) fp16 logits to fp32 and saves the fp32 log-softmax.

The measured runs start with empty Triton caches and count the compile time, and a kernel costs
0.3-0.5 s to compile whatever its size (tools/compile_times.py). So each op here is ONE kernel
whose forward and backward are two branches chosen at run time, and integers that change from
batch to batch are not specialized (a new sequence length would otherwise compile it again).
"""

import torch
import triton
import triton.language as tl

from .offload import OFFLOADER

ACT_BLOCK = 1024
CE_BLOCK = 4096
NEG = tl.constexpr(-1.0e30)  # a finite "minus infinity": exp(NEG - x) is 0 and NEG - NEG never gives nan


# --------------------------------------------------------------------------------------------
# SwiGLU on fp16 tensors: out = silu(gate) * up, rounded to fp16 once.
# The reference path computes silu in fp32 on fp16-exact values and the next matmul casts the
# product to fp16, so rounding once at the end gives the same numbers as the fp32 reference.
# Grid (rows, column blocks); gate / up rows are `in_stride` apart, so they can be the two halves
# of one (rows, 2 * cols) matmul output (the fused gate + up projection, nf4.py), and the
# gradients are written in the same layout, ready for the fused backward matmul.
# mode 0 (forward):  p0 = gate, p1 = up, p2 = out
# mode 1 (backward): p0 = d_out, p1 = gate, p2 = up, p3 = d_gate, p4 = d_up
# --------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["cols", "in_stride", "mode"])
def silu_mul_kernel(p0, p1, p2, p3, p4, cols, in_stride, mode, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < cols
    at_in = row * in_stride + col  # gate, up and their gradients
    at_out = row * cols + col  # out and its gradient: contiguous rows
    if mode == 0:
        gate = tl.load(p0 + at_in, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(p1 + at_in, mask=mask, other=0.0).to(tl.float32)
        tl.store(p2 + at_out, (gate * tl.sigmoid(gate) * up).to(p2.dtype.element_ty), mask=mask)
    else:
        dout = tl.load(p0 + at_out, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(p1 + at_in, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(p2 + at_in, mask=mask, other=0.0).to(tl.float32)
        sig = tl.sigmoid(gate)
        tl.store(p4 + at_in, (dout * gate * sig).to(p4.dtype.element_ty), mask=mask)
        tl.store(p3 + at_in, (dout * up * sig * (1.0 + gate * (1.0 - sig))).to(
            p3.dtype.element_ty), mask=mask)


def _swiglu(p0, p1, p2, p3, p4, rows, cols, in_stride, mode):
    silu_mul_kernel[(rows, triton.cdiv(cols, ACT_BLOCK))](p0, p1, p2, p3, p4, cols, in_stride,
                                                          mode, BLOCK=ACT_BLOCK)


class SiluMul(torch.autograd.Function):
    """silu(gate) * up on fp16 tensors; keeps only gate and up (both fp16) for the backward,
    in pinned host memory when `offload` (offload.py)."""

    @staticmethod
    def forward(ctx, gate, up, offload):
        gate, up = gate.contiguous(), up.contiguous()
        cols = gate.shape[-1]
        rows = gate.numel() // cols
        out = torch.empty_like(gate)
        _swiglu(gate, up, out, out, out, rows, cols, cols, 0)
        if offload:
            ctx.item = OFFLOADER.save(gate, up)
        else:
            ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, dout):
        if hasattr(ctx, "item"):
            gate, up = OFFLOADER.load(ctx.item)
        else:
            gate, up = ctx.saved_tensors
        dout = dout.contiguous()
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        cols = gate.shape[-1]
        _swiglu(dout, gate, up, dgate, dup, gate.numel() // cols, cols, cols, 1)
        return dgate, dup, None


class SiluMulPacked(torch.autograd.Function):
    """The same on one (rows, 2 * cols) tensor holding [gate | up] per row (the fused gate + up
    matmul): (rows, cols) out; the gradient comes back in the same [d_gate | d_up] layout."""

    @staticmethod
    def forward(ctx, gate_up, offload):
        gate_up = gate_up.contiguous()
        rows, both = gate_up.shape
        cols = both // 2
        out = torch.empty(rows, cols, dtype=gate_up.dtype, device=gate_up.device)
        _swiglu(gate_up, gate_up[:, cols:], out, out, out, rows, cols, both, 0)
        if offload:
            ctx.item = OFFLOADER.save(gate_up)
        else:
            ctx.save_for_backward(gate_up)
        return out

    @staticmethod
    def backward(ctx, dout):
        (gate_up,) = OFFLOADER.load(ctx.item) if hasattr(ctx, "item") else ctx.saved_tensors
        rows, both = gate_up.shape
        cols = both // 2
        dout = dout.contiguous()
        grad = torch.empty_like(gate_up)
        _swiglu(dout, gate_up, gate_up[:, cols:], grad, grad[:, cols:], rows, cols, both, 1)
        return grad, None


def silu_mul(gate, up):
    offload = (gate.requires_grad or up.requires_grad) and OFFLOADER.wants(gate, up)
    return SiluMul.apply(gate, up, offload)


def silu_mul_packed(gate_up):
    """silu(gate_up[..., :c]) * gate_up[..., c:] for c = half the last dimension."""
    lead = gate_up.shape[:-1]
    flat = gate_up.reshape(-1, gate_up.shape[-1])
    offload = flat.requires_grad and OFFLOADER.wants(flat)
    return SiluMulPacked.apply(flat, offload).reshape(*lead, -1)


# --------------------------------------------------------------------------------------------
# Cross-entropy over the vocabulary, straight from the fp16 logits.
# One program per token row: a single pass keeps a running max, sum and sum of exp * logit (online
# log-sum-exp), so no fp32 copy of the logits and no saved log-softmax exist; only one fp32 number
# per row is kept. The same pass gives the entropy of the softmax (TRL logs it) and the argmax
# (first maximal index, like torch.argmax; TRL's token accuracy) when asked: `want` bit 1 and 2.
# --------------------------------------------------------------------------------------------
@triton.jit
def _ce_stats(logits_ptr, target_ptr, lse_ptr, loss_ptr, entropy_ptr, argmax_ptr, vocab, want,
              BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(target_ptr + row)
    base = logits_ptr + row * vocab
    offs = tl.arange(0, BLOCK)
    run_max = tl.full([BLOCK], NEG, tl.float32)
    run_sum = tl.zeros([BLOCK], tl.float32)
    run_dot = tl.zeros([BLOCK], tl.float32)
    run_idx = tl.zeros([BLOCK], tl.int32)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        mask = idx < vocab
        x = tl.load(base + idx, mask=mask, other=NEG).to(tl.float32)
        run_idx = tl.where(x > run_max, idx, run_idx)  # first maximum of this lane
        new_max = tl.maximum(run_max, x)
        rescale = tl.exp(run_max - new_max)
        weight = tl.exp(x - new_max)
        run_sum = run_sum * rescale + weight
        run_dot = run_dot * rescale + weight * tl.where(mask, x, 0.0)
        run_max = new_max
    row_max = tl.max(run_max, axis=0)
    align = tl.exp(run_max - row_max)
    total = tl.sum(run_sum * align, axis=0)
    lse = row_max + tl.log(total)
    picked = tl.load(base + tl.maximum(target, 0)).to(tl.float32)
    valid = target >= 0
    tl.store(lse_ptr + row, tl.where(valid, lse, 0.0))
    tl.store(loss_ptr + row, tl.where(valid, lse - picked, 0.0))
    if want % 2 == 1:
        dot = tl.sum(run_dot * align, axis=0)
        # entropy of softmax(x) = logsumexp(x) - E[x]
        tl.store(entropy_ptr + row, lse - dot / total)
    if want >= 2:
        first = tl.min(tl.where(run_max == row_max, run_idx, 2147483647), axis=0)
        tl.store(argmax_ptr + row, first.to(tl.int64))


@triton.jit
def _ce_backward(logits_ptr, target_ptr, lse_ptr, scale_ptr, grad_ptr, vocab, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(target_ptr + row)
    lse = tl.load(lse_ptr + row)
    scale = tl.load(scale_ptr)
    base = logits_ptr + row * vocab
    out = grad_ptr + row * vocab
    offs = tl.arange(0, BLOCK)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        mask = idx < vocab
        x = tl.load(base + idx, mask=mask, other=NEG).to(tl.float32)
        prob = tl.exp(x - lse)
        hot = (idx == target).to(tl.float32)
        grad = tl.where(target >= 0, (prob - hot) * scale, 0.0)
        tl.store(out + idx, grad.to(grad_ptr.dtype.element_ty), mask=mask)


# One compiled kernel for both passes (each compile costs 0.3-0.6 s of the measured run): mode
# 0-3 is the statistics pass with `want` = mode, mode 4 the gradient pass. Arguments one pass does
# not use get tensors of the same dtypes, so both calls share the compiled kernel.
@triton.jit(do_not_specialize=["mode"])
def ce_kernel(logits_ptr, target_ptr, lse_ptr, loss_ptr, entropy_ptr, argmax_ptr, scale_ptr,
              grad_ptr, vocab, mode, BLOCK: tl.constexpr):
    if mode == 4:
        _ce_backward(logits_ptr, target_ptr, lse_ptr, scale_ptr, grad_ptr, vocab, BLOCK)
    else:
        _ce_stats(logits_ptr, target_ptr, lse_ptr, loss_ptr, entropy_ptr, argmax_ptr, vocab, mode,
                  BLOCK)


def ce_grad(logits, targets, lse, scale, grad):
    """grad = scale * (softmax(logits) - onehot(targets)) per row, 0 for rows with target < 0;
    `grad` may be `logits` itself (each element is read before it is overwritten)."""
    rows, vocab = logits.shape
    ce_kernel[(rows,)](logits, targets, lse, lse, lse, targets, scale, grad, vocab, 4,
                       BLOCK=CE_BLOCK, num_warps=8)


def ce_stats(logits, targets, want_entropy, out=None):
    """Per-row logsumexp, loss (0 for rows with target < 0) and, if asked, entropy of (rows,
    vocab) logits. `out` = (lse, loss, entropy, argmax) buffers to fill, argmax included."""
    rows, vocab = logits.shape
    if out is not None:
        lse, loss, entropy, argmax = out
        want = 3
    else:
        lse = torch.empty(rows, dtype=torch.float32, device=logits.device)
        loss = torch.empty_like(lse)
        entropy = torch.empty_like(lse) if want_entropy else lse
        argmax = targets  # not written
        want = int(want_entropy)
    scale = lse  # unused by the statistics pass
    ce_kernel[(rows,)](logits, targets, lse, loss, entropy, argmax, scale, logits, vocab, want,
                       BLOCK=CE_BLOCK, num_warps=8)
    return lse, loss, (entropy if want % 2 else None)


class FusedCrossEntropy(torch.autograd.Function):
    """sum over rows of (logsumexp - logit[target]) / denom, rows with target < 0 ignored.

    With `inplace_grad` the gradient is written over the logits themselves (each element is read
    once and then replaced by its own gradient, so the kernel is safe in place), which saves a
    second (rows, vocab) buffer at the peak of the step. Only for logits nothing reads after the
    backward pass."""

    @staticmethod
    def forward(ctx, logits, targets, denom, inplace_grad):
        lse, losses, _ = ce_stats(logits, targets, False)
        ctx.save_for_backward(logits, targets, lse, denom)
        ctx.inplace_grad = inplace_grad
        return losses.sum() / denom

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, lse, denom = ctx.saved_tensors
        rows, vocab = logits.shape
        scale = (grad_output / denom).to(torch.float32).reshape(1)
        grad = logits if ctx.inplace_grad else torch.empty_like(logits)
        ce_grad(logits, targets, lse, scale, grad)
        return grad, None, None, None


def fused_cross_entropy(logits, targets, denom, inplace_grad=False):
    """`logits` (rows, vocab) fp16/fp32, `targets` (rows,) int64 with -100 for ignored rows,
    `denom` a 0-d fp32 tensor on the same device."""
    return FusedCrossEntropy.apply(logits.contiguous(), targets, denom, inplace_grad)
