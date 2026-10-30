"""Our own Triton kernels and the autograd functions around them.

Why these two ops (measured with tools/probe.py on a T4, batch of 1336 tokens):
- Soup runs Qwen2 with fp32 weights for the embeddings and norms, so the whole residual stream and
  the MLP intermediates are fp32 although every matmul runs in fp16 (the values are fp16-exact
  inside an fp32 container). The three (tokens, 8960) fp32 tensors each MLP layer keeps for the
  backward are 57% of everything saved for the backward.
- The loss upcasts the (tokens, 151936) fp16 logits to fp32 and saves the fp32 log-softmax.
"""

import torch
import triton
import triton.language as tl

ACT_BLOCK = 1024
CE_BLOCK = 4096
NEG = tl.constexpr(-1.0e30)  # a finite "minus infinity": exp(NEG - x) is 0 and NEG - NEG never gives nan


# --------------------------------------------------------------------------------------------
# SwiGLU on fp16 tensors: out = silu(gate) * up, rounded to fp16 once.
# The reference path computes silu in fp32 on fp16-exact values and the next matmul casts the
# product to fp16, so rounding once at the end gives the same numbers as the fp32 reference.
# --------------------------------------------------------------------------------------------
@triton.jit
def silu_mul_forward_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    act = gate * tl.sigmoid(gate)
    tl.store(out_ptr + offs, (act * up).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def silu_mul_backward_kernel(dout_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements,
                             BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dout = tl.load(dout_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    act = gate * sig
    tl.store(dup_ptr + offs, (dout * act).to(dup_ptr.dtype.element_ty), mask=mask)
    dact = dout * up
    tl.store(dgate_ptr + offs, (dact * sig * (1.0 + gate * (1.0 - sig))).to(
        dgate_ptr.dtype.element_ty), mask=mask)


class SiluMul(torch.autograd.Function):
    """silu(gate) * up on fp16 tensors; keeps only gate and up (both fp16) for the backward."""

    @staticmethod
    def forward(ctx, gate, up):
        gate, up = gate.contiguous(), up.contiguous()
        out = torch.empty_like(gate)
        n = gate.numel()
        silu_mul_forward_kernel[(triton.cdiv(n, ACT_BLOCK),)](gate, up, out, n, BLOCK=ACT_BLOCK)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, dout):
        gate, up = ctx.saved_tensors
        dout = dout.contiguous()
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        silu_mul_backward_kernel[(triton.cdiv(n, ACT_BLOCK),)](dout, gate, up, dgate, dup, n,
                                                              BLOCK=ACT_BLOCK)
        return dgate, dup


def silu_mul(gate, up):
    return SiluMul.apply(gate, up)


# --------------------------------------------------------------------------------------------
# Cross-entropy over the vocabulary, straight from the fp16 logits.
# One program per token row: a single pass keeps a running max and sum (online log-sum-exp), so
# no fp32 copy of the logits and no saved log-softmax exist; only one fp32 number per row is kept.
# --------------------------------------------------------------------------------------------
@triton.jit
def ce_forward_kernel(logits_ptr, target_ptr, lse_ptr, loss_ptr, vocab, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(target_ptr + row)
    base = logits_ptr + row * vocab
    offs = tl.arange(0, BLOCK)
    run_max = tl.full([BLOCK], NEG, tl.float32)
    run_sum = tl.zeros([BLOCK], tl.float32)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        x = tl.load(base + idx, mask=idx < vocab, other=NEG).to(tl.float32)
        new_max = tl.maximum(run_max, x)
        run_sum = run_sum * tl.exp(run_max - new_max) + tl.exp(x - new_max)
        run_max = new_max
    row_max = tl.max(run_max, axis=0)
    total = tl.sum(run_sum * tl.exp(run_max - row_max), axis=0)
    lse = row_max + tl.log(total)
    picked = tl.load(base + tl.maximum(target, 0)).to(tl.float32)
    valid = target >= 0
    tl.store(lse_ptr + row, tl.where(valid, lse, 0.0))
    tl.store(loss_ptr + row, tl.where(valid, lse - picked, 0.0))


@triton.jit
def ce_backward_kernel(logits_ptr, target_ptr, lse_ptr, scale_ptr, grad_ptr, vocab,
                       BLOCK: tl.constexpr):
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


class FusedCrossEntropy(torch.autograd.Function):
    """sum over rows of (logsumexp - logit[target]) / denom, rows with target < 0 ignored."""

    @staticmethod
    def forward(ctx, logits, targets, denom):
        rows, vocab = logits.shape
        lse = torch.empty(rows, dtype=torch.float32, device=logits.device)
        losses = torch.empty_like(lse)
        ce_forward_kernel[(rows,)](logits, targets, lse, losses, vocab, BLOCK=CE_BLOCK,
                                   num_warps=8)
        ctx.save_for_backward(logits, targets, lse, denom)
        return losses.sum() / denom

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, lse, denom = ctx.saved_tensors
        rows, vocab = logits.shape
        scale = (grad_output / denom).to(torch.float32).reshape(1)
        grad = torch.empty_like(logits)
        ce_backward_kernel[(rows,)](logits, targets, lse, scale, grad, vocab, BLOCK=CE_BLOCK,
                                    num_warps=8)
        return grad, None, None


def fused_cross_entropy(logits, targets, denom):
    """`logits` (rows, vocab) fp16/fp32, `targets` (rows,) int64 with -100 for ignored rows,
    `denom` a 0-d fp32 tensor on the same device."""
    return FusedCrossEntropy.apply(logits.contiguous(), targets, denom)
