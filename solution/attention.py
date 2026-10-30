"""Q/K/V preparation for Qwen2 attention in three Triton launches each way.

The reference path (transformers' Qwen2Attention with SDPA) takes the fp32 q/k/v projections and runs
about 15 small ops forward: transpose, `q*cos + rotate_half(q)*sin` (slice, neg, cat, two muls, add)
for q and for k, the autocast casts to fp16 for scaled_dot_product_attention, and `repeat_kv`, which
copies the two key/value heads six times each (GQA: 12 query heads, 2 key/value heads); the
backward repeats all of it. At 28 layers that is thousands of launches per step on a GPU that is
partly idle waiting for the CPU. Here:

  forward : q  -> rotate -> fp16 (B, 12, T, 128)
            k  -> rotate -> fp16, written for all 12 heads (the 6 copies of each kv head)
            v  ->           fp16, written for all 12 heads
  backward: the inverse rotation for q and k, and the sum over the 6 copies for k and v,
            written back in the fp32 (B, T, heads * 128) layout of the projections.

The arithmetic is the reference arithmetic: the rotation is computed in fp32 and rounded once to
fp16, which is what the autocast cast after the rotation did.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_q_forward_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, seq, heads, scb, sct,
                          HALF: tl.constexpr):
    pid = tl.program_id(0)  # token index b * seq + t
    head = tl.program_id(1)
    b = pid // seq
    t = pid % seq
    offs = tl.arange(0, HALF)
    xb = x_ptr + pid * (heads * 2 * HALF) + head * 2 * HALF
    x1 = tl.load(xb + offs).to(tl.float32)
    x2 = tl.load(xb + HALF + offs).to(tl.float32)
    cb = b * scb + t * sct
    c1 = tl.load(cos_ptr + cb + offs)
    c2 = tl.load(cos_ptr + cb + HALF + offs)
    s1 = tl.load(sin_ptr + cb + offs)
    s2 = tl.load(sin_ptr + cb + HALF + offs)
    yb = y_ptr + ((b * heads + head) * seq + t) * 2 * HALF
    tl.store(yb + offs, (x1 * c1 - x2 * s1).to(y_ptr.dtype.element_ty))
    tl.store(yb + HALF + offs, (x2 * c2 + x1 * s2).to(y_ptr.dtype.element_ty))


@triton.jit
def rope_k_forward_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, seq, kv_heads, scb, sct,
                          HALF: tl.constexpr, REP: tl.constexpr):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    b = pid // seq
    t = pid % seq
    offs = tl.arange(0, HALF)
    xb = x_ptr + pid * (kv_heads * 2 * HALF) + head * 2 * HALF
    x1 = tl.load(xb + offs).to(tl.float32)
    x2 = tl.load(xb + HALF + offs).to(tl.float32)
    cb = b * scb + t * sct
    c1 = tl.load(cos_ptr + cb + offs)
    c2 = tl.load(cos_ptr + cb + HALF + offs)
    s1 = tl.load(sin_ptr + cb + offs)
    s2 = tl.load(sin_ptr + cb + HALF + offs)
    y1 = (x1 * c1 - x2 * s1).to(y_ptr.dtype.element_ty)
    y2 = (x2 * c2 + x1 * s2).to(y_ptr.dtype.element_ty)
    for r in range(REP):
        yb = y_ptr + (((b * kv_heads + head) * REP + r) * seq + t) * 2 * HALF
        tl.store(yb + offs, y1)
        tl.store(yb + HALF + offs, y2)


@triton.jit
def expand_v_forward_kernel(x_ptr, y_ptr, seq, kv_heads, HALF: tl.constexpr, REP: tl.constexpr):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    b = pid // seq
    t = pid % seq
    offs = tl.arange(0, 2 * HALF)
    x = tl.load(x_ptr + pid * (kv_heads * 2 * HALF) + head * 2 * HALF + offs).to(
        y_ptr.dtype.element_ty)
    for r in range(REP):
        yb = y_ptr + (((b * kv_heads + head) * REP + r) * seq + t) * 2 * HALF
        tl.store(yb + offs, x)


@triton.jit
def rope_q_backward_kernel(dy_ptr, cos_ptr, sin_ptr, dx_ptr, seq, heads, scb, sct,
                           HALF: tl.constexpr):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    b = pid // seq
    t = pid % seq
    offs = tl.arange(0, HALF)
    yb = dy_ptr + ((b * heads + head) * seq + t) * 2 * HALF
    dy1 = tl.load(yb + offs).to(tl.float32)
    dy2 = tl.load(yb + HALF + offs).to(tl.float32)
    cb = b * scb + t * sct
    c1 = tl.load(cos_ptr + cb + offs)
    c2 = tl.load(cos_ptr + cb + HALF + offs)
    s1 = tl.load(sin_ptr + cb + offs)
    s2 = tl.load(sin_ptr + cb + HALF + offs)
    xb = dx_ptr + pid * (heads * 2 * HALF) + head * 2 * HALF
    tl.store(xb + offs, (dy1 * c1 + dy2 * s2).to(dx_ptr.dtype.element_ty))
    tl.store(xb + HALF + offs, (dy2 * c2 - dy1 * s1).to(dx_ptr.dtype.element_ty))


@triton.jit
def rope_k_backward_kernel(dy_ptr, cos_ptr, sin_ptr, dx_ptr, seq, kv_heads, scb, sct,
                           HALF: tl.constexpr, REP: tl.constexpr):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    b = pid // seq
    t = pid % seq
    offs = tl.arange(0, HALF)
    dy1 = tl.zeros([HALF], tl.float32)
    dy2 = tl.zeros([HALF], tl.float32)
    for r in range(REP):
        yb = dy_ptr + (((b * kv_heads + head) * REP + r) * seq + t) * 2 * HALF
        dy1 += tl.load(yb + offs).to(tl.float32)
        dy2 += tl.load(yb + HALF + offs).to(tl.float32)
    cb = b * scb + t * sct
    c1 = tl.load(cos_ptr + cb + offs)
    c2 = tl.load(cos_ptr + cb + HALF + offs)
    s1 = tl.load(sin_ptr + cb + offs)
    s2 = tl.load(sin_ptr + cb + HALF + offs)
    xb = dx_ptr + pid * (kv_heads * 2 * HALF) + head * 2 * HALF
    tl.store(xb + offs, (dy1 * c1 + dy2 * s2).to(dx_ptr.dtype.element_ty))
    tl.store(xb + HALF + offs, (dy2 * c2 - dy1 * s1).to(dx_ptr.dtype.element_ty))


@triton.jit
def expand_v_backward_kernel(dy_ptr, dx_ptr, seq, kv_heads, HALF: tl.constexpr,
                             REP: tl.constexpr):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    b = pid // seq
    t = pid % seq
    offs = tl.arange(0, 2 * HALF)
    acc = tl.zeros([2 * HALF], tl.float32)
    for r in range(REP):
        yb = dy_ptr + (((b * kv_heads + head) * REP + r) * seq + t) * 2 * HALF
        acc += tl.load(yb + offs).to(tl.float32)
    tl.store(dx_ptr + pid * (kv_heads * 2 * HALF) + head * 2 * HALF + offs,
             acc.to(dx_ptr.dtype.element_ty))


class QKVRotate(torch.autograd.Function):
    """(q, k, v) projections (B, T, heads * D) -> rotated, fp16, GQA-expanded (B, H, T, D) each."""

    @staticmethod
    def forward(ctx, q, k, v, cos, sin, heads, kv_heads):
        batch, seq, width = q.shape
        dim = width // heads
        rep = heads // kv_heads
        half = dim // 2
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        cos = cos.expand(batch, seq, dim).contiguous()
        sin = sin.expand(batch, seq, dim).contiguous()
        scb, sct = cos.stride(0), cos.stride(1)
        qo = torch.empty(batch, heads, seq, dim, dtype=torch.float16, device=q.device)
        ko = torch.empty_like(qo)
        vo = torch.empty_like(qo)
        tokens = batch * seq
        rope_q_forward_kernel[(tokens, heads)](q, cos, sin, qo, seq, heads, scb, sct, HALF=half)
        rope_k_forward_kernel[(tokens, kv_heads)](k, cos, sin, ko, seq, kv_heads, scb, sct,
                                                  HALF=half, REP=rep)
        expand_v_forward_kernel[(tokens, kv_heads)](v, vo, seq, kv_heads, HALF=half, REP=rep)
        ctx.save_for_backward(cos, sin)
        ctx.meta = (batch, seq, heads, kv_heads, dim, q.dtype)
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dq, dk, dv):
        cos, sin = ctx.saved_tensors
        batch, seq, heads, kv_heads, dim, dtype = ctx.meta
        rep, half = heads // kv_heads, dim // 2
        scb, sct = cos.stride(0), cos.stride(1)
        tokens = batch * seq
        dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
        dq_in = torch.empty(batch, seq, heads * dim, dtype=dtype, device=dq.device)
        dk_in = torch.empty(batch, seq, kv_heads * dim, dtype=dtype, device=dq.device)
        dv_in = torch.empty_like(dk_in)
        rope_q_backward_kernel[(tokens, heads)](dq, cos, sin, dq_in, seq, heads, scb, sct,
                                                HALF=half)
        rope_k_backward_kernel[(tokens, kv_heads)](dk, cos, sin, dk_in, seq, kv_heads, scb, sct,
                                                   HALF=half, REP=rep)
        expand_v_backward_kernel[(tokens, kv_heads)](dv, dv_in, seq, kv_heads, HALF=half,
                                                     REP=rep)
        return dq_in, dk_in, dv_in, None, None, None, None


def qkv_rotate(q, k, v, cos, sin, heads, kv_heads):
    return QKVRotate.apply(q, k, v, cos, sin, heads, kv_heads)
