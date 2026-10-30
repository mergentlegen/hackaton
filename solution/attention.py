"""Q/K/V preparation for Qwen2 attention in one Triton launch each way.

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
fp16, which is what the autocast cast after the rotation did. q, k and v are three branches of one
kernel (grid axis 1 picks the head slot), so the forward compiles once and so does the backward.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["seq", "scb", "sct"])
def qkv_forward_kernel(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr, vo_ptr, seq, scb,
                       sct, HEADS: tl.constexpr, KV: tl.constexpr, REP: tl.constexpr,
                       HALF: tl.constexpr):
    pid = tl.program_id(0)  # token index b * seq + t
    slot = tl.program_id(1)  # q heads, then k heads, then v heads
    b = pid // seq
    t = pid % seq
    offs = tl.arange(0, HALF)
    cb = b * scb + t * sct
    c1 = tl.load(cos_ptr + cb + offs)
    c2 = tl.load(cos_ptr + cb + HALF + offs)
    s1 = tl.load(sin_ptr + cb + offs)
    s2 = tl.load(sin_ptr + cb + HALF + offs)
    if slot < HEADS:
        xb = q_ptr + pid * (HEADS * 2 * HALF) + slot * 2 * HALF
        x1 = tl.load(xb + offs).to(tl.float32)
        x2 = tl.load(xb + HALF + offs).to(tl.float32)
        yb = qo_ptr + ((b * HEADS + slot) * seq + t) * 2 * HALF
        tl.store(yb + offs, (x1 * c1 - x2 * s1).to(qo_ptr.dtype.element_ty))
        tl.store(yb + HALF + offs, (x2 * c2 + x1 * s2).to(qo_ptr.dtype.element_ty))
    elif slot < HEADS + KV:
        head = slot - HEADS
        xb = k_ptr + pid * (KV * 2 * HALF) + head * 2 * HALF
        x1 = tl.load(xb + offs).to(tl.float32)
        x2 = tl.load(xb + HALF + offs).to(tl.float32)
        y1 = (x1 * c1 - x2 * s1).to(ko_ptr.dtype.element_ty)
        y2 = (x2 * c2 + x1 * s2).to(ko_ptr.dtype.element_ty)
        for r in range(REP):
            yb = ko_ptr + (((b * KV + head) * REP + r) * seq + t) * 2 * HALF
            tl.store(yb + offs, y1)
            tl.store(yb + HALF + offs, y2)
    else:
        head = slot - HEADS - KV
        xb = v_ptr + pid * (KV * 2 * HALF) + head * 2 * HALF
        v1 = tl.load(xb + offs).to(vo_ptr.dtype.element_ty)
        v2 = tl.load(xb + HALF + offs).to(vo_ptr.dtype.element_ty)
        for r in range(REP):
            yb = vo_ptr + (((b * KV + head) * REP + r) * seq + t) * 2 * HALF
            tl.store(yb + offs, v1)
            tl.store(yb + HALF + offs, v2)


@triton.jit(do_not_specialize=["seq", "scb", "sct"])
def qkv_backward_kernel(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, dqi_ptr, dki_ptr, dvi_ptr, seq,
                        scb, sct, HEADS: tl.constexpr, KV: tl.constexpr, REP: tl.constexpr,
                        HALF: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    b = pid // seq
    t = pid % seq
    offs = tl.arange(0, HALF)
    cb = b * scb + t * sct
    c1 = tl.load(cos_ptr + cb + offs)
    c2 = tl.load(cos_ptr + cb + HALF + offs)
    s1 = tl.load(sin_ptr + cb + offs)
    s2 = tl.load(sin_ptr + cb + HALF + offs)
    if slot < HEADS:
        yb = dq_ptr + ((b * HEADS + slot) * seq + t) * 2 * HALF
        dy1 = tl.load(yb + offs).to(tl.float32)
        dy2 = tl.load(yb + HALF + offs).to(tl.float32)
        xb = dqi_ptr + pid * (HEADS * 2 * HALF) + slot * 2 * HALF
        tl.store(xb + offs, (dy1 * c1 + dy2 * s2).to(dqi_ptr.dtype.element_ty))
        tl.store(xb + HALF + offs, (dy2 * c2 - dy1 * s1).to(dqi_ptr.dtype.element_ty))
    elif slot < HEADS + KV:
        head = slot - HEADS
        dy1 = tl.zeros([HALF], tl.float32)
        dy2 = tl.zeros([HALF], tl.float32)
        for r in range(REP):
            yb = dk_ptr + (((b * KV + head) * REP + r) * seq + t) * 2 * HALF
            dy1 += tl.load(yb + offs).to(tl.float32)
            dy2 += tl.load(yb + HALF + offs).to(tl.float32)
        xb = dki_ptr + pid * (KV * 2 * HALF) + head * 2 * HALF
        tl.store(xb + offs, (dy1 * c1 + dy2 * s2).to(dki_ptr.dtype.element_ty))
        tl.store(xb + HALF + offs, (dy2 * c2 - dy1 * s1).to(dki_ptr.dtype.element_ty))
    else:
        head = slot - HEADS - KV
        d1 = tl.zeros([HALF], tl.float32)
        d2 = tl.zeros([HALF], tl.float32)
        for r in range(REP):
            yb = dv_ptr + (((b * KV + head) * REP + r) * seq + t) * 2 * HALF
            d1 += tl.load(yb + offs).to(tl.float32)
            d2 += tl.load(yb + HALF + offs).to(tl.float32)
        xb = dvi_ptr + pid * (KV * 2 * HALF) + head * 2 * HALF
        tl.store(xb + offs, d1.to(dvi_ptr.dtype.element_ty))
        tl.store(xb + HALF + offs, d2.to(dvi_ptr.dtype.element_ty))


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
        qo = torch.empty(batch, heads, seq, dim, dtype=torch.float16, device=q.device)
        ko = torch.empty_like(qo)
        vo = torch.empty_like(qo)
        qkv_forward_kernel[(batch * seq, heads + 2 * kv_heads)](
            q, k, v, cos, sin, qo, ko, vo, seq, cos.stride(0), cos.stride(1),
            HEADS=heads, KV=kv_heads, REP=rep, HALF=half, num_warps=2)
        ctx.save_for_backward(cos, sin)
        ctx.meta = (batch, seq, heads, kv_heads, dim, q.dtype)
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dq, dk, dv):
        cos, sin = ctx.saved_tensors
        batch, seq, heads, kv_heads, dim, dtype = ctx.meta
        rep, half = heads // kv_heads, dim // 2
        dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
        dq_in = torch.empty(batch, seq, heads * dim, dtype=dtype, device=dq.device)
        dk_in = torch.empty(batch, seq, kv_heads * dim, dtype=dtype, device=dq.device)
        dv_in = torch.empty_like(dk_in)
        qkv_backward_kernel[(batch * seq, heads + 2 * kv_heads)](
            dq, dk, dv, cos, sin, dq_in, dk_in, dv_in, seq, cos.stride(0), cos.stride(1),
            HEADS=heads, KV=kv_heads, REP=rep, HALF=half, num_warps=2)
        return dq_in, dk_in, dv_in, None, None, None, None


def qkv_rotate(q, k, v, cos, sin, heads, kv_heads):
    return QKVRotate.apply(q, k, v, cos, sin, heads, kv_heads)
