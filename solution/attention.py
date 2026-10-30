"""Attention kernels for the packed (padding-free) forward.

A batch is 4 sequences padded to the longest, and on this data 43% of its positions are padding
(mean length 207 tokens, but the longest of four is ~360). The packed forward (packed.py) runs
every per-token op (projections, norms, MLP, LoRA, lm_head, loss) on the N real tokens only; only
scaled_dot_product_attention still sees the padded (B, heads, T, D) layout, because it needs one
sequence per row. These kernels move between the two layouts and do the rest of the attention
preparation in the same pass.

QKV rotate (one launch forward, one backward; the reference runs ~15 small ops each way):
  forward : q -> rotate -> fp16 (B, 12, T, 128)
            k -> rotate -> fp16, written for all 12 heads (the 6 copies of each kv head)
            v ->           fp16, written for all 12 heads
            (padding positions stay zero)
  backward: the inverse rotation for q and k, and the sum over the 6 copies for k and v,
            read from the padded gradients and written back packed, fp32, (N, heads * 128).
The arithmetic is the reference arithmetic: the rotation is computed in fp32 and rounded once to
fp16, which is what the autocast cast after the rotation did.

Pack / unpack: the attention output (B, 12, T, 128) -> (N, 1536) and its gradient back.

Every kernel is one compile with a run-time branch (grid axis 1 picks q, k or v; `mode` picks the
direction), because each compile costs 0.3-0.5 s in a run that starts with empty caches.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["seq", "scb", "sct"])
def qkv_forward_kernel(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, b_ptr, t_ptr, qo_ptr, ko_ptr,
                       vo_ptr, seq, scb, sct, HEADS: tl.constexpr, KV: tl.constexpr,
                       REP: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)  # packed token index
    slot = tl.program_id(1)  # q heads, then k heads, then v heads
    b = tl.load(b_ptr + pid)
    t = tl.load(t_ptr + pid)
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
def qkv_backward_kernel(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, b_ptr, t_ptr, dqi_ptr, dki_ptr,
                        dvi_ptr, seq, scb, sct, HEADS: tl.constexpr, KV: tl.constexpr,
                        REP: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    slot = tl.program_id(1)
    b = tl.load(b_ptr + pid)
    t = tl.load(t_ptr + pid)
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
    """Packed (q, k, v) projections (N, heads * D) -> rotated, fp16, GQA-expanded (B, H, T, D)."""

    @staticmethod
    def forward(ctx, q, k, v, cos, sin, b_idx, t_idx, batch, seq, heads, kv_heads):
        dim = q.shape[-1] // heads
        rep, half = heads // kv_heads, dim // 2
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        cos, sin = cos.contiguous(), sin.contiguous()
        scb = 0 if cos.shape[0] == 1 else cos.stride(0)
        qo = torch.zeros(batch, heads, seq, dim, dtype=torch.float16, device=q.device)
        ko = torch.zeros_like(qo)
        vo = torch.zeros_like(qo)
        qkv_forward_kernel[(q.shape[0], heads + 2 * kv_heads)](
            q, k, v, cos, sin, b_idx, t_idx, qo, ko, vo, seq, scb, cos.stride(1),
            HEADS=heads, KV=kv_heads, REP=rep, HALF=half, num_warps=2)
        ctx.save_for_backward(cos, sin, b_idx, t_idx)
        ctx.meta = (seq, scb, heads, kv_heads, rep, half, q.dtype)
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dq, dk, dv):
        cos, sin, b_idx, t_idx = ctx.saved_tensors
        seq, scb, heads, kv_heads, rep, half, dtype = ctx.meta
        dim = 2 * half
        tokens = b_idx.shape[0]
        dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
        dq_in = torch.empty(tokens, heads * dim, dtype=dtype, device=dq.device)
        dk_in = torch.empty(tokens, kv_heads * dim, dtype=dtype, device=dq.device)
        dv_in = torch.empty_like(dk_in)
        qkv_backward_kernel[(tokens, heads + 2 * kv_heads)](
            dq, dk, dv, cos, sin, b_idx, t_idx, dq_in, dk_in, dv_in, seq, scb, cos.stride(1),
            HEADS=heads, KV=kv_heads, REP=rep, HALF=half, num_warps=2)
        return dq_in, dk_in, dv_in, None, None, None, None, None, None, None, None


def qkv_rotate(q, k, v, cos, sin, b_idx, t_idx, batch, seq, heads, kv_heads):
    return QKVRotate.apply(q, k, v, cos, sin, b_idx, t_idx, batch, seq, heads, kv_heads)


@triton.jit(do_not_specialize=["seq", "mode"])
def pack_kernel(src_ptr, dst_ptr, b_ptr, t_ptr, seq, mode, HEADS: tl.constexpr, DIM: tl.constexpr):
    pid = tl.program_id(0)
    head = tl.program_id(1)
    b = tl.load(b_ptr + pid)
    t = tl.load(t_ptr + pid)
    offs = tl.arange(0, DIM)
    padded = ((b * HEADS + head) * seq + t) * DIM
    packed = pid * (HEADS * DIM) + head * DIM
    if mode == 0:  # padded (B, H, T, D) -> packed (N, H * D)
        tl.store(dst_ptr + packed + offs, tl.load(src_ptr + padded + offs))
    else:  # packed gradient (N, H * D) -> padded (B, H, T, D), zeros elsewhere
        tl.store(dst_ptr + padded + offs, tl.load(src_ptr + packed + offs))


class PackRows(torch.autograd.Function):
    """(B, H, T, D) fp16 attention output -> (N, H * D) rows of the real tokens."""

    @staticmethod
    def forward(ctx, attn, b_idx, t_idx):
        batch, heads, seq, dim = attn.shape
        attn = attn.contiguous()
        out = torch.empty(b_idx.shape[0], heads * dim, dtype=attn.dtype, device=attn.device)
        pack_kernel[(b_idx.shape[0], heads)](attn, out, b_idx, t_idx, seq, 0, HEADS=heads,
                                             DIM=dim, num_warps=4)
        ctx.save_for_backward(b_idx, t_idx)
        ctx.meta = (batch, heads, seq, dim)
        return out

    @staticmethod
    def backward(ctx, grad):
        b_idx, t_idx = ctx.saved_tensors
        batch, heads, seq, dim = ctx.meta
        grad = grad.contiguous()
        padded = torch.zeros(batch, heads, seq, dim, dtype=grad.dtype, device=grad.device)
        pack_kernel[(b_idx.shape[0], heads)](grad, padded, b_idx, t_idx, seq, 1, HEADS=heads,
                                             DIM=dim, num_warps=4)
        return padded, None, None


def pack_rows(attn, b_idx, t_idx):
    return PackRows.apply(attn, b_idx, t_idx)
