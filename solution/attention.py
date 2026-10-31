"""Attention kernels for the packed (padding-free) forward.

A batch is 4 sequences padded to the longest, and on this data 43% of its positions are padding
(mean length 207 tokens, but the longest of four is ~360). The packed forward (packed.py) runs
every per-token op (projections, norms, MLP, LoRA, lm_head, loss) on the N real tokens only.

Attention itself runs the memory-efficient attention kernel once over the packed rows with the
sequence boundaries (`varlen_attention`), so it never computes, stores or differentiates padding
positions: with the padding on the right a sequence's real tokens are one contiguous block of the
packed rows. Only a
batch with another padding layout falls back to the padded (B, heads, T, D) layout and
transformers' mask (`layout` 1 below, plus `pack_rows`).

QKV rotate (one launch forward, one backward; the reference runs ~15 small ops each way):
  forward : q -> rotate -> fp16
            k -> rotate -> fp16, written for all 12 heads (the 6 copies of each kv head)
            v ->           fp16, written for all 12 heads
            packed layout (N, heads, D), or padded layout (B, heads, T, D) with zeros elsewhere
  backward: the inverse rotation for q and k, and the sum over the 6 copies for k and v,
            written back packed, (N, heads * D), in the dtype of the projections.
The arithmetic is the reference arithmetic: the rotation is computed in fp32 and rounded once to
fp16, which is what the autocast cast after the rotation did.

Every kernel is one compile with run-time branches (grid axis 1 picks q, k or v; `layout` and
`mode` are plain integers), because each compile costs 0.3-0.5 s in a run that starts with empty
caches.
"""

import torch
import triton
import triton.language as tl

from .offload import OFFLOADER


@triton.jit
def _qkv_forward(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, b_ptr, t_ptr, qo_ptr, ko_ptr, vo_ptr, seq,
                 scb, sct, layout, HEADS: tl.constexpr, KV: tl.constexpr, REP: tl.constexpr,
                 HALF: tl.constexpr):
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
    # output head h of this token: packed (N, H, D) when layout == 0, padded (B, H, T, D) when 1
    packed_base = pid * HEADS
    padded_base = b * HEADS * seq + t
    padded_step = seq
    if layout == 0:
        padded_base = packed_base
        padded_step = 1
    if slot < HEADS:
        xb = q_ptr + pid * (HEADS * 2 * HALF) + slot * 2 * HALF
        x1 = tl.load(xb + offs).to(tl.float32)
        x2 = tl.load(xb + HALF + offs).to(tl.float32)
        yb = qo_ptr + (padded_base + slot * padded_step) * 2 * HALF
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
            yb = ko_ptr + (padded_base + (head * REP + r) * padded_step) * 2 * HALF
            tl.store(yb + offs, y1)
            tl.store(yb + HALF + offs, y2)
    else:
        head = slot - HEADS - KV
        xb = v_ptr + pid * (KV * 2 * HALF) + head * 2 * HALF
        v1 = tl.load(xb + offs).to(vo_ptr.dtype.element_ty)
        v2 = tl.load(xb + HALF + offs).to(vo_ptr.dtype.element_ty)
        for r in range(REP):
            yb = vo_ptr + (padded_base + (head * REP + r) * padded_step) * 2 * HALF
            tl.store(yb + offs, v1)
            tl.store(yb + HALF + offs, v2)


@triton.jit
def _qkv_backward(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, b_ptr, t_ptr, dqi_ptr, dki_ptr,
                  dvi_ptr, seq, scb, sct, layout, HEADS: tl.constexpr, KV: tl.constexpr,
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
    packed_base = pid * HEADS
    padded_base = b * HEADS * seq + t
    padded_step = seq
    if layout == 0:
        padded_base = packed_base
        padded_step = 1
    if slot < HEADS:
        yb = dq_ptr + (padded_base + slot * padded_step) * 2 * HALF
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
            yb = dk_ptr + (padded_base + (head * REP + r) * padded_step) * 2 * HALF
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
            yb = dv_ptr + (padded_base + (head * REP + r) * padded_step) * 2 * HALF
            d1 += tl.load(yb + offs).to(tl.float32)
            d2 += tl.load(yb + HALF + offs).to(tl.float32)
        xb = dvi_ptr + pid * (KV * 2 * HALF) + head * 2 * HALF
        tl.store(xb + offs, d1.to(dvi_ptr.dtype.element_ty))
        tl.store(xb + HALF + offs, d2.to(dvi_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["seq", "scb", "sct", "layout", "mode"])
def qkv_kernel(x0, x1, x2, cos_ptr, sin_ptr, b_ptr, t_ptr, y0, y1, y2, seq, scb, sct, layout, mode,
               _f, HEADS: tl.constexpr, KV: tl.constexpr, REP: tl.constexpr, HALF: tl.constexpr):
    """mode 0: forward (x = q, k, v projections, y = rotated heads); mode 1: backward (x = their
    gradients, y = the projections' gradients). One compiled kernel for both."""
    if mode == 0:
        _qkv_forward(x0, x1, x2, cos_ptr, sin_ptr, b_ptr, t_ptr, y0, y1, y2, seq, scb, sct, layout,
                     HEADS, KV, REP, HALF)
    else:
        _qkv_backward(x0, x1, x2, cos_ptr, sin_ptr, b_ptr, t_ptr, y0, y1, y2, seq, scb, sct,
                      layout, HEADS, KV, REP, HALF)


class QKVRotate(torch.autograd.Function):
    """Packed (q, k, v) projections (N, heads * D) -> rotated, fp16, GQA-expanded heads:
    (N, H, D) each when `padded` is False, (B, H, T, D) each when it is True."""

    @staticmethod
    def forward(ctx, q, k, v, cos, sin, b_idx, t_idx, batch, seq, heads, kv_heads, padded):
        dim = q.shape[-1] // heads
        rep, half = heads // kv_heads, dim // 2
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        cos, sin = cos.contiguous(), sin.contiguous()
        scb = 0 if cos.shape[0] == 1 else cos.stride(0)
        tokens = q.shape[0]
        if padded:
            qo = torch.zeros(batch, heads, seq, dim, dtype=torch.float16, device=q.device)
            ko, vo = torch.zeros_like(qo), torch.zeros_like(qo)
        else:
            qo = torch.empty(tokens, heads, dim, dtype=torch.float16, device=q.device)
            ko, vo = torch.empty_like(qo), torch.empty_like(qo)
        qkv_kernel[(tokens, heads + 2 * kv_heads)](
            q, k, v, cos, sin, b_idx, t_idx, qo, ko, vo, seq, scb, cos.stride(1), int(padded), 0,
            0.0, HEADS=heads, KV=kv_heads, REP=rep, HALF=half, num_warps=2)
        ctx.save_for_backward(cos, sin, b_idx, t_idx)
        ctx.meta = (seq, scb, heads, kv_heads, rep, half, q.dtype, int(padded))
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dq, dk, dv):
        cos, sin, b_idx, t_idx = ctx.saved_tensors
        seq, scb, heads, kv_heads, rep, half, dtype, layout = ctx.meta
        dim = 2 * half
        tokens = b_idx.shape[0]
        dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
        dq_in = torch.empty(tokens, heads * dim, dtype=dtype, device=dq.device)
        dk_in = torch.empty(tokens, kv_heads * dim, dtype=dtype, device=dq.device)
        dv_in = torch.empty_like(dk_in)
        qkv_kernel[(tokens, heads + 2 * kv_heads)](
            dq, dk, dv, cos, sin, b_idx, t_idx, dq_in, dk_in, dv_in, seq, scb, cos.stride(1),
            layout, 1, 0.0, HEADS=heads, KV=kv_heads, REP=rep, HALF=half, num_warps=2)
        return dq_in, dk_in, dv_in, None, None, None, None, None, None, None, None, None


def qkv_rotate(q, k, v, cos, sin, b_idx, t_idx, batch, seq, heads, kv_heads, padded=False):
    return QKVRotate.apply(q, k, v, cos, sin, b_idx, t_idx, batch, seq, heads, kv_heads, padded)


_CU_SEQLENS = {}


def _cu_seqlens(lengths, device):
    """int32 [0, l0, l0 + l1, ...] on the GPU, built once per batch shape (28 layers share it)."""
    key = (device, tuple(lengths))
    cu = _CU_SEQLENS.get(key)
    if cu is None:
        if len(_CU_SEQLENS) > 16:
            _CU_SEQLENS.clear()
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + int(length))
        cu = _CU_SEQLENS[key] = torch.tensor(offsets, dtype=torch.int32, device=device)
    return cu


class VarlenAttention(torch.autograd.Function):
    """Causal attention of every sequence of the batch in ONE call of the memory-efficient
    (cutlass) attention kernel that scaled_dot_product_attention itself runs on a T4, given the
    sequence boundaries (`cu_seqlens`) instead of one call per sequence plus a concatenation.
    Inputs (N, H, D) fp16 packed rows (the BSHD layout with B = 1), output (N, H * D)."""

    @staticmethod
    def forward(ctx, query, key, value, cu_seqlens, max_len, scale):
        q, k, v = query[None], key[None], value[None]
        out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
            q, k, v, None, cu_seqlens, cu_seqlens, max_len, max_len, 0.0, 1, True, scale=scale)
        ctx.save_for_backward(q, k, v, out, lse, seed, offset, cu_seqlens)
        ctx.max_len, ctx.scale = max_len, scale
        return out[0].reshape(query.shape[0], -1)

    @staticmethod
    def backward(ctx, grad):
        q, k, v, out, lse, seed, offset, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
            grad.reshape(out.shape).contiguous(), q, k, v, None, out, cu_seqlens, cu_seqlens,
            ctx.max_len, ctx.max_len, lse, 0.0, seed, offset, 1, False, scale=ctx.scale)
        return dq[0], dk[0], dv[0], None, None, None


def varlen_attention(query, key, value, lengths, scale):
    """Causal attention of each sequence on its own, on packed (N, H, D) fp16 rows where
    sequence i is the contiguous block of `lengths[i]` rows; returns (N, H * D)."""
    lengths = [int(length) for length in lengths if int(length) > 0]
    cu_seqlens = _cu_seqlens(lengths, query.device)
    return VarlenAttention.apply(query.contiguous(), key.contiguous(), value.contiguous(),
                                 cu_seqlens, max(lengths), scale)


def _attend(q, k, v, cos, sin, b_idx, t_idx, seq, heads, kv_heads, lengths, scale):
    query, key, value = qkv_rotate(q, k, v, cos, sin, b_idx, t_idx, 0, seq, heads, kv_heads)
    return varlen_attention(query, key, value, lengths, scale)


class PackedAttention(torch.autograd.Function):
    """Rotation + per-sequence causal attention of the packed rows, (N, heads * D) fp16 out.

    Keeps only the three projections (2048 fp16 values per token: pre-rotation, k and v not yet
    repeated for GQA) for the backward, in pinned host memory when `offload`; the backward
    recomputes the rotation and the attention forward (about 1% of the step) and differentiates
    them. Without this, autograd keeps the rotated q, the 6x repeated k and v, the attention
    output and its log-sum-exp on the GPU: 3x the bytes."""

    @staticmethod
    def forward(ctx, q, k, v, cos, sin, b_idx, t_idx, seq, heads, kv_heads, lengths, scale,
                offload):
        with torch.no_grad():
            out = _attend(q, k, v, cos, sin, b_idx, t_idx, seq, heads, kv_heads, lengths, scale)
        ctx.consts = (cos, sin, b_idx, t_idx)  # small, and shared by all layers
        ctx.meta = (seq, heads, kv_heads, lengths, scale)
        if offload:
            ctx.item = OFFLOADER.save(q, k, v)
        else:
            ctx.save_for_backward(q, k, v)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v = OFFLOADER.load(ctx.item) if hasattr(ctx, "item") else ctx.saved_tensors
        inputs = [t.detach().requires_grad_() for t in (q, k, v)]
        with torch.enable_grad():
            out = _attend(*inputs, *ctx.consts, *ctx.meta)
        grads = torch.autograd.grad(out, inputs, dout)
        return (*grads, None, None, None, None, None, None, None, None, None, None)


def packed_attention(q, k, v, cos, sin, b_idx, t_idx, seq, heads, kv_heads, lengths, scale):
    offload = OFFLOADER.wants(q, k, v) and (q.requires_grad or k.requires_grad or v.requires_grad)
    return PackedAttention.apply(q, k, v, cos, sin, b_idx, t_idx, seq, heads, kv_heads, lengths,
                                 scale, offload)


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
    """(B, H, T, D) fp16 attention output -> (N, H * D) rows of the real tokens (fallback path)."""

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
