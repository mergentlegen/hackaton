"""Our own Triton kernels and the autograd functions around them.

Everything here works on *packed* tokens: the pad tokens of a batch are removed, so every
tensor is [N, features] with N = number of real tokens. Only attention goes back to the padded
[B, T, heads, D] layout (see `attn_prep` / `gather_heads`).

Kernels (all @triton.jit, all written for this project):
  _add_rmsnorm_fwd / _add_rmsnorm_bwd   residual add + RMSNorm, fp32 residual stream -> fp16
  _attn_prep_fwd / _attn_prep_bwd       LoRA add + RoPE + GQA expansion + scatter to padded layout
  _gather_heads / _scatter_heads        attention output <-> packed layout
  _swiglu_fwd / _swiglu_bwd             silu(gate) * up, backward in place
  _dropout_kernel                       LoRA input dropout from a seed (mask is never stored)
  _ce_kernel                            cross entropy + entropy + argmax + dlogits over the vocabulary

The numerics follow plain Soup on a T4: matmuls in fp16, residual stream and norms in fp32,
rounding to fp16 at the same places where autocast rounds.
"""

import torch
import triton
import triton.language as tl

CDT = torch.float16  # compute dtype of the matmuls (a T4 has no bf16); tests may change it
GSCALE = 1024.0  # dlogits are stored multiplied by this so small probabilities survive fp16
CE_BLOCK = 2048  # vocabulary tile of the cross entropy kernel (tests on the CPU use a bigger one)


def _pow2(n):
    return triton.next_power_of_2(int(n))


# --------------------------------------------------------------------------------------------
# residual add + RMSNorm
# --------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n_rows"])
def _add_rmsnorm_fwd(x_ptr, d_ptr, w_ptr, s_ptr, y_ptr, rstd_ptr, n_rows, H, eps,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    off = row * H + cols
    s = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32) + \
        tl.load(d_ptr + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(s_ptr + off, s, mask=mask)
    inv = tl.rsqrt(tl.sum(s * s, axis=0) / H + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + off, ((s * inv) * w).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, inv)


@triton.jit(do_not_specialize=["n_rows"])
def _add_rmsnorm_bwd(s_ptr, w_ptr, rstd_ptr, dy_ptr, dres_ptr, ds_ptr, dd_ptr, n_rows, H,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    off = row * H + cols
    inv = tl.load(rstd_ptr + row)
    xhat = tl.load(s_ptr + off, mask=mask, other=0.0) * inv
    g = tl.load(dy_ptr + off, mask=mask, other=0.0).to(tl.float32) * \
        tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean_gx = tl.sum(g * xhat, axis=0) / H
    ds = tl.load(dres_ptr + off, mask=mask, other=0.0) + inv * (g - xhat * mean_gx)
    tl.store(ds_ptr + off, ds, mask=mask)
    tl.store(dd_ptr + off, ds.to(dd_ptr.dtype.element_ty), mask=mask)


class AddRMSNorm(torch.autograd.Function):
    """s = x + delta (fp32 residual stream); y = rmsnorm(s) * weight (fp16). Returns (s, y).

    The backward adds the gradient that reaches `s` through the residual connection, so the
    residual-gradient add is fused into the norm backward as well. The norm weight is frozen.
    """

    @staticmethod
    def forward(ctx, x, delta, weight, eps):
        n, h = x.shape
        s = torch.empty_like(x)
        y = torch.empty(n, h, dtype=CDT, device=x.device)
        rstd = torch.empty(n, dtype=torch.float32, device=x.device)
        _add_rmsnorm_fwd[(n,)](x, delta, weight, s, y, rstd, n, h, eps, BLOCK=_pow2(h))
        ctx.save_for_backward(s, weight, rstd)
        return s, y

    @staticmethod
    def backward(ctx, grad_s, grad_y):
        s, weight, rstd = ctx.saved_tensors
        n, h = s.shape
        if grad_s is None:
            grad_s = torch.zeros_like(s)
        ds = torch.empty_like(s)
        dd = torch.empty(n, h, dtype=CDT, device=s.device)
        _add_rmsnorm_bwd[(n,)](s, weight, rstd, grad_y.contiguous(), grad_s.contiguous(), ds, dd,
                               n, h, BLOCK=_pow2(h))
        return ds, dd, None, None


# --------------------------------------------------------------------------------------------
# attention layout: LoRA add + RoPE + GQA expansion (forward), and the reverse (backward)
# --------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n_tok", "sb", "sh", "st"])
def _attn_prep_fwd(qb_ptr, ql_ptr, kb_ptr, vb_ptr, vl_ptr, cos_ptr, sin_ptr, tb_ptr, tt_ptr,
                   qp_ptr, kp_ptr, vp_ptr, n_tok, sb, sh, st,
                   NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    REP: tl.constexpr = NQ // NKV
    offs = tl.arange(0, HALF)
    b = tl.load(tb_ptr + n).to(tl.int64)
    t = tl.load(tt_ptr + n).to(tl.int64)
    c1 = tl.load(cos_ptr + t * D + offs)
    c2 = tl.load(cos_ptr + t * D + HALF + offs)
    s1 = tl.load(sin_ptr + t * D + offs)
    s2 = tl.load(sin_ptr + t * D + HALF + offs)
    if head < NQ:
        src = n * (NQ * D) + head * D
        x1 = tl.load(qb_ptr + src + offs).to(tl.float32) + tl.load(ql_ptr + src + offs).to(tl.float32)
        x2 = tl.load(qb_ptr + src + HALF + offs).to(tl.float32) + \
            tl.load(ql_ptr + src + HALF + offs).to(tl.float32)
        dst = b * sb + head * sh + t * st
        tl.store(qp_ptr + dst + offs, (x1 * c1 - x2 * s1).to(qp_ptr.dtype.element_ty))
        tl.store(qp_ptr + dst + HALF + offs, (x2 * c2 + x1 * s2).to(qp_ptr.dtype.element_ty))
    else:
        kh = head - NQ
        src = n * (NKV * D) + kh * D
        k1 = tl.load(kb_ptr + src + offs).to(tl.float32)
        k2 = tl.load(kb_ptr + src + HALF + offs).to(tl.float32)
        v1 = tl.load(vb_ptr + src + offs).to(tl.float32) + tl.load(vl_ptr + src + offs).to(tl.float32)
        v2 = tl.load(vb_ptr + src + HALF + offs).to(tl.float32) + \
            tl.load(vl_ptr + src + HALF + offs).to(tl.float32)
        ko1 = (k1 * c1 - k2 * s1).to(kp_ptr.dtype.element_ty)
        ko2 = (k2 * c2 + k1 * s2).to(kp_ptr.dtype.element_ty)
        for r in tl.static_range(REP):
            dst = b * sb + (kh * REP + r) * sh + t * st
            tl.store(kp_ptr + dst + offs, ko1)
            tl.store(kp_ptr + dst + HALF + offs, ko2)
            tl.store(vp_ptr + dst + offs, v1.to(vp_ptr.dtype.element_ty))
            tl.store(vp_ptr + dst + HALF + offs, v2.to(vp_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["n_tok", "sb", "sh", "st", "gsb", "gsh", "gst"])
def _attn_prep_bwd(dqp_ptr, dkp_ptr, dvp_ptr, cos_ptr, sin_ptr, tb_ptr, tt_ptr,
                   dq_ptr, dk_ptr, dv_ptr, n_tok, gsb, gsh, gst,
                   NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    REP: tl.constexpr = NQ // NKV
    offs = tl.arange(0, HALF)
    b = tl.load(tb_ptr + n).to(tl.int64)
    t = tl.load(tt_ptr + n).to(tl.int64)
    c1 = tl.load(cos_ptr + t * D + offs)
    c2 = tl.load(cos_ptr + t * D + HALF + offs)
    s1 = tl.load(sin_ptr + t * D + offs)
    s2 = tl.load(sin_ptr + t * D + HALF + offs)
    if head < NQ:
        src = b * gsb + head * gsh + t * gst
        g1 = tl.load(dqp_ptr + src + offs).to(tl.float32)
        g2 = tl.load(dqp_ptr + src + HALF + offs).to(tl.float32)
        dst = n * (NQ * D) + head * D
        tl.store(dq_ptr + dst + offs, (g1 * c1 + g2 * s2).to(dq_ptr.dtype.element_ty))
        tl.store(dq_ptr + dst + HALF + offs, (g2 * c2 - g1 * s1).to(dq_ptr.dtype.element_ty))
    else:
        kh = head - NQ
        gk1 = tl.zeros([HALF], dtype=tl.float32)
        gk2 = tl.zeros([HALF], dtype=tl.float32)
        gv1 = tl.zeros([HALF], dtype=tl.float32)
        gv2 = tl.zeros([HALF], dtype=tl.float32)
        for r in tl.static_range(REP):
            src = b * gsb + (kh * REP + r) * gsh + t * gst
            gk1 += tl.load(dkp_ptr + src + offs).to(tl.float32)
            gk2 += tl.load(dkp_ptr + src + HALF + offs).to(tl.float32)
            gv1 += tl.load(dvp_ptr + src + offs).to(tl.float32)
            gv2 += tl.load(dvp_ptr + src + HALF + offs).to(tl.float32)
        dst = n * (NKV * D) + kh * D
        tl.store(dk_ptr + dst + offs, (gk1 * c1 + gk2 * s2).to(dk_ptr.dtype.element_ty))
        tl.store(dk_ptr + dst + HALF + offs, (gk2 * c2 - gk1 * s1).to(dk_ptr.dtype.element_ty))
        tl.store(dv_ptr + dst + offs, gv1.to(dv_ptr.dtype.element_ty))
        tl.store(dv_ptr + dst + HALF + offs, gv2.to(dv_ptr.dtype.element_ty))


class Geometry:
    """Where every packed token lives in the padded [B, T, heads, D] layout."""

    def __init__(self, tok_b, tok_t, batch, seq, cos, sin, n_q, n_kv, head_dim):
        self.tok_b, self.tok_t = tok_b, tok_t
        self.batch, self.seq = batch, seq
        self.cos, self.sin = cos, sin
        self.n_q, self.n_kv, self.d = n_q, n_kv, head_dim
        self.n = tok_b.numel()


def _padded(geo, dtype, device, zeros):
    shape = (geo.batch, geo.seq, geo.n_q, geo.d)
    return torch.zeros(shape, dtype=dtype, device=device) if zeros else \
        torch.empty(shape, dtype=dtype, device=device)


def _btHd(x):
    """x is [B, heads, T, D]; return it as a view of a contiguous [B, T, heads, D] tensor."""
    y = x.transpose(1, 2)
    return x if y.is_contiguous() else y.contiguous().transpose(1, 2)


class AttnPrep(torch.autograd.Function):
    """(q, q_lora, k, v, v_lora) packed -> padded, rotated, GQA-expanded q, k, v.

    The padded tensors are stored as [B, T, heads, D] and handed out as [B, heads, T, D] views,
    which is the layout the memory-efficient SDPA kernel uses anyway (no hidden copy).
    """

    @staticmethod
    def forward(ctx, qb, ql, kb, vb, vl, geo):
        qp = _padded(geo, qb.dtype, qb.device, True)
        kp = _padded(geo, qb.dtype, qb.device, True)
        vp = _padded(geo, qb.dtype, qb.device, True)
        sb, st, sh = qp.stride(0), qp.stride(1), qp.stride(2)
        _attn_prep_fwd[(geo.n, geo.n_q + geo.n_kv)](
            qb, ql, kb, vb, vl, geo.cos, geo.sin, geo.tok_b, geo.tok_t, qp, kp, vp,
            geo.n, sb, sh, st, NQ=geo.n_q, NKV=geo.n_kv, D=geo.d)
        ctx.geo = geo
        return qp.transpose(1, 2), kp.transpose(1, 2), vp.transpose(1, 2)

    @staticmethod
    def backward(ctx, dqp, dkp, dvp):
        geo = ctx.geo
        n = geo.n
        # the three gradients must share one [B, T, heads, D] memory layout
        dqp, dkp, dvp = _btHd(dqp), _btHd(dkp), _btHd(dvp)
        dq = torch.empty(n, geo.n_q * geo.d, dtype=dqp.dtype, device=dqp.device)
        dk = torch.empty(n, geo.n_kv * geo.d, dtype=dqp.dtype, device=dqp.device)
        dv = torch.empty_like(dk)
        # gradients arrive as [B, heads, T, D] views with arbitrary strides
        _attn_prep_bwd[(n, geo.n_q + geo.n_kv)](
            dqp, dkp, dvp, geo.cos, geo.sin, geo.tok_b, geo.tok_t, dq, dk, dv, n,
            dqp.stride(0), dqp.stride(1), dqp.stride(2), NQ=geo.n_q, NKV=geo.n_kv, D=geo.d)
        return dq, dq, dk, dv, dv, None


@triton.jit(do_not_specialize=["n_tok", "sb", "sh", "st"])
def _gather_heads(o_ptr, tb_ptr, tt_ptr, out_ptr, n_tok, sb, sh, st,
                  NQ: tl.constexpr, D: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs = tl.arange(0, D)
    b = tl.load(tb_ptr + n).to(tl.int64)
    t = tl.load(tt_ptr + n).to(tl.int64)
    x = tl.load(o_ptr + b * sb + head * sh + t * st + offs)
    tl.store(out_ptr + n * (NQ * D) + head * D + offs, x)


@triton.jit(do_not_specialize=["n_tok", "sb", "sh", "st"])
def _scatter_heads(g_ptr, tb_ptr, tt_ptr, out_ptr, n_tok, sb, sh, st,
                   NQ: tl.constexpr, D: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs = tl.arange(0, D)
    b = tl.load(tb_ptr + n).to(tl.int64)
    t = tl.load(tt_ptr + n).to(tl.int64)
    x = tl.load(g_ptr + n * (NQ * D) + head * D + offs)
    tl.store(out_ptr + b * sb + head * sh + t * st + offs, x)


class GatherHeads(torch.autograd.Function):
    """SDPA output [B, heads, T, D] (any strides) -> packed [N, heads * D]."""

    @staticmethod
    def forward(ctx, o, geo):
        out = torch.empty(geo.n, geo.n_q * geo.d, dtype=o.dtype, device=o.device)
        _gather_heads[(geo.n, geo.n_q)](o, geo.tok_b, geo.tok_t, out, geo.n, o.stride(0),
                                        o.stride(1), o.stride(2), NQ=geo.n_q, D=geo.d)
        ctx.geo = geo
        return out

    @staticmethod
    def backward(ctx, g):
        geo = ctx.geo
        full = _padded(geo, g.dtype, g.device, True)
        _scatter_heads[(geo.n, geo.n_q)](g.contiguous(), geo.tok_b, geo.tok_t, full, geo.n,
                                         full.stride(0), full.stride(2), full.stride(1),
                                         NQ=geo.n_q, D=geo.d)
        return full.transpose(1, 2), None


# --------------------------------------------------------------------------------------------
# SwiGLU
# --------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n_rows"])
def _swiglu_fwd(g_ptr, u_ptr, o_ptr, n_rows, I, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    at = row * I + cols
    gate = tl.load(g_ptr + at, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(u_ptr + at, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + at, (gate / (1.0 + tl.exp(-gate)) * up).to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["n_rows"])
def _swiglu_bwd(da_ptr, g_ptr, u_ptr, n_rows, I, BLOCK: tl.constexpr):
    """Overwrites gate with d(gate) and up with d(up): the saved tensors die here anyway."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    at = row * I + cols
    da = tl.load(da_ptr + at, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(g_ptr + at, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(u_ptr + at, mask=mask, other=0.0).to(tl.float32)
    sg = 1.0 / (1.0 + tl.exp(-gate))
    tl.store(g_ptr + at, (da * up * sg * (1.0 + gate * (1.0 - sg))).to(g_ptr.dtype.element_ty),
             mask=mask)
    tl.store(u_ptr + at, (da * gate * sg).to(u_ptr.dtype.element_ty), mask=mask)


class SwiGLU(torch.autograd.Function):
    """silu(gate) * up on fp16 [N, I]; saves only gate and up (no activation output)."""

    @staticmethod
    def forward(ctx, gate, up):
        n, i = gate.shape
        out = torch.empty_like(gate)
        block = 1024
        _swiglu_fwd[(n, triton.cdiv(i, block))](gate, up, out, n, i, BLOCK=block)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        n, i = gate.shape
        block = 1024
        _swiglu_bwd[(n, triton.cdiv(i, block))](grad.contiguous(), gate, up, n, i, BLOCK=block)
        return gate, up


# --------------------------------------------------------------------------------------------
# LoRA: dropout from a seed + two skinny matmuls
# --------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n_elem", "seed"])
def _dropout_kernel(x_ptr, out_ptr, n_elem, seed, p, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    keep = tl.rand(seed, offs.to(tl.int32)) >= p
    tl.store(out_ptr + offs, tl.where(keep, x / (1.0 - p), 0.0).to(out_ptr.dtype.element_ty),
             mask=mask)


def dropout_from_seed(x, p, seed):
    """Same mask for the same seed, so the backward rebuilds it instead of storing it."""
    if p <= 0.0:
        return x
    x = x.contiguous()
    out = torch.empty_like(x)
    block = 1024
    n = x.numel()
    _dropout_kernel[(triton.cdiv(n, block),)](x, out, n, seed, p, BLOCK=block)
    return out


class LoraDelta(torch.autograd.Function):
    """scaling * B(A(dropout(x))) for one LoRA adapter, saving only x, the rank-r activation
    and a seed. The dropout mask is rebuilt in the backward from the seed."""

    @staticmethod
    def forward(ctx, x, a_w, b_w, scaling, p, seed):
        a16, b16 = a_w.to(x.dtype), b_w.to(x.dtype)
        xd = dropout_from_seed(x, p, seed)
        t = torch.mm(xd, a16.t())
        out = torch.mm(t, b16.t()) * scaling
        ctx.save_for_backward(x, t, a_w, b_w)
        ctx.cfg = (scaling, p, seed)
        return out

    @staticmethod
    def backward(ctx, g):
        x, t, a_w, b_w = ctx.saved_tensors
        scaling, p, seed = ctx.cfg
        a16, b16 = a_w.to(x.dtype), b_w.to(x.dtype)
        gs = (g * scaling).contiguous()
        db = torch.mm(gs.t(), t)
        dt = torch.mm(gs, b16)
        xd = dropout_from_seed(x, p, seed)
        da = torch.mm(dt.t(), xd)
        dx = None
        if ctx.needs_input_grad[0]:
            dx = dropout_from_seed(torch.mm(dt, a16), p, seed)
        return dx, da.to(a_w.dtype), db.to(b_w.dtype), None, None, None


# --------------------------------------------------------------------------------------------
# cross entropy over the vocabulary, chunked, with entropy / argmax / dlogits in one pass
# --------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["n_rows"])
def _ce_kernel(logits_ptr, tgt_ptr, nll_ptr, ent_ptr, pred_ptr, n_rows, stride, gscale,
               V: tl.constexpr, WRITE_GRAD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride
    tgt = tl.load(tgt_ptr + row).to(tl.int64)
    m = tl.full([], float("-inf"), tl.float32)
    s = tl.zeros([], tl.float32)
    tx = tl.zeros([], tl.float32)
    best = tl.full([], float("-inf"), tl.float32)
    best_i = tl.zeros([], tl.int64)
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(base + cols, mask=mask, other=float("-inf")).to(tl.float32)
        tile_max = tl.max(x, axis=0)
        tile_arg = tl.argmax(x, axis=0).to(tl.int64) + start
        better = tile_max > best
        best = tl.where(better, tile_max, best)
        best_i = tl.where(better, tile_arg, best_i)
        new_m = tl.maximum(m, tile_max)
        alpha = tl.exp(m - new_m)
        e = tl.exp(x - new_m)  # exactly 0 where the column is padding (x = -inf)
        s = s * alpha + tl.sum(e, axis=0)
        tx = tx * alpha + tl.sum(e * tl.where(mask, x, 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    xt = tl.load(base + tgt, mask=tgt >= 0, other=0.0).to(tl.float32)
    tl.store(nll_ptr + row, tl.where(tgt >= 0, lse - xt, 0.0))
    tl.store(ent_ptr + row, lse - tx / s)
    tl.store(pred_ptr + row, best_i.to(tl.int32))
    if WRITE_GRAD:
        for start in range(0, V, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < V
            x = tl.load(base + cols, mask=mask, other=float("-inf")).to(tl.float32)
            g = tl.exp(x - lse) - tl.where(cols == tgt, 1.0, 0.0)
            g = tl.where(tgt >= 0, g * gscale, 0.0)
            tl.store(base + cols, g.to(logits_ptr.dtype.element_ty), mask=mask)


class FusedLMHeadLoss(torch.autograd.Function):
    """sum of token cross entropy of `hidden @ weight.T` without ever holding all the logits.

    The vocabulary projection runs in chunks of rows. For each chunk one kernel reads the logits
    and writes the loss terms, the entropy, the argmax and (in place) d(sum loss)/d(logits).
    The gradient w.r.t. the hidden states is then a single matmul per chunk and is kept
    (N x H, small), so the backward is a rescale. `stats` receives per-token nll, entropy, argmax.
    """

    @staticmethod
    def forward(ctx, hidden, weight, tgt, stats, chunk):
        n, h = hidden.shape
        need_grad = ctx.needs_input_grad[0]
        nll = torch.empty(n, dtype=torch.float32, device=hidden.device)
        ent = torch.empty_like(nll)
        pred = torch.empty(n, dtype=torch.int32, device=hidden.device)
        dh = torch.empty(n, h, dtype=hidden.dtype, device=hidden.device) if need_grad else None
        v = weight.shape[0]
        buf = torch.empty(min(chunk, n), v, dtype=hidden.dtype, device=hidden.device)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            logits = buf[: end - start]
            torch.mm(hidden[start:end], weight.t(), out=logits)  # one reused logits buffer
            _ce_kernel[(end - start,)](logits, tgt[start:end], nll[start:end], ent[start:end],
                                       pred[start:end], end - start, logits.stride(0), GSCALE,
                                       V=v, WRITE_GRAD=need_grad, BLOCK=CE_BLOCK)
            if need_grad:
                torch.mm(logits, weight, out=dh[start:end])
        del buf
        stats["nll"], stats["entropy"], stats["pred"] = nll, ent, pred
        ctx.save_for_backward(dh)
        return nll.sum()

    @staticmethod
    def backward(ctx, grad_out):
        (dh,) = ctx.saved_tensors
        if dh is None:
            return None, None, None, None, None
        scale = grad_out.to(torch.float32) / GSCALE
        return (dh.float() * scale).to(dh.dtype), None, None, None, None
