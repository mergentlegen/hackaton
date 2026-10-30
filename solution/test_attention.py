"""The packed attention kernels against transformers' rotation, casts and repeat_kv."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def reference(q, k, v, cos, sin, heads, kv_heads):
    """transformers: transpose, apply_rotary_pos_emb, the autocast cast, repeat_kv."""
    batch, seq, _ = q.shape
    dim = cos.shape[-1]
    q4 = q.view(batch, seq, heads, dim).transpose(1, 2)
    k4 = k.view(batch, seq, kv_heads, dim).transpose(1, 2)
    v4 = v.view(batch, seq, kv_heads, dim).transpose(1, 2)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    q_emb = q4 * c + rotate_half(q4) * s
    k_emb = k4 * c + rotate_half(k4) * s
    rep = heads // kv_heads

    def repeat_kv(x):
        return x[:, :, None].expand(batch, kv_heads, rep, seq, dim).reshape(batch, heads, seq, dim)

    return q_emb.half(), repeat_kv(k_emb).half(), repeat_kv(v4).half()


def token_maps(lengths, seq):
    real = torch.zeros(len(lengths), seq, dtype=torch.bool, device="cuda")
    for row, length in enumerate(lengths):
        real[row, :length] = True
    rows = torch.nonzero(real.reshape(-1), as_tuple=True)[0]
    return rows, (rows // seq).to(torch.int32), (rows % seq).to(torch.int32)


@pytest.mark.parametrize("lengths,seq", [([64, 37, 50, 12], 64), ([334, 200, 100, 7], 334)])
def test_qkv_rotate_matches_transformers(lengths, seq):
    from solution.attention import qkv_rotate

    torch.manual_seed(0)
    batch, heads, kv_heads, dim = len(lengths), 12, 2, 128
    rows, b_idx, t_idx = token_maps(lengths, seq)
    n = rows.numel()
    q = torch.randn(n, heads * dim, device="cuda", requires_grad=True)
    k = torch.randn(n, kv_heads * dim, device="cuda", requires_grad=True)
    v = torch.randn(n, kv_heads * dim, device="cuda", requires_grad=True)
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    freqs = torch.arange(seq, device="cuda").float()[:, None] * inv_freq[None]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None], emb.sin()[None]  # (1, T, D) like transformers' rotary_emb
    grads = [torch.randn(batch, heads, seq, dim, device="cuda").half() for _ in range(3)]

    def padded(x):
        full = torch.zeros(batch * seq, x.shape[-1], device="cuda")
        return full.index_copy(0, rows, x).view(batch, seq, -1)

    ref = reference(padded(q), padded(k), padded(v), cos.expand(batch, seq, dim),
                    sin.expand(batch, seq, dim), heads, kv_heads)
    torch.autograd.backward(ref, grads)
    ref_grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None

    out = qkv_rotate(q, k, v, cos, sin, b_idx, t_idx, batch, seq, heads, kv_heads)
    torch.autograd.backward(out, grads)
    for mine, theirs in zip(out, ref):
        torch.testing.assert_close(mine.float(), theirs.float(), rtol=2e-3, atol=2e-3)
    for mine, theirs in zip((q.grad, k.grad, v.grad), ref_grads):
        torch.testing.assert_close(mine, theirs, rtol=2e-3, atol=2e-2)


def test_pack_rows_matches_indexing():
    from solution.attention import pack_rows

    torch.manual_seed(0)
    lengths, seq, heads, dim = [50, 20, 33], 64, 12, 128
    rows, b_idx, t_idx = token_maps(lengths, seq)
    attn = torch.randn(len(lengths), heads, seq, dim, device="cuda", dtype=torch.float16,
                       requires_grad=True)
    grad = torch.randn(rows.numel(), heads * dim, device="cuda", dtype=torch.float16)

    ref = attn.permute(0, 2, 1, 3).reshape(-1, heads * dim)[rows]
    ref.backward(grad)
    ref_grad = attn.grad.clone()
    attn.grad = None

    out = pack_rows(attn, b_idx, t_idx)
    out.backward(grad)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(attn.grad, ref_grad, rtol=0, atol=0)
