"""The Q/K/V kernels against transformers' rotation, casts and repeat_kv."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def reference(q, k, v, cos, sin, heads, kv_heads):
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


@pytest.mark.parametrize("batch,seq", [(4, 334), (2, 37)])
def test_qkv_rotate_matches_transformers(batch, seq):
    from solution.attention import qkv_rotate

    torch.manual_seed(0)
    heads, kv_heads, dim = 12, 2, 128
    q = torch.randn(batch, seq, heads * dim, device="cuda", requires_grad=True)
    k = torch.randn(batch, seq, kv_heads * dim, device="cuda", requires_grad=True)
    v = torch.randn(batch, seq, kv_heads * dim, device="cuda", requires_grad=True)
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    freqs = torch.arange(seq, device="cuda").float()[:, None] * inv_freq[None]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None].expand(batch, seq, dim).contiguous()
    sin = emb.sin()[None].expand(batch, seq, dim).contiguous()
    grads = [torch.randn(batch, heads, seq, dim, device="cuda").half() for _ in range(3)]

    ref = reference(q, k, v, cos, sin, heads, kv_heads)
    torch.autograd.backward(ref, grads)
    ref_grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None

    out = qkv_rotate(q, k, v, cos, sin, heads, kv_heads)
    torch.autograd.backward(out, grads)
    for mine, theirs in zip(out, ref):
        torch.testing.assert_close(mine.float(), theirs.float(), rtol=2e-3, atol=2e-3)
    for mine, theirs in zip((q.grad, k.grad, v.grad), ref_grads):
        torch.testing.assert_close(mine, theirs, rtol=2e-3, atol=2e-2)
