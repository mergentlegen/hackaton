"""Offloaded activations come back bit for bit, and their GPU memory is released meanwhile."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_items_round_trip_in_backward_order(monkeypatch):
    from solution import offload
    from solution.offload import Offloader

    off = Offloader()
    torch.manual_seed(0)
    originals = [(torch.randn(300, 700, device="cuda").half(), torch.randn(300, 5, device="cuda"))
                 for _ in range(12)]
    per_item = sum(t.numel() * t.element_size() for t in originals[0])
    monkeypatch.setattr(offload, "KEEP_BYTES", 3 * per_item)
    monkeypatch.setattr(offload, "PREFETCH_BYTES", 2 * per_item)
    off.begin(torch.device("cuda"))
    before = torch.cuda.memory_allocated()
    items = [off.save(*(t.clone() for t in pair)) for pair in originals]
    off.end()
    torch.cuda.synchronize()
    torch.empty(1, device="cuda")  # lets the allocator see the finished copies
    held = torch.cuda.memory_allocated() - before
    assert held <= 4 * per_item * 1.1  # only the newest KEEP_BYTES stay on the GPU
    for item, pair in zip(reversed(items), reversed(originals)):
        back = off.load(item)
        for mine, theirs in zip(back, pair):
            assert mine.dtype == theirs.dtype and mine.shape == theirs.shape
            assert torch.equal(mine, theirs)


def test_silu_mul_and_rmsnorm_offloaded_gradients_match(monkeypatch):
    from solution import offload
    from solution.kernels import silu_mul
    from solution.offload import OFFLOADER
    from solution.rmsnorm import rms_norm

    monkeypatch.setattr(offload, "KEEP_BYTES", 1 << 20)
    monkeypatch.setattr(offload, "PREFETCH_BYTES", 1 << 20)
    torch.manual_seed(0)
    x = torch.randn(257, 384, device="cuda", requires_grad=True)
    w = torch.rand(384, device="cuda") + 0.5
    wg = torch.randn(384, 1000, device="cuda").half() * 0.05
    wu = torch.randn(384, 1000, device="cuda").half() * 0.05

    def run(offload):
        x.grad = None
        if offload:
            OFFLOADER.begin(x.device)
        h = x
        for _ in range(4):  # a few "layers" so that some items really leave the GPU
            n = rms_norm(h, w, 1e-6, to_half=True)
            h = h + silu_mul(n @ wg, n @ wu).float() @ wg.float().t()
        if offload:
            OFFLOADER.end()
        h.square().mean().backward()
        return h.detach(), x.grad.clone()

    out_ref, grad_ref = run(False)
    out, grad = run(True)
    assert torch.equal(out, out_ref)
    assert torch.equal(grad, grad_ref)


def test_packed_attention_offloaded_matches_plain_autograd(monkeypatch):
    from solution import offload
    from solution.attention import packed_attention, qkv_rotate, varlen_attention
    from solution.offload import OFFLOADER

    monkeypatch.setattr(offload, "KEEP_BYTES", 1 << 20)
    monkeypatch.setattr(offload, "PREFETCH_BYTES", 1 << 20)
    torch.manual_seed(0)
    lengths, heads, kv_heads, dim = [70, 33, 51], 12, 2, 128
    seq = max(lengths)
    real = torch.zeros(len(lengths), seq, dtype=torch.bool, device="cuda")
    for row, length in enumerate(lengths):
        real[row, :length] = True
    rows = torch.nonzero(real.reshape(-1), as_tuple=True)[0]
    b_idx, t_idx = (rows // seq).to(torch.int32), (rows % seq).to(torch.int32)
    n = rows.numel()
    emb = torch.arange(seq, device="cuda").float()[:, None] * torch.rand(dim // 2, device="cuda")
    emb = torch.cat((emb, emb), dim=-1)[None]
    cos, sin = emb.cos(), emb.sin()
    q = (torch.randn(n, heads * dim, device="cuda") * 0.3).half().requires_grad_()
    k = (torch.randn(n, kv_heads * dim, device="cuda") * 0.3).half().requires_grad_()
    v = (torch.randn(n, kv_heads * dim, device="cuda") * 0.3).half().requires_grad_()
    grad = torch.randn(n, heads * dim, device="cuda").half()

    query, key, value = qkv_rotate(q, k, v, cos, sin, b_idx, t_idx, 0, seq, heads, kv_heads)
    ref = varlen_attention(query, key, value, lengths, dim ** -0.5)
    ref.backward(grad)
    ref_grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None

    OFFLOADER.begin(q.device)
    outs = [packed_attention(q, k, v, cos, sin, b_idx, t_idx, seq, heads, kv_heads, lengths,
                             dim ** -0.5) for _ in range(3)]  # 3 "layers": some leave the GPU
    OFFLOADER.end()
    torch.autograd.backward(outs, [grad] * 3)
    assert torch.equal(outs[0].detach(), ref.detach())
    for t, theirs in zip((q, k, v), ref_grads):
        torch.testing.assert_close(t.grad.float(), 3 * theirs.float(), rtol=1e-2, atol=1e-2)
