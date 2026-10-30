"""Offloaded activations come back bit for bit, and their GPU memory is released meanwhile."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_items_round_trip_in_backward_order():
    from solution.offload import KEEP, Offloader

    off = Offloader()
    torch.manual_seed(0)
    originals = [(torch.randn(300, 700, device="cuda").half(), torch.randn(300, 5, device="cuda"))
                 for _ in range(12)]
    off.begin(torch.device("cuda"))
    before = torch.cuda.memory_allocated()
    items = [off.save(*(t.clone() for t in pair)) for pair in originals]
    off.end()
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - before
    per_item = sum(t.numel() * t.element_size() for t in originals[0])
    assert held <= (KEEP + 1) * per_item * 1.1  # only the last KEEP items stay on the GPU
    for item, pair in zip(reversed(items), reversed(originals)):
        back = off.load(item)
        for mine, theirs in zip(back, pair):
            assert mine.dtype == theirs.dtype and mine.shape == theirs.shape
            assert torch.equal(mine, theirs)


def test_silu_mul_and_rmsnorm_offloaded_gradients_match():
    from solution.kernels import silu_mul
    from solution.offload import OFFLOADER
    from solution.rmsnorm import rms_norm

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
