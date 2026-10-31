"""Correctness of the RMSNorm kernel and of the logit statistics against plain PyTorch."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("shape", [(4, 334, 1536), (3, 77, 1000)])
@pytest.mark.parametrize("train_weight", [False, True])
@pytest.mark.parametrize("to_half", [False, True])
def test_rms_norm_matches_qwen2(shape, train_weight, to_half):
    from solution.rmsnorm import rms_norm

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", requires_grad=True)
    weight = (1 + 0.1 * torch.randn(shape[-1], device="cuda")).requires_grad_(train_weight)
    dy = torch.randn(shape, device="cuda")
    eps = 1e-6
    out_dtype = torch.float16 if to_half else torch.float32

    variance = x.pow(2).mean(-1, keepdim=True)
    ref = (weight * (x * torch.rsqrt(variance + eps))).to(out_dtype)
    ref.backward(dy.to(out_dtype))
    ref_dx = x.grad.clone()
    ref_dw = weight.grad.clone() if train_weight else None
    x.grad = None
    if train_weight:
        weight.grad = None

    out = rms_norm(x, weight, eps, to_half)
    assert out.dtype == out_dtype
    out.backward(dy.to(out_dtype))
    tol = 2e-3 if to_half else 1e-5
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, ref_dx, rtol=1e-3 if to_half else 1e-4, atol=1e-3 if to_half else 1e-5)
    if train_weight:
        torch.testing.assert_close(weight.grad, ref_dw, rtol=2e-3 if to_half else 1e-4,
                                   atol=3e-2 if to_half else 1e-3)


def test_entropy_matches_trl():
    from solution.logit_stats import entropy_from_logits

    torch.manual_seed(0)
    logits = (3 * torch.randn(3, 41, 151936, device="cuda")).half()
    logps = torch.log_softmax(logits.float(), dim=-1)
    ref = -(logps.exp() * logps).sum(-1)
    torch.testing.assert_close(entropy_from_logits(logits), ref, rtol=1e-4, atol=1e-4)


def test_row_losses_match_soup():
    from solution.logit_stats import row_cross_entropy

    torch.manual_seed(0)
    batch, length, vocab = 3, 50, 151936
    logits = (3 * torch.randn(batch, length, vocab, device="cuda")).half()
    labels = torch.randint(0, vocab, (batch, length), device="cuda")
    labels[:, : length // 3] = -100
    labels[2, :] = -100  # a row without supervised tokens
    losses, counts = row_cross_entropy(logits, labels)

    for i in range(batch):
        targets = labels[i, 1:]
        mask = targets != -100
        if mask.sum() == 0:
            assert counts[i] == 0 and losses[i] == 0.0
            continue
        token = torch.nn.functional.cross_entropy(logits[i, :-1].float(), targets,
                                                  reduction="none", ignore_index=-100)
        assert counts[i] == int(mask.sum())
        assert abs(losses[i] - float(token[mask].mean())) < 1e-4 * max(1.0, losses[i])


@pytest.mark.parametrize("to_half", [True, False])
def test_add_rmsnorm_matches_add_then_rmsnorm(to_half):
    from solution.rmsnorm import add_rms_norm, rms_norm

    torch.manual_seed(0)
    residual = torch.randn(300, 1536, device="cuda", requires_grad=True)
    delta = torch.randn(300, 1536, device="cuda").half().requires_grad_()
    weight = torch.rand(1536, device="cuda") + 0.5
    g_total = torch.randn(300, 1536, device="cuda")
    g_out = torch.randn(300, 1536, device="cuda")

    total = residual + delta
    out = rms_norm(total, weight, 1e-6, to_half=to_half)
    torch.autograd.backward([total, out], [g_total, g_out.to(out.dtype)])
    ref = [total.detach(), out.detach(), residual.grad.clone(), delta.grad.clone()]
    residual.grad = delta.grad = None

    total, out = add_rms_norm(residual, delta, weight, 1e-6, to_half=to_half)
    torch.autograd.backward([total, out], [g_total, g_out.to(out.dtype)])
    mine = [total.detach(), out.detach(), residual.grad, delta.grad]
    assert torch.equal(mine[0], ref[0])  # the sum is the same fp32 addition
    for a, b in zip(mine[1:], ref[1:]):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a.float(), b.float(), rtol=1e-3, atol=1e-3)
