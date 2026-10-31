"""Chunked lm_head + loss against lm_head followed by the fused loss, and torch's argmax."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_linear_cross_entropy_matches_full_logits():
    from solution.kernels import ce_stats, fused_cross_entropy
    from solution.lm_loss import linear_cross_entropy

    torch.manual_seed(0)
    rows, hidden_size, vocab = 600, 256, 20011  # more rows than one chunk, odd vocab
    hidden = (torch.randn(rows, hidden_size, device="cuda")).half()
    weight = (torch.randn(vocab, hidden_size, device="cuda") * 0.05).half()
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    targets[::7] = -100
    denom = (targets != -100).sum().float()
    grad_output = torch.tensor(1234.5, device="cuda")

    h_ref = hidden.clone().requires_grad_()
    logits = torch.nn.functional.linear(h_ref, weight)
    ref_loss = fused_cross_entropy(logits, targets, denom)
    (ref_loss * grad_output).backward()
    full = logits.detach()
    no_target = torch.full((rows,), -1, dtype=torch.int64, device="cuda")
    _, ref_losses, _ = ce_stats(full, targets, False)
    _, _, ref_entropy = ce_stats(full, no_target, True)

    h = hidden.clone().requires_grad_()
    loss, losses, entropy, argmax = linear_cross_entropy(h, weight, targets, denom)
    (loss * grad_output).backward()

    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(losses, ref_losses, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-4, atol=1e-4)
    assert torch.equal(argmax, full.argmax(dim=-1))
    cos = torch.nn.functional.cosine_similarity(h.grad.float().flatten(),
                                                h_ref.grad.float().flatten(), dim=0)
    assert cos > 0.9999
    torch.testing.assert_close(h.grad.float(), h_ref.grad.float(), rtol=2e-2, atol=2e-3)


def test_argmax_takes_the_first_maximum_like_torch():
    from solution.kernels import ce_stats

    logits = torch.zeros(3, 9000, device="cuda").half()
    logits[0, 17] = logits[0, 8000] = 5.0
    logits[1, 4100] = logits[1, 4099] = 2.0  # neighbours in different kernel lanes
    targets = torch.zeros(3, dtype=torch.int64, device="cuda")
    out = (torch.empty(3, device="cuda"), torch.empty(3, device="cuda"),
           torch.empty(3, device="cuda"), torch.empty(3, dtype=torch.int64, device="cuda"))
    ce_stats(logits, targets, True, out=out)
    assert out[3].tolist() == logits.argmax(dim=-1).tolist() == [17, 4099, 0]
