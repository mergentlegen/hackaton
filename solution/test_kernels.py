"""Correctness of our kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_silu_mul_matches_pytorch(shape):
    from solution.kernels import silu_mul

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=torch.float16)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = silu_mul(gate, up)
    out.backward(grad)
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=2e-3, atol=2e-3)


def reference_loss(logits, labels, num_items):
    """transformers' ForCausalLMLoss, written out."""
    padded = torch.nn.functional.pad(labels, (0, 1), value=-100)
    targets = padded[..., 1:].contiguous().view(-1)
    reduction = "sum" if num_items is not None else "mean"
    loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), targets,
                                             ignore_index=-100, reduction=reduction)
    return loss / num_items if num_items is not None else loss


@pytest.mark.parametrize("shape", [(2, 64, 151936), (3, 37, 1001)])
@pytest.mark.parametrize("num_items", [None, 100])
def test_patched_loss_matches_transformers(shape, num_items):
    from transformers.loss import loss_utils

    from solution.patches import patch_causal_lm_loss

    original = loss_utils.ForCausalLMLoss
    patch_causal_lm_loss()
    try:
        fused = loss_utils.ForCausalLMLoss
        torch.manual_seed(0)
        batch, length, vocab = shape
        logits = (3 * torch.randn(shape, device="cuda")).half().requires_grad_(True)
        labels = torch.randint(0, vocab, (batch, length), device="cuda")
        labels[:, : length // 3] = -100  # the prompt tokens are ignored
        labels[0, -5:] = -100

        ref = reference_loss(logits, labels, num_items)
        ref.backward()
        ref_grad = logits.grad.float()
        logits.grad = None

        out = fused(logits, labels, vocab, num_items_in_batch=num_items)
        out.backward()
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(logits.grad.float(), ref_grad, rtol=2e-3, atol=2e-6)
    finally:
        loss_utils.ForCausalLMLoss = original
        for key, value in list(loss_utils.LOSS_MAPPING.items()):
            if value is fused:
                loss_utils.LOSS_MAPPING[key] = original
