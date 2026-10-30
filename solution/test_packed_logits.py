"""PackedLogits answers TRL's and Soup's metric questions like the padded tensor would."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def make(lengths=(50, 20, 33), seq=64, vocab=3001):
    from solution.packed_logits import PackedLogits

    torch.manual_seed(0)
    real = torch.zeros(len(lengths), seq, dtype=torch.bool, device="cuda")
    for row, length in enumerate(lengths):
        real[row, :length] = True
    rows = torch.nonzero(real.reshape(-1), as_tuple=True)[0]
    packed = (3 * torch.randn(rows.numel(), vocab, device="cuda")).half()
    labels = torch.randint(0, vocab, (len(lengths), seq), device="cuda")
    labels[~real] = -100
    labels[:, :8] = -100
    dense = torch.zeros(len(lengths) * seq, vocab, dtype=torch.float16, device="cuda")
    dense[rows] = packed
    lazy = PackedLogits(packed, rows, len(lengths), seq)
    return lazy, dense.view(len(lengths), seq, vocab), real, labels


def test_shape_and_window():
    lazy, dense, real, labels = make()
    assert lazy.shape == dense.shape and lazy.dtype == dense.dtype and lazy.ndim == 3
    window = lazy[..., :-1, :].contiguous()
    assert window.shape == dense[..., :-1, :].shape
    assert lazy[:, 0:, :].shape == lazy.shape


def test_argmax_like_trl_accuracy():
    lazy, dense, real, labels = make()
    shift_labels = labels[..., 1:]
    mask = shift_labels != -100
    mine = lazy[..., :-1, :].contiguous()[:, 0:, :].argmax(dim=-1)
    theirs = dense[..., :-1, :].contiguous().argmax(dim=-1)
    assert mine.shape == theirs.shape
    assert torch.equal(mine[real[:, :-1]], theirs[real[:, :-1]])
    assert torch.equal((mine == shift_labels) & mask, (theirs == shift_labels) & mask)


def test_entropy_and_row_losses_match_dense():
    from solution.logit_stats import entropy_from_logits, row_cross_entropy

    lazy, dense, real, labels = make()
    mine = entropy_from_logits(lazy)
    theirs = entropy_from_logits(dense)
    torch.testing.assert_close(mine[real], theirs[real], rtol=1e-4, atol=1e-4)
    assert float(mine[~real].abs().max()) == 0.0
    losses_a, counts_a = row_cross_entropy(lazy, labels)
    losses_b, counts_b = row_cross_entropy(dense, labels)
    assert counts_a == counts_b
    torch.testing.assert_close(torch.tensor(losses_a), torch.tensor(losses_b), rtol=1e-4, atol=1e-4)


def test_fp32_conversion_stays_packed_and_exact():
    """accelerate's convert_to_fp32 (fp16 mixed precision) calls .float() on the model outputs."""
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from solution.logit_stats import entropy_from_logits, row_cross_entropy
    from solution.packed_logits import PackedLogits

    lazy, dense, real, labels = make()
    try:
        from accelerate.utils import convert_to_fp32
    except ImportError:
        converted = lazy.float()
    else:
        out = convert_to_fp32(CausalLMOutputWithPast(loss=torch.zeros(()), logits=lazy))
        converted = out.logits
    assert isinstance(converted, PackedLogits) and converted._dense is None
    assert converted.dtype == torch.float32 and converted.shape == dense.shape
    dense32 = dense.float()
    torch.testing.assert_close(entropy_from_logits(converted)[real],
                               entropy_from_logits(dense32)[real], rtol=1e-4, atol=1e-4)
    assert row_cross_entropy(converted, labels)[1] == row_cross_entropy(dense32, labels)[1]
    window = converted[..., :-1, :].contiguous()
    assert window.dtype == torch.float32
    assert torch.equal(window.argmax(dim=-1)[real[:, :-1]],
                       dense32[..., :-1, :].argmax(dim=-1)[real[:, :-1]])
    assert torch.equal(converted.dense(), dense32)


def test_anything_else_falls_back_to_the_padded_tensor():
    lazy, dense, real, labels = make()
    torch.testing.assert_close(lazy.float().sum(), dense.float().sum())
    assert torch.equal(lazy[real], dense[real])
