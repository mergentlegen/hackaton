"""The whole packed training step against transformers' own forward on a tiny Qwen2.

Run last on purpose: `solution.apply()` patches the transformers classes for the rest of the
process, so the original forward is measured first.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)


def test_packed_step_matches_original_forward():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import solution

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=1000, hidden_size=512, intermediate_size=1024,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256)
    model = Qwen2ForCausalLM._from_config(config, attn_implementation="sdpa").cuda().train()
    lengths = [40, 25, 33]
    ids = torch.randint(0, 1000, (3, 40), device="cuda")
    mask = torch.zeros(3, 40, dtype=torch.long, device="cuda")
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[:, :8] = -100

    def step():
        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return out.loss.detach(), out.logits.detach().clone(), grads

    ref_loss, ref_logits, ref_grads = step()
    solution.apply()
    loss, logits, grads = step()

    torch.testing.assert_close(loss, ref_loss, rtol=3e-3, atol=3e-3)
    assert logits.shape == ref_logits.shape == (3, 40, 1000)
    real = mask.bool()
    torch.testing.assert_close(logits[real].float(), ref_logits[real].float(), rtol=3e-2, atol=3e-2)
    assert float(logits[~real].abs().max()) == 0.0  # padding rows are zeros
    assert set(grads) == set(ref_grads)
    for name in ref_grads:
        assert cosine(grads[name], ref_grads[name]) > 0.995, name
