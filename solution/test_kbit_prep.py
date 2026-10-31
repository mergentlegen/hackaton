"""peft's kbit preparation with the embedding table kept as the same fp16 values."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_embedding_table_stays_fp16_everything_else_as_peft_leaves_it():
    import peft
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.patches import patch_kbit_preparation

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=1000, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=True)

    def model():
        torch.manual_seed(0)
        return Qwen2ForCausalLM._from_config(config).cuda().half()

    reference = peft.prepare_model_for_kbit_training(model(), use_gradient_checkpointing=False)
    patch_kbit_preparation()
    mine = peft.prepare_model_for_kbit_training(model(), use_gradient_checkpointing=False)

    table = mine.get_input_embeddings().weight
    assert table.dtype == torch.float16 and not table.requires_grad
    assert mine.get_output_embeddings().weight is table
    assert torch.equal(table.float(), reference.get_input_embeddings().weight)
    for (name, p), (_, q) in zip(mine.named_parameters(), reference.named_parameters()):
        assert not p.requires_grad and not q.requires_grad
        if p is not table:
            assert p.dtype == q.dtype == torch.float32, name
            assert torch.equal(p, q), name
