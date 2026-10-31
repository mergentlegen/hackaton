"""The fused LoRA branch against peft's own 4-bit LoRA layer, forward and backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_fused_lora_matches_peft():
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import bnb as lora_bnb

    from solution.lora import fused_forward

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = bnb.nn.Linear4bit(1536, 1536, bias=True, compute_dtype=torch.float16,
                                            quant_type="nf4")
            self.v_proj = bnb.nn.Linear4bit(1536, 256, bias=True, compute_dtype=torch.float16,
                                            quant_type="nf4")

        def forward(self, x):
            return self.q_proj(x).sum() + self.v_proj(x).sum()

    torch.manual_seed(0)
    toy = Toy()
    toy.is_loaded_in_4bit = True
    toy = toy.cuda()
    model = get_peft_model(toy, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                           target_modules=["q_proj", "v_proj"]))
    model.train()
    layers = [m for m in model.modules() if isinstance(m, lora_bnb.Linear4bit)]
    assert len(layers) == 2
    for layer in layers:  # B starts at zero in peft: give it values so every gradient is tested
        torch.nn.init.normal_(layer.lora_B["default"].weight, std=0.02)
    peft_forward = getattr(lora_bnb.Linear4bit.forward, "peft_forward",
                           lora_bnb.Linear4bit.forward)
    x0 = torch.randn(300, 1536, device="cuda").half()

    def run(fused):
        model.zero_grad()
        x = x0.clone().requires_grad_()
        torch.manual_seed(1)  # same dropout masks for both runs
        with torch.autocast("cuda", dtype=torch.float16):
            outs = [fused_forward(layer, x) if fused else peft_forward(layer, x)
                    for layer in layers]
        assert all(o is not None for o in outs)
        sum((o.float() * (i + 1)).sum() for i, o in enumerate(outs)).backward()
        grads = [x.grad.clone()] + [p.grad.clone() for layer in layers
                                    for p in (layer.lora_A["default"].weight,
                                              layer.lora_B["default"].weight)]
        return [o.detach() for o in outs], grads

    ref_outs, ref_grads = run(False)
    outs, grads = run(True)
    for mine, theirs in zip(outs, ref_outs):
        assert mine.dtype == theirs.dtype
        torch.testing.assert_close(mine, theirs, rtol=2e-3, atol=2e-3)
    for mine, theirs in zip(grads, ref_grads):
        assert mine.dtype == theirs.dtype
        cos = torch.nn.functional.cosine_similarity(mine.float().flatten(),
                                                    theirs.float().flatten(), dim=0)
        assert cos > 0.9999
        torch.testing.assert_close(mine.float(), theirs.float(), rtol=1e-2, atol=1e-3)
