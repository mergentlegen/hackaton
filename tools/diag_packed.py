"""Per-parameter gradient cosines of the packed step against the original forward (analysis only).

    python tools/diag_packed.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import solution

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=1000, hidden_size=512, intermediate_size=1024,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256)
    model = Qwen2ForCausalLM._from_config(config, attn_implementation="sdpa").cuda().train()
    ids = torch.randint(0, 1000, (3, 40), device="cuda")
    right = torch.zeros(3, 40, dtype=torch.long, device="cuda")
    for row, length in enumerate([40, 25, 33]):
        right[row, :length] = 1
    masks = {"right": right, "left": right.flip(1)}

    def step(mask, fp32=False):
        labels = ids.clone()
        labels[mask == 0] = -100
        labels[:, :8] = -100
        labels[:, 1:][mask[:, :-1] == 0] = -100  # no target predicted from a padding position
        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        return float(out.loss), {n: p.grad.clone() for n, p in model.named_parameters()
                                 if p.grad is not None}

    exact = {k: step(m, fp32=True) for k, m in masks.items()}  # fp32 reference
    refs = {k: step(m) for k, m in masks.items()}
    solution.apply()
    mine = {k: step(m) for k, m in masks.items()}

    def cos(a, b):
        return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))

    for k in masks:
        print(k, "loss fp32", exact[k][0], "orig", refs[k][0], "packed", mine[k][0])
        worst = sorted(((cos(mine[k][1][n], refs[k][1][n]), cos(refs[k][1][n], exact[k][1][n]),
                         cos(mine[k][1][n], exact[k][1][n]), n) for n in refs[k][1]))[:5]
        for c_mo, c_oe, c_me, n in worst:
            print(f"  {n:45s} packed~orig {c_mo:.5f}  orig~fp32 {c_oe:.5f}  packed~fp32 {c_me:.5f}")


if __name__ == "__main__":
    main()
