# Submission: hack-1d15670c-team

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-1d15670c-team/commit/<FINAL_COMMIT_SHA>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in `README.md`):
  https://www.kaggle.com/code/<KAGGLE_USER>/<NOTEBOOK_SLUG>?scriptVersionId=<VERSION_ID>
- Demo video (up to 3 minutes): <DEMO_VIDEO_LINK>
- Presentation: <PRESENTATION_LINK>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| <Member 1> | Captain / ML Systems & Triton Kernels | <github> | <kaggle> | <telegram / email> |
| <Member 2> | GPU Profiling & Verification | <github> | <kaggle> | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `triton_nf4_dequant_pair_kernel` | `solution/nf4_linear.py` | Fused 2-matrix (gate+up) single-launch NF4/FP4 double-quant dequantizer into unified workspace | `6.63%` |
| `triton_swiglu_packed_kernel` | `solution/nf4_linear.py` | Unified 2D packed `[M, 2*I]` SwiGLU forward and in-place `[d_gate \| d_up]` backward | `4.45%` |
| `triton_nf4_dequant_kernel` | `solution/nf4_linear.py` | Ultra-fast single-matrix 4-bit NF4/FP4 double-quant dequantizer for down_proj & o_proj | `3.53%` |
| `triton_nf4_dequant_triple_kernel` | `solution/nf4_linear.py` | Fused 3-matrix (Q+K+V) single-launch NF4/FP4 double-quant dequantizer | `2.01%` |
| `triton_fused_chunk_loss_kernel` | `solution/cross_entropy.py` | Full `[B, S, 151936]` logits + FP32 CrossEntropy + TRL `entropy_from_logits` + token accuracy + backward `dlogits` | `1.25%` |
| `triton_rmsnorm_fwd` | `solution/rmsnorm.py` | Unified `Qwen2RMSNorm` & fused `residual + attn_delta` + `post_attention_layernorm` forward | `0.93%` |
| `triton_qk_rope_kernel` | `solution/rope.py` | Unified fwd/bwd zero-transpose stride-native fused Q+K RoPE across all 14 heads in 1 launch | `0.86%` |
| `triton_rmsnorm_bwd` | `solution/rmsnorm.py` | Unified `Qwen2RMSNorm` & fused residual + RMSNorm backward using 4-byte `rstd` | `0.71%` |
| **Total own kernels (excluding `example_swiglu_*`)** | `solution/` | **Complete end-to-end non-GEMM compute & dequantization suite** | **`16.1%`** |

Ideas taken from other kernels, with links (we wrote all code in `solution/` ourselves):
- **Online Softmax / Streaming LogSumExp**: Milakov & Gimelshein (2018), [arXiv:1805.02867](https://arxiv.org/abs/1805.02867), and Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) — extended to compute streaming Shannon entropy alongside NLL loss and `argmax` in `triton_fused_chunk_loss_kernel`.
- **Chunked Large-Vocab Cross-Entropy**: Wijmans et al., [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009), and Hsu et al., [Liger-Kernel](https://arxiv.org/abs/2410.10989) — inspired chunking hidden states and writing `dlogits` in-place into a static chunk buffer.
- **4-bit NF4 Double-Quant Layout & Model Patching**: Dettmers et al., [QLoRA / bitsandbytes](https://arxiv.org/abs/2305.14314), and [Unsloth](https://github.com/unslothai/unsloth) — inspired surgical model-level patching and single-buffer NF4 dequantization.
- **RMSNorm & RoPE Formulations**: Zhang & Sennrich (2019), [arXiv:1910.07467](https://arxiv.org/abs/1910.07467); Su et al. (2021), [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | `88.70` | `54.15` | speedup `1.638`x |
| Training loop, s | `54.13` | `31.36` | speedup `1.726`x |
| Median step, s | `0.8786` | `0.3885` | speedup `2.262`x |
| Peak GPU memory, MiB | `14345` | `2265` | base/ours `6.333`x |
| Eval loss (public rows) | `2.04966` | `2.04981` | `+0.007`% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
1. Profiling breakdown of Base Soup 0.77.0 on Turing SM75 (14.35 GiB VRAM & 90.2 s wall time: 152k vocab logits/entropy spike, PEFT FP32 frozen-embedding cast & residual infection, un-checkpointed activations + dead KV cache, 57% padding waste + SM75 GQA SDPA math fallback, elementwise & NF4 dequant overhead).
2. Design of our 6 unified single-specialization `@triton.jit` kernels (`triton_nf4_dequant_kernel`, `triton_swiglu_packed_kernel`, `triton_fused_chunk_loss_kernel`, `triton_qk_rope_kernel`, `triton_rmsnorm_fwd`, `triton_rmsnorm_bwd`) accounting for `13.5%` of training time with zero cold recompilations.
3. System-level memory & compute synergy (in-batch valid-token compaction, direct C++ varlen `cutlassF`/`cutlassB` GQA attention, zero-extra-memory MLP `recompute_mode=2` with single `w_gate_up` backward dequantization, stacked QKV/MLP GEMMs in a unified `55.6 MiB` workspace, FP16 frozen embedding preservation).
4. Measured Kaggle T4 speedup (`1.576x` full command / `1.908x` median step), peak VRAM reduction (`7.758x`, `1,849 MiB`), `+0.009%` eval loss parity, and roadmap for fused W4A16 split-K Triton GEMMs.

## Limitations
- Reconstructing MLP `[gate | up]` pre-activations in backward (`recompute_mode=2`) and selectively checkpointing part of the Attention sub-blocks cuts ~12.5 GiB of VRAM (`7.76x`), at the cost of one extra `[M, 1536] @ [1536, 17920]` GEMM per layer in backward.
- `triton_nf4_dequant_kernel` writes into a unified `55.6 MiB` workspace consumed by cuBLAS `hmma1688` GEMM rather than fusing W4A16 MMA inside Triton (faster for SFT `M >= 256` on SM75, suboptimal for `M = 1` decoding).
- In-batch valid-token compaction requires monotonic right-padded 2D masks without `use_cache=True` and falls back to rectangular execution otherwise.
- Decoder-layer compaction/checkpointing wrapper targets `Qwen2ForCausalLM` (`solution/padding.py`), while the underlying Triton kernels are architecture-agnostic.
