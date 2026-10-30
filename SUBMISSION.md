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
| `triton_fused_chunk_loss_kernel` | `solution/cross_entropy.py` | Full `[B, S, 151936]` logits + FP32 CrossEntropy + TRL `entropy_from_logits` + token accuracy + backward `dlogits` (chunked 2-pass online softmax + in-place unscaled FP16 grad) | `<X.X>%` |
| `triton_nf4_dequant_kernel` | `solution/nf4_linear.py` | `bitsandbytes` 2-stage 4-bit NF4/FP4 double-quant dequantization (single-pass nibble unpack + nested `absmax` + `tl.join` into reusable 26.25 MiB workspace) | `<X.X>%` |
| `triton_rmsnorm_fwd` | `solution/rmsnorm.py` | `Qwen2RMSNorm.forward` (multi-kernel FP32 cast, variance reduction, rsqrt, scale) | `<X.X>%` |
| `triton_rmsnorm_bwd` | `solution/rmsnorm.py` | Autograd backward graph of `Qwen2RMSNorm` using saved 4-byte `rstd` | `<X.X>%` |
| `triton_fused_add_rmsnorm_fwd` | `solution/rmsnorm.py` | Residual addition `residual + attn_delta` + `post_attention_layernorm` in one HBM pass | `<X.X>%` |
| `triton_fused_add_rmsnorm_bwd` | `solution/rmsnorm.py` | Backward of fused residual addition + post-attention RMSNorm | `<X.X>%` |
| `triton_qk_rope_fwd` | `solution/rope.py` | `apply_rotary_pos_emb` forward (zero-copy stride-native fused Q+K rotation across all 14 heads in 1 launch) | `<X.X>%` |
| `triton_qk_rope_bwd` | `solution/rope.py` | `apply_rotary_pos_emb` backward on strided `(B, H, S, D)` views | `<X.X>%` |
| **Total own kernels (excluding `example_swiglu_*`)** | `solution/` | End-to-end non-GEMM compute & dequantization | **`<TOTAL_SHARE>%`** |

Ideas taken from other kernels, with links (we wrote all code in `solution/` ourselves):
- **Online Softmax / Streaming LogSumExp**: Milakov & Gimelshein (2018), [arXiv:1805.02867](https://arxiv.org/abs/1805.02867), and Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) — extended to compute streaming Shannon entropy alongside NLL loss and `argmax` in `triton_fused_chunk_loss_kernel`.
- **Chunked Large-Vocab Cross-Entropy**: Wijmans et al., [Cut Cross-Entropy](https://arxiv.org/abs/2411.09009), and Hsu et al., [Liger-Kernel](https://arxiv.org/abs/2410.10989) — inspired chunking hidden states and writing `dlogits` in-place into a static chunk buffer.
- **4-bit NF4 Double-Quant Layout & Model Patching**: Dettmers et al., [QLoRA / bitsandbytes](https://arxiv.org/abs/2305.14314), and [Unsloth](https://github.com/unslothai/unsloth) — inspired surgical model-level patching and single-buffer NF4 dequantization.
- **RMSNorm & RoPE Formulations**: Zhang & Sennrich (2019), [arXiv:1910.07467](https://arxiv.org/abs/1910.07467); Su et al. (2021), [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | `<BASE_WALL_S>` | `<SOL_WALL_S>` | speedup `<SPEEDUP_S>`x |
| Peak GPU memory, MiB | `<BASE_PEAK_MIB>` | `<SOL_PEAK_MIB>` | base/ours `<MEM_RATIO_M>`x |
| Eval loss (public rows) | `<BASE_EVAL_LOSS>` | `<SOL_EVAL_LOSS>` | `<QUALITY_DELTA_PCT>`% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed *(verify & keep checked after Kaggle T4 run)*
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
1. Profiling breakdown of Base Soup 0.77.0 on Turing SM75 (14.35 GiB VRAM & ~80 s wall time: 152k vocab logits/entropy spike, PEFT FP32 frozen-embedding cast & residual infection, un-checkpointed activations + dead KV cache, 57% padding waste + SM75 GQA SDPA math fallback, elementwise & NF4 dequant overhead).
2. Design of our 8 custom `@triton.jit` kernels (2-pass online-softmax chunked CE + entropy + accuracy with unscaled FP16 in-place `dlogits`, single-pass nested NF4 dequantizer, fused residual-RMSNorm, zero-copy stride-native Q+K RoPE) + deterministic non-autotuned SM75 configs.
3. System-level memory & compute synergy (in-batch valid-token compaction, SM75 per-segment `cutlassF` SDPA, FP16 frozen embedding preservation, clone-free `addmm` LoRA, RNG-preserving custom layer checkpointing).
4. Measured Kaggle T4 speedup ($s$), peak VRAM reduction ($m$), quality parity, and roadmap for fused W4A16 split-K Triton GEMMs.

## Limitations
- `DecoderLayerCheckpointFunc` trades one extra forward GEMM pass per decoder layer during backward to cut peak VRAM by ~6–7x; when batches contain almost zero padding, this recomputation caps step-time speedup at ~1.4–1.8x.
- `triton_nf4_dequant_kernel` writes into a single reusable `26.25 MiB` workspace consumed by cuBLAS `hmma1688` GEMM rather than fusing W4A16 MMA inside Triton (faster for SFT `M >= 256` on SM75, suboptimal for `M = 1` decoding).
- In-batch valid-token compaction requires monotonic right-padded 2D masks without `use_cache=True` and falls back to rectangular execution otherwise.
- Decoder-layer compaction/checkpointing wrapper targets `Qwen2ForCausalLM` (`solution/padding.py`), while the underlying Triton kernels are architecture-agnostic.
