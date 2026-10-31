# Submission: hack-1d15670c-team

## Links
- Final Commit (Official Repo): https://github.com/BAITC-Hacks/hack-1d15670c-team/commit/284d68e3cf6bb77bd5eebd9e709f91456fbdab02
- Hackathon Repository: https://github.com/BAITC-Hacks/hack-1d15670c-team
- Primary GitHub Repository: https://github.com/nursayatdev/nurskernel-byte
- Public Kaggle notebook (official T4 benchmark): https://www.kaggle.com/code/nurale/soup-kernel-t4-bench
- Demo video (up to 3 minutes): <DEMO_VIDEO_LINK>
- Presentation: <PRESENTATION_LINK>

## Team
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Nursayat | Captain / ML Systems & Triton Kernels | nursayatdev | nurale | @nursayat |

## Our Kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | Vectorized 32-bit word NF4 double-quant dequantizer writing into static reusable buffer `_DEQUANT_BUF` | `6.8%` |
| `silu_mul_kernel` | `solution/kernels.py` | Unified 2-mode FP16 SwiGLU forward & in-place backward kernel with single compilation | `4.2%` |
| `rmsnorm_kernel` | `solution/rmsnorm.py` | Unified FP32 residual stream RMSNorm + fused Add-RMSNorm forward & backward kernel | `2.5%` |
| `qkv_kernel` | `solution/attention.py` | Fused QKV Rotary Position Embedding & GQA head replication (12 heads) + backward accumulation | `1.2%` |
| `ce_kernel` | `solution/kernels.py` | 2-pass online Cross-Entropy + exact Shannon entropy + token argmax accuracy + backward dlogits | `1.0%` |
| **Total own kernels (excluding example copies)** | `solution/` | **Complete end-to-end non-GEMM compute & dequantization suite** | **`19.1%`** ($\ge 5.0\%$ required) |

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the benchmark)
| Metric | Base Soup 0.77.0 | Our Solution | Official Ratio |
|---|---|---|---|
| Time of `soup train`, s | `76.81` | **`51.53`** (fastest: **`50.98`**) | speedup **`1.491`x** |
| Training loop, s | `54.19` | **`29.25`** (fastest: **`28.74`**) | speedup **`1.853`x** |
| Median step, s | `0.8564` | **`0.3538`** | speedup **`2.421`x** |
| Peak GPU memory, MiB | `14345` | **`1869`** | base/ours **`7.675`x** (87.0% cut) |
| Eval loss (public rows) | `2.04966` | **`2.04991`** | **`+0.012`%** vs base ($\le +0.5\%$ rule) |

- [x] All 10 checks of `harness/check_rules.py` passed (10/10 PASS)
- [x] Soup regression tests: 79 passed in 0.42s
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits
