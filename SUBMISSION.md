# Submission: hack-1d15670c-team

## Links
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
| `nf4_dequant_kernel` | `solution/nf4.py` | Vectorized 32-bit word NF4 double-quant dequantizer writing into static reusable buffer `_DEQUANT_BUF` | `6.3%` |
| `silu_mul_kernel` | `solution/kernels.py` | Unified 2-mode FP16 SwiGLU forward & in-place backward kernel with single compilation | `4.1%` |
| `rmsnorm_kernel` | `solution/rmsnorm.py` | Unified FP32 residual stream RMSNorm forward & backward kernel caching 1 FP32 `rstd` scalar per token | `1.1%` |
| `qkv_kernel` | `solution/attention.py` | Fused QKV Rotary Position Embedding & GQA head replication (12 heads) + backward accumulation | `1.1%` |
| `ce_kernel` | `solution/kernels.py` | 2-pass online Cross-Entropy + exact Shannon entropy + token argmax accuracy + backward dlogits | `0.9%` |
| **Total own kernels (excluding example copies)** | `solution/` | **Complete end-to-end non-GEMM compute & dequantization suite** | **`17.2%`** ($\ge 5.0\%$ required) |

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| Metric | Base Soup 0.77.0 | Our Solution | Official Ratio |
|---|---|---|---|
| Time of `soup train`, s | `78.06` | **`52.70`** | speedup **`1.481`x** |
| Training loop, s | `54.59` | **`30.11`** | speedup **`1.813`x** |
| Median step, s | `0.8459` | **`0.3666`** | speedup **`2.307`x** |
| Peak GPU memory, MiB | `14345` | **`1931`** | base/ours **`7.429`x** (86.5% cut) |
| Eval loss (public rows) | `2.04965` | **`2.04983`** | **`+0.009`%** vs base ($\le +0.5\%$ rule) |

- [x] All 10 checks of `harness/check_rules.py` passed (10/10 PASS)
- [x] Soup regression tests: 79 passed in 0.53s
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits
