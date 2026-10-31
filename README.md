# Soup Kernel Challenge — Team `hack-1d15670c-team`

Custom `@triton.jit` kernel suite and SM75-aware memory/compute pipeline for `soup train` (`soup-cli==0.77.0`) fine-tuning **Qwen/Qwen2.5-1.5B-Instruct** (4-bit NF4 double-quant, LoRA `r=16, alpha=32` on `q_proj` & `v_proj`) on a single **Kaggle Tesla T4** (15 GiB VRAM, Turing SM75, FP16 AMP).

- **Primary GitHub Repository**: https://github.com/nursayatdev/nurskernel-byte
- **Hackathon Repository**: https://github.com/BAITC-Hacks/hack-1d15670c-team
- **Public Kaggle T4 Benchmark Notebook**: https://www.kaggle.com/code/nurale/soup-kernel-t4-bench

---

## 1. What We Did

### Root-Cause Bottlenecks in Base `soup train` on Tesla T4 (~14,350 MiB, ~80 s)
Profiling `soup-cli==0.77.0` + `transformers==5.16.1` + `peft==0.20.0` + `trl==0.29.1` + `bitsandbytes==0.50.2` on Turing SM75 exposed five compounding bottlenecks:

1. **Full-Vocabulary Logits & Metrics Spike (~4.25 GiB VRAM + 7% GPU time)**:
   Materializing `[B=4, S=512, V=151936]` logits alongside FP32 Cross-Entropy upcasts, TRL's `entropy_from_logits` (`log_softmax`), Soup's `rewind.jsonl` per-row loss reduction, and backward `dlogits` simultaneously allocates multiple ~0.6–1.2 GiB tensors.
2. **PEFT Frozen Embedding Upcast & Residual FP32 Infection (~1.33 GiB load spike + 392 extra casts/step)**:
   `peft.prepare_model_for_kbit_training` casts every non-`Params4bit` parameter to `fp32`—including the frozen tied `embed_tokens` / `lm_head` matrix `[151936, 1536]` (`445 MiB` $\rightarrow$ `890 MiB`). Because `F.embedding` is not downcast by `torch.amp.autocast`, `inputs_embeds` enters layer 0 in `fp32`, keeping the residual stream in `fp32` across all 28 layers and forcing `fp32 <-> fp16` casts around every quantized projection.
3. **Un-checkpointed 28-Layer Activations & Unused Training KV Cache (~8 GiB VRAM)**:
   Saving 28 layers of MLP (`gate`, `up` at `intermediate_size=8960` = `70 MiB/layer` alone), norms, QKV projections, and attention intermediates—plus an unused 28-layer `DynamicCache` allocated because `Qwen2Config.use_cache=True` by default when `gradient_checkpointing=False`—consumes ~8 GiB.
4. **Right-Padding Waste (~57% tokens) & Turing SM75 GQA SDPA Fallback**:
   Public training batches (`batch_size=4`, `max_length=512`) have mean length ~220 tokens (median ~140), leaving over half of rectangular `[4, 512]` tensors as padding.
5. **PCIe Memory Transfer Contention & Dynamic Allocation Churn**:
   Over-aggressive CPU offloading can choke PCIe bandwidth with thousands of small memory copies, while unbuffered tensor allocations create allocator contention.

---

### Our Architecture & Custom `@triton.jit` Kernels (`solution/`)

Everything plugs in through `solution.apply()` with **zero banned libraries**, **zero changes to `TRAINER_ARGS`**, and **zero `@triton.autotune`** (every kernel uses `@triton.jit(do_not_specialize=...)` with a shared argument layout to guarantee fast single-compilation on cold start):

| `@triton.jit` Kernel | File | What It Replaces & How It Works | Share (Check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | **Vectorized 32-bit word NF4 double-quant dequantizer**. Unpacks two nibbles per byte, resolves scales directly, and writes decoded weights into a static reused buffer `_DEQUANT_BUF`. | **`6.3%`** |
| `silu_mul_kernel` | `solution/kernels.py` | **Unified 2-mode FP16 SwiGLU forward & in-place backward kernel**. Computes `silu(gate) * up` in forward and exact gradients in backward within a single compiled kernel. | **`4.1%`** |
| `rmsnorm_kernel` | `solution/rmsnorm.py` | **Unified FP32 residual stream RMSNorm forward & backward kernel**. Single-pass reduction caching 1 FP32 `rstd` scalar per token row (`4 B`). | **`1.1%`** |
| `qkv_kernel` | `solution/attention.py` | **Fused QKV Rotary Position Embedding & GQA expansion kernel**. Handles forward rotation + GQA head replication (12 heads) and backward transposed gradient accumulation. | **`1.1%`** |
| `ce_kernel` | `solution/kernels.py` | **Fused 2-pass online Cross-Entropy + exact Shannon entropy + argmax token accuracy + in-place unscaled dlogits**. Online log-sum-exp without materializing FP32 logits. | **`0.9%`** |
| **Total own kernels (excluding example copies)** | `solution/` | **Complete end-to-end non-GEMM compute & dequantization suite** | **`17.2%`** ($\ge 5.0\%$ required) |

### Memory & Compute Optimizations
- **Padding-Free Packed Forward (`solution/packed.py`)**: Gathers valid non-padding tokens per batch, bypassing ~57% wasted compute and memory across all 28 transformer layers.
- **Asynchronous Pinned-Host Offloading (`solution/offload.py`)**: Asynchronously streams heavy MLP activations to pinned CPU host memory overlapped with compute stream execution. Buffer thresholds (`KEEP_BYTES = 224 MiB`, `PREFETCH_BYTES = 224 MiB`) keep top layers GPU-resident while prefetching ahead to eliminate PCIe stalls.
- **GPU Retention of Attention QKV (`solution/attention.py`)**: Keeps lightweight QKV projections (~90 MiB total across all 28 layers) resident on GPU, eliminating 10,752 individual PCIe transfers across the 64-step run.
- **Chunked Cross-Entropy with Static Buffer Reuse (`solution/lm_loss.py`)**: Computes `hidden @ lm_head.T` in row chunks into a pre-allocated reusable static buffer `_LOGITS_BUF`, eliminating 256 transient 74 MiB allocations per run.
- **FP16 Embedding Protection & Double-Quant Flattening (`solution/patches.py`)**: Preserves FP16 embedding weights during PEFT kbit initialization and decodes double-quantized nested scales once on step 1.

---

## 2. How to Run It

### A. On Kaggle (Official Tesla T4 Benchmark)
1. Open our public notebook: **https://www.kaggle.com/code/nurale/soup-kernel-t4-bench**.
2. Click **Save & Run All** (~10 minutes). The notebook automatically executes:
   - `python task/prepare_data.py`
   - `python harness/run_task.py --repeats 2 --profile`
   - `python harness/check_rules.py` (verifies all 10 admission checks, including 79 Soup tests $\rightarrow$ `runs/checks.json`)
   - `python harness/score.py runs/result.json`

### B. Locally on a Linux / CUDA Machine
```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest

python task/prepare_data.py                     # writes data/train.jsonl & data/eval_public.jsonl
python -m pytest solution -v                    # 11 correctness & parity tests
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q # 79 Soup regression tests
python harness/run_task.py --repeats 2          # runs base & solution -> runs/result.json
python harness/check_rules.py                   # verifies all 10 checks -> runs/checks.json
python harness/score.py runs/result.json        # prints speedup (s), memory ratio (m) & quality
```

---

## 3. Official Benchmark Results on Kaggle Tesla T4

| Metric | Base Soup 0.77.0 | Solution (Our Record Run) | Official Improvement | Rule / Requirement |
|---|---|---|---|---|
| Whole `soup train` wall time (`wall_s`) | `78.06` s | **`52.70` s** | **`1.481x` faster** | $\le 1.10\times$ margin (**PASS**) |
| 64-step training loop (`train_loop_s`) | `54.59` s | **`30.11` s** | **`1.813x` faster** | Measurable speedup (**PASS**) |
| Median step time (`median_step_s`) | `0.8459` s | **`0.3666` s** | **`2.307x` faster** | Steady-state throughput (**PASS**) |
| Peak GPU memory (`nvidia-smi`, `gpu_peak_mib`) | `14,345` MiB | **`1,931` MiB** | **`7.429x` lower (86.5% cut)** | $\le 1.10\times$ margin (**PASS**) |
| Held-out public eval loss (`eval_adapter.py`) | `2.04965` | **`2.04983`** | **`+0.009%` parity** | $\le +0.5\%$ rule (**PASS**) |
| Own `@triton.jit` share (Check 4, steps 3+) | — | **`17.2%`** | **`3.44x` above minimum** | $\ge 5.0\%$ required (**PASS**) |
| Soup regression tests (Check 10) | — | **`79 passed in 0.53s`** | **`100% test pass rate`** | 79 passed required (**PASS**) |
| `harness/check_rules.py` (10 / 10 checks) | — | **`10 / 10 PASS`** | **`100% Admission Passed`** | All 10 checks required (**PASS**) |
