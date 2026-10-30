"""Correctness, gradient parity, SM75 shared-memory budget, and rule checks for `solution/`.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}
SM75_MAX_SHARED_MEM_BYTES = 65536


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (2, 63, 1536)])
def test_rmsnorm_matches_pytorch(dtype, shape):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)
    eps = 1e-6

    x_ref = x.detach().clone().float().requires_grad_(True)
    w_ref = weight.detach().clone().float().requires_grad_(True)
    var = x_ref.pow(2).mean(-1, keepdim=True)
    ref = x_ref * torch.rsqrt(var + eps) * w_ref
    ref.backward(grad.float())

    out = rmsnorm(x, weight, eps=eps)
    out.backward(grad)

    tol = 3e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), x_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(weight.grad.float(), w_ref.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_add_rmsnorm_matches_pytorch(dtype):
    from solution.rmsnorm import fused_add_rmsnorm

    torch.manual_seed(7)
    shape = (4, 127, 1536)
    res = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    delta = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32, requires_grad=True)
    g_s = torch.randn(shape, device="cuda", dtype=dtype)
    g_y = torch.randn(shape, device="cuda", dtype=dtype)
    eps = 1e-6

    res_ref = res.detach().clone().requires_grad_(True)
    del_ref = delta.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    s_ref = res_ref + del_ref
    s_f32 = s_ref.float()
    y_ref = (w_ref * (s_f32 * torch.rsqrt(s_f32.pow(2).mean(-1, keepdim=True) + eps))).to(dtype)
    torch.autograd.backward([s_ref, y_ref], [g_s, g_y])

    s_out, y_out = fused_add_rmsnorm(res, delta, weight, eps=eps)
    torch.autograd.backward([s_out, y_out], [g_s, g_y])

    tol = 3e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(s_out.float(), s_ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(y_out.float(), y_ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(res.grad.float(), res_ref.grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(delta.grad.float(), del_ref.grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(weight.grad.float(), w_ref.grad.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rope_matches_pytorch(dtype):
    from solution.rope import apply_rope, fused_qk_rope

    torch.manual_seed(42)
    b, h_q, h_kv, s, d = 2, 12, 2, 128, 128
    # Use realistic non-contiguous .transpose(1, 2) layout just like Qwen2Attention
    q_raw = torch.randn((b, s, h_q, d), device="cuda", dtype=dtype, requires_grad=True)
    k_raw = torch.randn((b, s, h_kv, d), device="cuda", dtype=dtype, requires_grad=True)
    cos = torch.randn((1, s, d), device="cuda", dtype=dtype)
    sin = torch.randn((1, s, d), device="cuda", dtype=dtype)
    gq = torch.randn((b, h_q, s, d), device="cuda", dtype=dtype)
    gk = torch.randn((b, h_kv, s, d), device="cuda", dtype=dtype)

    def ref_rot(x, c, sv):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat(
            (x1 * c[..., :half] - x2 * sv[..., :half], x2 * c[..., half:] + x1 * sv[..., half:]),
            dim=-1,
        )

    qr_ref = q_raw.detach().clone().requires_grad_(True)
    kr_ref = k_raw.detach().clone().requires_grad_(True)
    qo_ref = ref_rot(qr_ref.transpose(1, 2).float(), cos.unsqueeze(1).float(), sin.unsqueeze(1).float())
    ko_ref = ref_rot(kr_ref.transpose(1, 2).float(), cos.unsqueeze(1).float(), sin.unsqueeze(1).float())
    torch.autograd.backward([qo_ref, ko_ref], [gq.float(), gk.float()])

    qo, ko = fused_qk_rope(q_raw.transpose(1, 2), k_raw.transpose(1, 2), cos, sin)
    torch.autograd.backward([qo, ko], [gq, gk])

    tol = 2e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(qo.float(), qo_ref, rtol=tol, atol=tol)
    torch.testing.assert_close(ko.float(), ko_ref, rtol=tol, atol=tol)
    torch.testing.assert_close(q_raw.grad.float(), qr_ref.grad.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(k_raw.grad.float(), kr_ref.grad.float(), rtol=tol, atol=tol)

    # Single-tensor helper check
    x = torch.randn((b, h_q, s, d), device="cuda", dtype=dtype, requires_grad=True)
    out_single = apply_rope(x, cos.unsqueeze(1), sin.unsqueeze(1))
    assert out_single.shape == x.shape


@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
@pytest.mark.parametrize("double_quant", [True, False])
def test_nf4_dequant_and_lora_matches_bitsandbytes(quant_type, double_quant):
    import bitsandbytes as bnb
    from solution.nf4_linear import TritonNF4LinearFunc, dequantize_nf4_triton

    torch.manual_seed(11)
    w = torch.randn((1536, 1536), device="cuda", dtype=torch.float16)
    qw, qs = bnb.functional.quantize_4bit(
        w, blocksize=64, compress_statistics=double_quant, quant_type=quant_type
    )
    dq_ref = bnb.functional.dequantize_4bit(qw, qs)
    dq_tri = dequantize_nf4_triton(qw, qs, dtype=torch.float16)
    torch.testing.assert_close(dq_tri, dq_ref, rtol=1e-3, atol=1e-3)

    x = torch.randn((3, 79, 1536), device="cuda", dtype=torch.float16, requires_grad=True)
    bias = torch.randn((1536,), device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn((3, 79, 1536), device="cuda", dtype=torch.float16)

    x_ref = x.detach().clone().requires_grad_(True)
    b_ref = bias.detach().clone().requires_grad_(True)
    y_ref = torch.nn.functional.linear(x_ref, dq_ref, b_ref)
    y_ref.backward(grad)

    y_tri = TritonNF4LinearFunc.apply(x, qw, bias, qs, torch.float16)
    y_tri.backward(grad)
    # 18 / 2.36M dequant weights on an exact FP16 midpoint can differ by 1 ULP (4.88e-4),
    # shifting ~1-4 / 364k 1536-dim FP16 dot products (std ~39.2, in [32, 64)) by 1 FP16 ULP (0.03125 = 2^-5).
    torch.testing.assert_close(y_tri, y_ref, rtol=5e-3, atol=4e-2)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=5e-3, atol=4e-2)
    torch.testing.assert_close(bias.grad, b_ref.grad, rtol=2e-3, atol=2e-3)


def test_chunked_cross_entropy_matches_pytorch():
    from solution.cross_entropy import FusedChunkedHeadLossFunc

    torch.manual_seed(42)
    b, s, d, v = 2, 64, 1536, 151936
    hidden = torch.randn((b, s, d), device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn((v, d), device="cuda", dtype=torch.float16) * 0.02
    target = torch.randint(0, v, (b, s), device="cuda", dtype=torch.long)
    target[0, 10:25] = -100
    target[1, 0:15] = -100

    denom = (target != -100).sum().clamp(min=1)

    h_ref = hidden.detach().clone().float().requires_grad_(True)
    logits_ref = h_ref @ weight.float().T
    ref_token_loss = torch.nn.functional.cross_entropy(
        logits_ref.view(-1, v),
        target.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(b, s)
    ref_loss = ref_token_loss.sum() / denom
    log_p = torch.nn.functional.log_softmax(logits_ref, dim=-1)
    ref_entropy = -(log_p.exp() * log_p).sum(dim=-1)
    mask = target != -100
    ref_hits = (logits_ref.argmax(dim=-1)[mask] == target[mask]).sum()

    # Scale backward by 65536.0 just like FP16 GradScaler on Tesla T4!
    (ref_loss * 65536.0).backward()

    out_loss, token_entropy, token_loss, hits = FusedChunkedHeadLossFunc.apply(
        hidden, weight, target, 32, denom
    )
    (out_loss * 65536.0).backward()

    torch.testing.assert_close(out_loss.float(), ref_loss, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(token_loss.float(), ref_token_loss, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(token_entropy.float(), ref_entropy, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(hidden.grad.float(), h_ref.grad, rtol=5e-3, atol=5e-2)
    assert hits.item() == ref_hits.item()


def test_sm75_shared_memory_and_rule_compliance():
    """Verify AST check passes with 0 lookalikes/banned imports and all compiled kernels fit well within SM75 64KB smem."""
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    if str(repo / "harness") not in sys.path:
        sys.path.insert(0, str(repo / "harness"))

    from check_rules import source_kernels
    from common import SOLUTION_DIR

    own, _copies, lookalikes, banned, broken = source_kernels(SOLUTION_DIR)
    assert len(own) >= 6
    assert lookalikes == []
    assert banned == []
    assert broken == []

    from solution.cross_entropy import triton_fused_chunk_loss_kernel
    from solution.nf4_linear import triton_nf4_dequant_kernel
    from solution.rmsnorm import (
        triton_fused_add_rmsnorm_bwd,
        triton_fused_add_rmsnorm_fwd,
        triton_rmsnorm_bwd,
        triton_rmsnorm_fwd,
    )
    from solution.rope import triton_qk_rope_bwd, triton_qk_rope_fwd

    for jit_fn in [
        triton_fused_chunk_loss_kernel,
        triton_nf4_dequant_kernel,
        triton_rmsnorm_fwd,
        triton_rmsnorm_bwd,
        triton_fused_add_rmsnorm_fwd,
        triton_fused_add_rmsnorm_bwd,
        triton_qk_rope_fwd,
        triton_qk_rope_bwd,
    ]:
        for dev_cache in jit_fn.device_caches.values():
            cache_dict = dev_cache[0] if isinstance(dev_cache, tuple) else dev_cache
            for compiled in cache_dict.values():
                meta = getattr(compiled, "metadata", None)
                smem = int(getattr(meta, "shared", 0)) if meta is not None else 0
                assert smem <= SM75_MAX_SHARED_MEM_BYTES // 4, (
                    f"{jit_fn.fn.__name__} uses {smem} bytes shared memory"
                )


def test_packed_checkpointed_decoder_layer_lora_parity():
    """End-to-end parity check of unpadded batch compaction + custom decoder checkpointing
    + NF4 SwiGLU MLP + clone-free addmm LoRA (non-zero lora_A & lora_B) + chunked loss."""
    import copy
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    import solution
    from solution.cross_entropy import TritonChunkedCausalLMLoss
    from solution.padding import install_model_layer_optimizations

    def _quantize_model_4bit(m: torch.nn.Module):
        for name, child in list(m.named_children()):
            if isinstance(child, torch.nn.Linear) and name != "lm_head":
                l4 = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.float16,
                    compress_statistics=True,
                    quant_type="nf4",
                )
                l4.weight = bnb.nn.Params4bit(
                    child.weight.data.to(torch.float16),
                    requires_grad=False,
                    compress_statistics=True,
                    quant_type="nf4",
                )
                if child.bias is not None:
                    l4.bias = torch.nn.Parameter(child.bias.data.to(torch.float16), requires_grad=False)
                setattr(m, name, l4.to("cuda"))
            else:
                _quantize_model_4bit(child)

    torch.manual_seed(123)
    cfg = Qwen2Config(
        vocab_size=4096,
        hidden_size=256,
        intermediate_size=640,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    base_raw = Qwen2ForCausalLM(cfg).to(dtype=torch.float16, device="cuda")
    _quantize_model_4bit(base_raw)
    for p in base_raw.parameters():
        p.requires_grad = False
        if p.ndim == 1 and p.is_floating_point():
            p.data = p.data.float()

    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
    ref_model = get_peft_model(base_raw, lora_cfg)
    # Initialize lora_B non-zero so BOTH lora_A.grad and lora_B.grad are exercised!
    for n, p in ref_model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.02)

    sol_model = copy.deepcopy(ref_model)

    b, s = 4, 64
    lengths = [37, 64, 19, 51]
    input_ids = torch.zeros((b, s), dtype=torch.long, device="cuda")
    attention_mask = torch.zeros((b, s), dtype=torch.long, device="cuda")
    labels = torch.full((b, s), -100, dtype=torch.long, device="cuda")
    for i, ln in enumerate(lengths):
        input_ids[i, :ln] = torch.randint(10, 4090, (ln,), device="cuda")
        attention_mask[i, :ln] = 1
        labels[i, ln // 3 : ln] = input_ids[i, ln // 3 : ln]

    with torch.amp.autocast("cuda", dtype=torch.float16):
        ref_out = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    (ref_out.loss * 1024.0).backward()

    solution.apply()
    causal_lm = sol_model.base_model.model
    install_model_layer_optimizations(causal_lm)
    hook = TritonChunkedCausalLMLoss(causal_lm, chunk_tokens=128)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        sol_out = sol_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    (sol_out.loss * 1024.0).backward()
    step_info = hook.take_step()

    assert step_info is not None
    torch.testing.assert_close(sol_out.loss.float(), ref_out.loss.float(), rtol=3e-3, atol=3e-3)

    ref_params = dict(ref_model.named_parameters())
    for name, p_sol in sol_model.named_parameters():
        if p_sol.requires_grad:
            p_ref = ref_params[name]
            assert p_sol.grad is not None and p_ref.grad is not None, f"Missing grad for {name}"
            cos = torch.nn.functional.cosine_similarity(
                p_sol.grad.flatten().float(), p_ref.grad.flatten().float(), dim=0
            ).item()
            assert cos > 0.999, f"Low grad cosine similarity on {name}: {cos}"

