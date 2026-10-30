"""Our kernels and the packed forward against plain PyTorch / the original Qwen2 forward.

On a GPU (Kaggle) the kernels run in fp16. Without a GPU the tests run the same Triton kernels in
Triton's CPU interpreter in fp32, which is how they were developed.
"""

import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")

from solution import kernels as K  # noqa: E402

DEVICE = "cuda" if HAS_CUDA else "cpu"
HALF = HAS_CUDA or os.environ.get("SOLUTION_TEST_FP16") == "1"  # fp16 also runs on the CPU interpreter
DT = torch.float16 if HALF else torch.float32
TOL = {"rtol": 2e-2, "atol": 2e-2} if HALF else {"rtol": 1e-4, "atol": 1e-5}


@pytest.fixture(autouse=True)
def _compute_dtype(monkeypatch):
    monkeypatch.setattr(K, "CDT", DT)


def close(a, b, **kw):
    torch.testing.assert_close(a.float(), b.float(), **(kw or TOL))


def rms_ref(s, w, eps):
    s32 = s.float()
    return w * (s32 * torch.rsqrt(s32.pow(2).mean(-1, keepdim=True) + eps))


def test_add_rmsnorm_forward_backward():
    torch.manual_seed(0)
    n, h, eps = 7, 96, 1e-6
    x = torch.randn(n, h, device=DEVICE, requires_grad=True)
    d = torch.randn(n, h, device=DEVICE, dtype=DT, requires_grad=True)
    w = torch.rand(h, device=DEVICE) + 0.5
    gs = torch.randn(n, h, device=DEVICE)
    gy = torch.randn(n, h, device=DEVICE, dtype=DT)

    s, y = K.AddRMSNorm.apply(x, d, w, eps)
    assert y.dtype == DT and s.dtype == torch.float32
    ((s * gs).sum() + (y.float() * gy.float()).sum()).backward()

    # reference: s feeds the residual path (gs) and the norm (gy)
    x2 = x.detach().clone().requires_grad_(True)
    d2 = d.detach().clone().requires_grad_(True)
    s2 = x2 + d2.float()
    y2 = rms_ref(s2, w, eps)
    ((s2 * gs).sum() + (y2 * gy.float()).sum()).backward()
    close(s, s2)
    close(y, y2)
    close(x.grad, x2.grad)
    close(d.grad, d2.grad)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def test_attn_prep_and_gather_match_reference():
    torch.manual_seed(1)
    b, t, nq, nkv, d = 3, 9, 4, 2, 16
    lens = torch.tensor([9, 5, 2], device=DEVICE)
    n = int(lens.sum())
    offsets = torch.cumsum(lens, 0) - lens
    tok_b = torch.repeat_interleave(torch.arange(b, device=DEVICE), lens, output_size=n)
    tok_t = (torch.arange(n, device=DEVICE) - offsets[tok_b]).to(torch.int32)
    tok_b = tok_b.to(torch.int32)
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float32, device=DEVICE) / d))
    freqs = torch.arange(t, device=DEVICE)[:, None].float() * inv[None, :]
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos().contiguous(), emb.sin().contiguous()
    geo = K.Geometry(tok_b, tok_t, b, t, cos, sin, nq, nkv, d)

    mk = lambda *s: torch.randn(*s, device=DEVICE, dtype=DT, requires_grad=True)  # noqa: E731
    qb, ql, kb, vb, vl = mk(n, nq * d), mk(n, nq * d), mk(n, nkv * d), mk(n, nkv * d), mk(n, nkv * d)
    qp, kp, vp = K.AttnPrep.apply(qb, ql, kb, vb, vl, geo)

    # reference: scatter to the padded layout, then the HF maths
    def pad(x, heads):
        full = torch.zeros(b, t, heads, d, device=DEVICE, dtype=torch.float32)
        full[tok_b.long(), tok_t.long()] = x.float().view(n, heads, d)
        return full.transpose(1, 2)

    qr, kr, vr = pad(qb.float() + ql.float(), nq), pad(kb, nkv), pad(vb.float() + vl.float(), nkv)
    c, s = cos[None, None], sin[None, None]
    qr = qr * c + rotate_half(qr) * s
    kr = kr * c + rotate_half(kr) * s
    kr = kr[:, :, None].expand(b, nkv, nq // nkv, t, d).reshape(b, nq, t, d)
    vr = vr[:, :, None].expand(b, nkv, nq // nkv, t, d).reshape(b, nq, t, d)
    close(qp, qr)
    close(kp, kr)
    close(vp, vr)

    gq, gk, gv = (torch.randn_like(qp), torch.randn_like(kp), torch.randn_like(vp))
    (qp * gq).sum().backward(retain_graph=True)
    (kp * gk).sum().backward(retain_graph=True)
    (vp * gv).sum().backward()
    # reference gradients by autograd on the float reference
    qb2, ql2, kb2, vb2, vl2 = [x.detach().clone().float().requires_grad_(True)
                               for x in (qb, ql, kb, vb, vl)]
    q3, k3, v3 = pad(qb2 + ql2, nq), pad(kb2, nkv), pad(vb2 + vl2, nkv)
    q3 = q3 * c + rotate_half(q3) * s
    k3 = k3 * c + rotate_half(k3) * s
    k3 = k3[:, :, None].expand(b, nkv, nq // nkv, t, d).reshape(b, nq, t, d)
    v3 = v3[:, :, None].expand(b, nkv, nq // nkv, t, d).reshape(b, nq, t, d)
    ((q3 * gq.float()).sum() + (k3 * gk.float()).sum() + (v3 * gv.float()).sum()).backward()
    close(qb.grad, qb2.grad)
    close(ql.grad, ql2.grad)
    close(kb.grad, kb2.grad)
    close(vb.grad, vb2.grad)
    close(vl.grad, vl2.grad)

    o = torch.randn(b, nq, t, d, device=DEVICE, dtype=DT, requires_grad=True)
    out = K.GatherHeads.apply(o, geo)
    ref = o.float().transpose(1, 2)[tok_b.long(), tok_t.long()].reshape(n, nq * d)
    close(out, ref)
    g = torch.randn_like(out)
    out.backward(g)
    expect = torch.zeros(b, t, nq, d, device=DEVICE)
    expect[tok_b.long(), tok_t.long()] = g.float().view(n, nq, d)
    close(o.grad, expect.transpose(1, 2))


def test_swiglu():
    torch.manual_seed(2)
    g = torch.randn(5, 300, device=DEVICE, dtype=DT, requires_grad=True)
    u = torch.randn(5, 300, device=DEVICE, dtype=DT, requires_grad=True)
    go = torch.randn(5, 300, device=DEVICE, dtype=DT)
    g2, u2 = g.detach().float().requires_grad_(True), u.detach().float().requires_grad_(True)
    ref = torch.nn.functional.silu(g2) * u2
    ref.backward(go.float())
    out = K.SwiGLU.apply(g, u)
    close(out, ref)
    out.backward(go)
    close(g.grad, g2.grad)
    close(u.grad, u2.grad)


def test_dropout_mask_is_reproducible_and_scaled():
    x = torch.ones(4096, device=DEVICE, dtype=DT)
    a = K.dropout_from_seed(x, 0.25, 123)
    b = K.dropout_from_seed(x, 0.25, 123)
    c = K.dropout_from_seed(x, 0.25, 124)
    assert torch.equal(a, b) and not torch.equal(a, c)
    kept = (a != 0).float().mean().item()
    assert 0.70 < kept < 0.80
    assert torch.allclose(a[a != 0].float(), torch.full_like(a[a != 0].float(), 1 / 0.75), rtol=1e-3)


@pytest.mark.parametrize("p", [0.0, 0.1])
def test_lora_delta_matches_reference(p):
    torch.manual_seed(3)
    n, i, o, r, scaling = 11, 32, 24, 4, 2.0
    x = torch.randn(n, i, device=DEVICE, dtype=DT, requires_grad=True)
    a = torch.randn(r, i, device=DEVICE, requires_grad=True)
    b = torch.randn(o, r, device=DEVICE, requires_grad=True)
    g = torch.randn(n, o, device=DEVICE, dtype=DT)
    out = K.LoraDelta.apply(x, a, b, scaling, p, 77)
    out.backward(g)
    mask_scale = K.dropout_from_seed(torch.ones(n, i, device=DEVICE, dtype=DT), p, 77).float()
    x2 = x.detach().float().requires_grad_(True)
    a2, b2 = a.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True)
    ref = ((x2 * mask_scale) @ a2.t() @ b2.t()) * scaling
    ref.backward(g.float())
    close(out, ref)
    close(x.grad, x2.grad)
    close(a.grad, a2.grad)
    close(b.grad, b2.grad)


def test_fused_lm_head_loss_matches_cross_entropy():
    torch.manual_seed(4)
    n, h, v = 21, 32, 5000
    hidden = torch.randn(n, h, device=DEVICE, dtype=DT, requires_grad=True)
    weight = (torch.randn(v, h, device=DEVICE) * 0.2).to(DT)
    tgt = torch.randint(0, v, (n,), device=DEVICE)
    tgt[::4] = -100
    stats = {}
    loss = K.FusedLMHeadLoss.apply(hidden, weight, tgt.to(torch.int32), stats, 8)
    loss.backward()

    h2 = hidden.detach().float().requires_grad_(True)
    logits = h2 @ weight.float().t()
    ref = torch.nn.functional.cross_entropy(logits, tgt, ignore_index=-100, reduction="sum")
    ref.backward()
    tol = {"rtol": 3e-2, "atol": 3e-2} if HALF else {"rtol": 1e-3, "atol": 1e-4}
    close(loss, ref, **tol)
    close(hidden.grad, h2.grad, **tol)
    lp = torch.log_softmax(logits.detach(), -1)
    ent = -(lp.exp() * lp).sum(-1)
    close(stats["entropy"], ent, **tol)
    assert (stats["pred"].long() == logits.detach().argmax(-1)).float().mean() > 0.9
    nll = torch.nn.functional.cross_entropy(logits.detach(), tgt, ignore_index=-100, reduction="none")
    close(stats["nll"], nll, **tol)


# ---- the whole packed forward against the original Qwen2 forward ---------------------------
def _tiny_model():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=700, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=True, attention_dropout=0.0)
    torch.manual_seed(5)
    lm = Qwen2ForCausalLM(cfg).to(DEVICE).float()
    for p in lm.parameters():
        p.requires_grad_(False)
    for m in lm.modules():
        if type(m).__name__ == "Qwen2RMSNorm":
            m.weight.data = torch.rand_like(m.weight) + 0.5
    peft_model = get_peft_model(lm, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                               target_modules=["q_proj", "v_proj"]))
    for n_, p in peft_model.named_parameters():
        if "lora_B" in n_:
            p.data.normal_(0, 0.05)
    return lm, peft_model


def test_packed_forward_matches_original_forward():
    from solution import qwen_fast

    qwen_fast.install()
    lm, peft_model = _tiny_model()
    lm.train()
    torch.manual_seed(6)
    lens = [12, 7, 3]
    t = max(lens)
    ids = torch.randint(0, 700, (3, t), device=DEVICE)
    mask = torch.zeros(3, t, dtype=torch.long, device=DEVICE)
    labels = torch.full((3, t), -100, dtype=torch.long, device=DEVICE)
    for r, length in enumerate(lens):
        mask[r, :length] = 1
        labels[r, length // 2:length] = ids[r, length // 2:length]
    num_items = (labels[:, 1:] != -100).sum()
    params = [p for p in peft_model.parameters() if p.requires_grad]

    ref = qwen_fast._original["forward"](lm, input_ids=ids, attention_mask=mask, labels=labels,
                                         num_items_in_batch=num_items, use_cache=False)
    ref.loss.backward()
    ref_grads = [p.grad.clone() for p in params]
    for p in params:
        p.grad = None

    out = peft_model(input_ids=ids, attention_mask=mask, labels=labels,
                     num_items_in_batch=num_items, use_cache=False)
    assert isinstance(out.logits, qwen_fast.FusedLogits), "fast path was not taken"
    out.loss.backward()
    tol = {"rtol": 5e-2, "atol": 5e-2} if HALF else {"rtol": 2e-3, "atol": 1e-5}
    close(out.loss, ref.loss, **tol)
    for g_ref, p in zip(ref_grads, params):
        close(p.grad, g_ref, **tol)

    # the metrics TRL / Soup read from the logits
    logits = ref.logits.float()
    lp = torch.log_softmax(logits, -1)
    ent_ref = -(lp.exp() * lp).sum(-1)
    ent = out.logits.entropy_padded()
    close(ent * mask, ent_ref * mask, **tol)
    rl, rt = out.logits.row_losses()
    for r, length in enumerate(lens):
        sl = logits[r, :-1][labels[r, 1:] != -100]
        tl_ = labels[r, 1:][labels[r, 1:] != -100]
        assert rt[r] == tl_.numel()
        if tl_.numel():
            expect = torch.nn.functional.cross_entropy(sl, tl_).item()
            assert abs(rl[r] - expect) < (0.1 if HALF else 1e-3)
