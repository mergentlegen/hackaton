"""Packed (pad-free) training forward of Qwen2 with fused kernels, and the hooks that let
TRL / Soup read the metrics they need without the full logits.

What changes against plain Soup, on the same weights and the same maths:
  * pad tokens are removed before the first layer and never come back (only attention uses the
    padded layout), so every matmul and every elementwise pass sees only real tokens;
  * the residual stream stays fp32 but every large intermediate is fp16 (plain Soup keeps the
    8960-wide MLP activations in fp32 because bitsandbytes casts its output back to fp32);
  * frozen 4-bit layers save nothing for the backward, LoRA rebuilds its dropout mask from a seed;
  * the vocabulary projection, loss, entropy, accuracy and per-row losses come from one chunked
    fused pass; the 152k-wide logits never exist as a whole.

Anything unexpected (other model, left padding, eval mode, merged adapter ...) falls back to the
original forward, so a patch problem never changes what Soup trains.
"""

import torch
import torch.nn.functional as F

from . import kernels as K

CHUNK = 256  # rows of the vocabulary projection kept alive at once
_PLAN_ATTR = "_soup_fast_plan"
_original = {}


# --------------------------------------------------------------------------------------------
# frozen (4-bit or plain) linear layers
# --------------------------------------------------------------------------------------------
def _is_bnb4(mod):
    return type(mod).__name__ in ("Linear4bit", "LinearNF4", "LinearFP4") and \
        getattr(getattr(mod, "weight", None), "quant_state", None) is not None


def _dequant(mod, dtype):
    import bitsandbytes.functional as bf

    return bf.dequantize_4bit(mod.weight.data, mod.weight.quant_state).to(dtype)


class FrozenLinear(torch.autograd.Function):
    """y = x @ W.T + b with W frozen. Saves nothing: the backward only needs W again."""

    @staticmethod
    def forward(ctx, x, mod, bias):
        ctx.mod = mod
        x = x.contiguous()
        if _is_bnb4(mod):
            import bitsandbytes as bnb

            return bnb.matmul_4bit(x, mod.weight, bias=bias, quant_state=mod.weight.quant_state)
        return F.linear(x, mod.weight.to(x.dtype), bias)

    @staticmethod
    def backward(ctx, g):
        mod = ctx.mod
        if _is_bnb4(mod):
            w = _dequant(mod, g.dtype)
        else:
            w = mod.weight.to(g.dtype)
        return torch.mm(g.contiguous(), w), None, None


def frozen_linear(x, mod):
    bias = None if mod.bias is None else mod.bias.to(x.dtype)
    return FrozenLinear.apply(x, mod, bias)


# --------------------------------------------------------------------------------------------
# plan: which modules to use, validated once per model
# --------------------------------------------------------------------------------------------
class _Lora:
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def parts(self):
        w = self.wrapper
        drop = w.lora_dropout["default"]
        p = 0.0 if isinstance(drop, torch.nn.Identity) else float(drop.p)
        return w.lora_A["default"].weight, w.lora_B["default"].weight, float(w.scaling["default"]), p


def _lora_ok(wrapper):
    try:
        if set(wrapper.lora_A.keys()) != {"default"} or wrapper.disable_adapters or wrapper.merged:
            return False
        if list(wrapper.active_adapters) != ["default"] or getattr(wrapper, "lora_variant", None):
            return False
        drop = wrapper.lora_dropout["default"]
        if not isinstance(drop, (torch.nn.Identity, torch.nn.Dropout)):
            return False
        base = wrapper.get_base_layer()
        return base.bias is None or True
    except Exception:  # noqa: BLE001
        return False


class Plan:
    pass


def _build_plan(lm):
    cfg = lm.config
    model = lm.model
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_q
    layer_types = getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers
    if cfg.hidden_act != "silu" or n_q % n_kv or d % 2 or (d & (d - 1)) or \
            getattr(cfg, "attention_dropout", 0.0) != 0.0 or set(layer_types) != {"full_attention"}:
        return None
    layers = []
    for layer in model.layers:
        at, mlp = layer.self_attn, layer.mlp
        entry = {"ln1": layer.input_layernorm, "ln2": layer.post_attention_layernorm}
        for name in ("q_proj", "v_proj"):
            wrapper = getattr(at, name)
            if not hasattr(wrapper, "lora_A") or not _lora_ok(wrapper):
                return None
            entry[name] = (wrapper.get_base_layer(), _Lora(wrapper))
        for holder, name in ((at, "k_proj"), (at, "o_proj"), (mlp, "gate_proj"),
                             (mlp, "up_proj"), (mlp, "down_proj")):
            mod = getattr(holder, name)
            if hasattr(mod, "lora_A"):
                return None
            entry[name] = (mod, None)
        for mod, _ in [entry[n] for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                          "up_proj", "down_proj")]:
            if _is_bnb4(mod):
                if getattr(mod, "compute_dtype", None) not in (None, K.CDT):
                    return None
            elif type(mod) is not torch.nn.Linear:
                return None
        for norm in (entry["ln1"], entry["ln2"]):
            if norm.weight.requires_grad or norm.weight.dtype != torch.float32:
                return None
        layers.append(entry)
    if model.norm.weight.requires_grad or model.norm.weight.dtype != torch.float32:
        return None
    plan = Plan()
    plan.layers = layers
    plan.n_q, plan.n_kv, plan.d, plan.eps = n_q, n_kv, d, cfg.rms_norm_eps
    plan.embed_converted = False
    return plan


def get_plan(lm):
    plan = lm.__dict__.get(_PLAN_ATTR, 0)
    if plan == 0:
        try:
            plan = _build_plan(lm)
        except Exception:  # noqa: BLE001 - an unknown model layout just means: no fast path
            plan = None
        lm.__dict__[_PLAN_ATTR] = plan
    return plan


def _head_weight(lm, plan):
    """The tied embedding / lm_head matrix as fp16. It is frozen and holds fp16-born values,
    so the fp32 copy that k-bit preparation made can be dropped (saves half a GB)."""
    w = lm.lm_head.weight
    if w.dtype != K.CDT and not w.requires_grad:
        tied = lm.model.embed_tokens.weight is w
        w.data = w.data.to(K.CDT)
        if w.is_cuda:
            torch.cuda.empty_cache()
        plan.embed_converted = tied
    return w


# --------------------------------------------------------------------------------------------
# stand-ins for the logits that TRL / Soup read
# --------------------------------------------------------------------------------------------
class _Shifted:
    def __init__(self, parent):
        self.parent = parent

    def contiguous(self):
        return self

    def argmax(self, dim=-1):
        return self.parent.pred_padded()[:, :-1]


class FusedLogits:
    """What `outputs.logits` is on the fast path. It answers the three questions TRL and the
    rewind recorder ask of the logits (per-token entropy, argmax, per-row loss) from the
    statistics the fused loss kernel already computed."""

    def __init__(self, stats, geo, tgt):
        self.stats, self.geo, self.tgt = stats, geo, tgt

    def _flat_index(self):
        return self.geo.tok_b.long() * self.geo.seq + self.geo.tok_t.long()

    def _pad(self, packed, fill, dtype):
        out = torch.full((self.geo.batch * self.geo.seq,), fill, dtype=dtype,
                         device=packed.device)
        out[self._flat_index()] = packed.to(dtype)
        return out.view(self.geo.batch, self.geo.seq)

    def entropy_padded(self):
        return self._pad(self.stats["entropy"], 0.0, torch.float32)

    def pred_padded(self):
        return self._pad(self.stats["pred"], -1, torch.long)

    def __getitem__(self, _):
        return _Shifted(self)

    def row_losses(self):
        geo = self.geo
        valid = (self.tgt >= 0)
        rows = geo.tok_b.long()
        total = torch.zeros(geo.batch, dtype=torch.float32, device=self.tgt.device)
        count = torch.zeros(geo.batch, dtype=torch.float32, device=self.tgt.device)
        total.index_add_(0, rows, self.stats["nll"] * valid)
        count.index_add_(0, rows, valid.to(torch.float32))
        mean = total / count.clamp(min=1.0)
        return [float(x) for x in mean.tolist()], [int(x) for x in count.tolist()]


# --------------------------------------------------------------------------------------------
# the forward
# --------------------------------------------------------------------------------------------
def _seed():
    return int(torch.randint(0, 2**31 - 1, (1,)).item())


def _lora(x, lora):
    a, b, scaling, p = lora.parts()
    return K.LoraDelta.apply(x, a, b, scaling, p, _seed() if p > 0 else 0)


def fast_forward(lm, input_ids, attention_mask, labels, num_items_in_batch, plan):
    """Returns (loss, FusedLogits) or None when this batch has to take the normal path."""
    device = input_ids.device
    batch, seq = input_ids.shape
    mask = attention_mask.to(torch.bool)
    lens = mask.sum(1)
    arange = torch.arange(seq, device=device)
    if not bool((mask == (arange[None, :] < lens[:, None])).all()):
        return None  # left or inner padding: not handled here
    lens_list = lens.tolist()
    n = int(sum(lens_list))
    if n == 0:
        return None

    cdt = K.CDT
    offsets = torch.cumsum(lens, 0) - lens
    tok_b = torch.repeat_interleave(torch.arange(batch, device=device), lens, output_size=n)
    tok_t = (torch.arange(n, device=device) - offsets[tok_b]).to(torch.int32)
    tok_b = tok_b.to(torch.int32)
    flat = tok_b.long() * seq + tok_t.long()
    ids = input_ids.reshape(-1)[flat]
    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:].reshape(-1)[flat]
    last = tok_t.long() == (lens[tok_b.long()] - 1)
    tgt = torch.where(last, torch.full_like(shifted, -100), shifted).to(torch.int32)

    model = lm.model
    head_w = _head_weight(lm, plan)
    embed = model.embed_tokens.weight
    h = F.embedding(ids, embed).float()

    cos, sin = model.rotary_emb(h, torch.arange(seq, device=device)[None, :])
    geo = K.Geometry(tok_b.contiguous(), tok_t.contiguous(), batch, seq, cos[0].float().contiguous(),
                     sin[0].float().contiguous(), plan.n_q, plan.n_kv, plan.d)

    delta = torch.zeros(n, h.shape[1], dtype=cdt, device=device)
    for ly in plan.layers:
        s, xn = K.AddRMSNorm.apply(h, delta, ly["ln1"].weight, plan.eps)
        qm, ql = ly["q_proj"]
        vm, vl = ly["v_proj"]
        qb = frozen_linear(xn, qm)
        kb = frozen_linear(xn, ly["k_proj"][0])
        vb = frozen_linear(xn, vm)
        qd = _lora(xn, ql)
        vd = _lora(xn, vl)
        qp, kp, vp = K.AttnPrep.apply(qb, qd, kb, vb, vd, geo)
        o = F.scaled_dot_product_attention(qp, kp, vp, is_causal=True)
        a = K.GatherHeads.apply(o, geo)
        o16 = frozen_linear(a, ly["o_proj"][0])
        s, xn = K.AddRMSNorm.apply(s, o16, ly["ln2"].weight, plan.eps)
        g = frozen_linear(xn, ly["gate_proj"][0])
        u = frozen_linear(xn, ly["up_proj"][0])
        act = K.SwiGLU.apply(g, u)
        delta = frozen_linear(act, ly["down_proj"][0])
        h = s
    _, xn = K.AddRMSNorm.apply(h, delta, model.norm.weight, plan.eps)

    stats = {}
    loss_sum = K.FusedLMHeadLoss.apply(xn, head_w, tgt, stats, CHUNK)
    if num_items_in_batch is None:
        denom = (tgt >= 0).sum().clamp(min=1).to(torch.float32)
    elif torch.is_tensor(num_items_in_batch):
        denom = num_items_in_batch.to(device=device, dtype=torch.float32)
    else:
        denom = torch.tensor(float(num_items_in_batch), device=device)
    return loss_sum / denom, FusedLogits(stats, geo, tgt)


# --------------------------------------------------------------------------------------------
# installing the hooks
# --------------------------------------------------------------------------------------------
def _patch_causal_lm():
    from transformers.models.qwen2 import modeling_qwen2 as m
    from transformers.modeling_outputs import CausalLMOutputWithPast

    cls = m.Qwen2ForCausalLM
    if "forward" in _original:
        return
    _original["forward"] = cls.forward
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        eligible = (self.training and torch.is_grad_enabled() and labels is not None
                    and input_ids is not None and inputs_embeds is None
                    and attention_mask is not None and attention_mask.dim() == 2
                    and position_ids is None and past_key_values is None
                    and isinstance(logits_to_keep, int) and logits_to_keep == 0
                    and input_ids.is_cuda == next(self.parameters()).is_cuda)
        if eligible:
            plan = get_plan(self)
            if plan is not None:
                done = fast_forward(self, input_ids, attention_mask, labels,
                                    kwargs.get("num_items_in_batch"), plan)
                if done is not None:
                    loss, logits = done
                    return CausalLMOutputWithPast(loss=loss, logits=logits)
        return original(self, input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                        logits_to_keep=logits_to_keep, **kwargs)

    cls.forward = forward


def _patch_trl_entropy():
    import trl.trainer.sft_trainer as sft

    if "entropy" in _original:
        return
    _original["entropy"] = sft.entropy_from_logits
    original = sft.entropy_from_logits

    def entropy_from_logits(logits, *args, **kwargs):
        if isinstance(logits, FusedLogits):
            return logits.entropy_padded()
        return original(logits, *args, **kwargs)

    sft.entropy_from_logits = entropy_from_logits


def _patch_rewind():
    import soup_cli.trainer.rewind_hf as rewind

    if "row_losses" in _original:
        return
    _original["row_losses"] = rewind.row_losses
    original = rewind.row_losses

    def row_losses(logits, labels):
        if isinstance(logits, FusedLogits):
            return logits.row_losses()
        return original(logits, labels)

    rewind.row_losses = row_losses


def install():
    """Install every hook. Each one is independent and fails soft."""
    done = []
    for name, fn in (("causal_lm", _patch_causal_lm), ("trl_entropy", _patch_trl_entropy),
                     ("rewind", _patch_rewind)):
        try:
            fn()
            done.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[solution] hook {name} not installed: {exc!r}", flush=True)
    return done
