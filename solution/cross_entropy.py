"""Fused Chunked Head + 2-Pass Online-Softmax Cross-Entropy, Entropy & Accuracy in Triton."""

from __future__ import annotations

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

IGNORE_INDEX = -100
DEFAULT_CHUNK_TOKENS = 192

# Unified per-device reusable workspace shared by chunked LM head AND stacked NF4 MLP/QKV dequant (~55.6 MiB)
_SHARED_WORKSPACE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def get_shared_workspace(numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (device, dtype)
    buf = _SHARED_WORKSPACE.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty((numel,), dtype=dtype, device=device)
        _SHARED_WORKSPACE[key] = buf
    return buf[:numel]


def _get_logits_workspace(m: int, v: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return get_shared_workspace(m * v, dtype, device).view(m, v)


@triton.jit(do_not_specialize=["with_grad"])
def triton_fused_chunk_loss_kernel(
    logits_ptr,
    targets_ptr,
    loss_out_ptr,
    entropy_out_ptr,
    hits_out_ptr,
    stride_logits_m,
    vocab_size,
    with_grad,
    BLOCK_V: tl.constexpr,
):
    row_idx = tl.program_id(0)
    target = tl.load(targets_ptr + row_idx)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_m

    # Pass 1: Single-pass online softmax + entropy accumulator + argmax
    m = -float("inf")
    d = 0.0
    w = 0.0
    best_val = -float("inf")
    best_idx = 0

    for start_v in range(0, vocab_size, BLOCK_V):
        cols = start_v + tl.arange(0, BLOCK_V)
        mask = cols < vocab_size
        val = tl.load(logits_row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)

        block_max = tl.max(val, axis=0)
        m_new = tl.maximum(m, block_max)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_new))
        exp_val = tl.where(mask, tl.exp(val - m_new), 0.0)
        val_safe = tl.where(mask, val, 0.0)

        d = d * alpha + tl.sum(exp_val, axis=0)
        w = w * alpha + tl.sum(exp_val * val_safe, axis=0)
        m = m_new

        if block_max > best_val:
            best_val = block_max
            best_idx = start_v + tl.argmax(val, axis=0)

    log_d = tl.log(d)
    inv_d = 1.0 / d
    entropy = log_d + m - w * inv_d
    tl.store(entropy_out_ptr + row_idx, entropy)

    is_labelled = (target != -100) & (target >= 0) & (target < vocab_size)
    if is_labelled:
        target_logit = tl.load(logits_row_ptr + target).to(tl.float32)
        token_loss = log_d + m - target_logit
        hit = 1 if best_idx == target else 0
    else:
        token_loss = 0.0
        hit = 0

    tl.store(loss_out_ptr + row_idx, token_loss)
    tl.store(hits_out_ptr + row_idx, hit)

    # Pass 2 (only when gradients are needed): write unscaled dlogits = (p - 1_{target})
    # in-place into logits_row_ptr so values stay in [-1, 1] with zero FP16 underflow!
    if with_grad != 0:
        out_ty = logits_ptr.dtype.element_ty
        if is_labelled:
            for start_v in range(0, vocab_size, BLOCK_V):
                cols = start_v + tl.arange(0, BLOCK_V)
                mask = cols < vocab_size
                val = tl.load(logits_row_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
                p = tl.exp(val - m) * inv_d
                grad = p - tl.where(cols == target, 1.0, 0.0)
                tl.store(logits_row_ptr + cols, grad.to(out_ty), mask=mask)
        else:
            z = tl.zeros((BLOCK_V,), dtype=out_ty)
            for start_v in range(0, vocab_size, BLOCK_V):
                cols = start_v + tl.arange(0, BLOCK_V)
                mask = cols < vocab_size
                tl.store(logits_row_ptr + cols, z, mask=mask)


class _FusedChunkedHeadLossImpl(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_2d: torch.Tensor,
        weight: torch.Tensor,
        target_1d: torch.Tensor,
        chunk_tokens: int,
        denom,
        need_hidden_grad: bool,
        need_weight_grad: bool,
    ):
        n_tokens, d = hidden_2d.shape
        v = weight.shape[0]
        device = hidden_2d.device

        # Choose GEMM dtype matching autocast / hidden / weight without mutating parameters
        if torch.is_autocast_enabled("cuda"):
            gemm_dtype = torch.get_autocast_dtype("cuda")
        elif hidden_2d.dtype in (torch.float16, torch.bfloat16) and weight.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            gemm_dtype = hidden_2d.dtype
        elif hidden_2d.dtype == weight.dtype:
            gemm_dtype = hidden_2d.dtype
        else:
            gemm_dtype = torch.float16 if hidden_2d.dtype == torch.float16 else torch.float32

        w_gemm = weight if weight.dtype == gemm_dtype else weight.to(gemm_dtype)
        h_gemm = hidden_2d if hidden_2d.dtype == gemm_dtype else hidden_2d.to(gemm_dtype)

        with_grad = 1 if (need_hidden_grad or need_weight_grad) else 0
        loss_all = torch.empty((n_tokens,), dtype=torch.float32, device=device)
        entropy_all = torch.empty((n_tokens,), dtype=torch.float32, device=device)
        hits_all = torch.empty((n_tokens,), dtype=torch.int64, device=device)

        hidden_grad_buf = (
            torch.empty((n_tokens, d), dtype=gemm_dtype, device=device)
            if need_hidden_grad
            else None
        )
        weight_grad_fp32 = (
            torch.zeros((v, d), dtype=torch.float32, device=device)
            if need_weight_grad
            else None
        )

        BLOCK_V = 2048
        num_warps = 8

        if n_tokens > 0:
            max_m = min(chunk_tokens, n_tokens)
            logits_ws = _get_logits_workspace(max_m, v, gemm_dtype, device)

            for start in range(0, n_tokens, chunk_tokens):
                stop = min(start + chunk_tokens, n_tokens)
                m = stop - start
                piece = h_gemm[start:stop]
                t_piece = target_1d[start:stop]
                logits = logits_ws[:m]

                torch.matmul(piece, w_gemm.t(), out=logits)

                triton_fused_chunk_loss_kernel[(m,)](
                    logits,
                    t_piece,
                    loss_all[start:stop],
                    entropy_all[start:stop],
                    hits_all[start:stop],
                    logits.stride(0),
                    v,
                    with_grad,
                    BLOCK_V=BLOCK_V,
                    num_warps=num_warps,
                )

                if need_hidden_grad:
                    # logits holds unscaled dlogits in [-1, 1]; write matmul directly into slice
                    torch.matmul(logits, w_gemm, out=hidden_grad_buf[start:stop])
                if need_weight_grad:
                    weight_grad_fp32.addmm_(logits.t().float(), piece.float())

        if torch.is_tensor(denom):
            denom_f32 = denom.to(device=device, dtype=torch.float32).clamp(min=1.0)
        else:
            denom_f32 = torch.tensor(max(1.0, float(denom)), dtype=torch.float32, device=device)

        loss = loss_all.sum() / denom_f32
        total_hits = hits_all.sum()

        saved = []
        if need_hidden_grad:
            saved.append(hidden_grad_buf)
        if need_weight_grad:
            saved.append(weight_grad_fp32)
        saved.append(denom_f32)
        ctx.save_for_backward(*saved)
        ctx.need_hidden_grad = need_hidden_grad
        ctx.need_weight_grad = need_weight_grad
        ctx.hidden_dtype = hidden_2d.dtype
        ctx.weight_dtype = weight.dtype
        ctx.set_materialize_grads(False)
        return loss, entropy_all, loss_all, total_hits

    @staticmethod
    def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_hits):
        if grad_loss is None or (not ctx.need_hidden_grad and not ctx.need_weight_grad):
            return None, None, None, None, None, None, None

        saved = ctx.saved_tensors
        denom_f32 = saved[-1]
        scale = grad_loss.float() / denom_f32

        idx = 0
        d_hidden = None
        d_weight = None
        if ctx.need_hidden_grad:
            hg = saved[idx]
            idx += 1
            d_hidden = (hg.float() * scale).to(ctx.hidden_dtype)
        if ctx.need_weight_grad:
            wg = saved[idx]
            d_weight = (wg * scale).to(ctx.weight_dtype)

        return d_hidden, d_weight, None, None, None, None, None


class FusedChunkedHeadLossFunc:
    """Public wrapper matching `FusedChunkedHeadLossFunc.apply(hidden, weight, target, chunk_size, denom)`."""

    @staticmethod
    def apply(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        chunk_size: int = DEFAULT_CHUNK_TOKENS,
        denom=None,
    ):
        orig_shape = hidden.shape[:-1]
        d = hidden.shape[-1]
        hidden_2d = hidden.reshape(-1, d).contiguous()
        target_1d = target.reshape(-1).contiguous()

        # Interpret small chunk_size (e.g. sequence chunk 32 or 64 on 3D input) vs token chunk
        if hidden.ndim == 3 and chunk_size <= hidden.shape[1]:
            chunk_tokens = max(DEFAULT_CHUNK_TOKENS, chunk_size * hidden.shape[0])
        else:
            chunk_tokens = max(64, int(chunk_size))

        if denom is None:
            denom = (target_1d != IGNORE_INDEX).sum().clamp(min=1)

        grad_enabled = torch.is_grad_enabled()
        need_h = grad_enabled and hidden.requires_grad
        need_w = grad_enabled and weight.requires_grad

        loss, entropy_1d, loss_1d, total_hits = _FusedChunkedHeadLossImpl.apply(
            hidden_2d,
            weight,
            target_1d,
            chunk_tokens,
            denom,
            need_h,
            need_w,
        )
        return (
            loss,
            entropy_1d.reshape(orig_shape),
            loss_1d.reshape(orig_shape),
            total_hits,
        )


def shift_for_next_token(labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    return F.pad(labels, (0, 1), value=ignore_index)[..., 1:]


class TritonChunkedCausalLMLoss:
    """Replaces full [B, S, V] logits materialization with fused Triton chunked head loss
    plus optional in-batch valid-token compaction."""

    def __init__(self, causal_lm: torch.nn.Module, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> None:
        self.causal_lm = causal_lm
        self.chunk_tokens = chunk_tokens
        self.head = causal_lm.get_output_embeddings()
        self.last_step = None
        self._holding = False
        self._held = None
        self._before = [
            (module, module.__dict__.get("forward"))
            for module in (self.head, causal_lm)
        ]

        self.head.forward = self._head_forward
        causal_lm.forward = self._causal_lm_forward

    def _head_forward(self, hidden, *args, **kwargs):
        if not self._holding:
            orig_head = self._before[0][1]
            if orig_head is not None:
                return orig_head(hidden, *args, **kwargs)
            return type(self.head).forward(self.head, hidden, *args, **kwargs)
        placeholder = hidden.new_empty((*hidden.shape[:-1], 0))
        self._held = (hidden, placeholder)
        return placeholder

    def _call_orig_causal_lm(self, *args, **kwargs):
        orig_fwd = self._before[1][1]
        if orig_fwd is not None:
            return orig_fwd(*args, **kwargs)
        return type(self.causal_lm).forward(self.causal_lm, *args, **kwargs)

    def _causal_lm_forward(self, *args, **kwargs):
        if kwargs.get("labels") is None:
            return self._call_orig_causal_lm(*args, **kwargs)
        labels = kwargs.pop("labels")

        orig_b, orig_s = labels.shape[0], labels.shape[1]
        shift_labels = kwargs.get("shift_labels")
        target_full = (
            shift_labels
            if shift_labels is not None
            else shift_for_next_token(labels, IGNORE_INDEX)
        )

        # During loss computation (labels is not None), avoid allocating a dead 28-layer DynamicCache
        # when the caller did not explicitly request `use_cache=True` or pass `past_key_values`.
        explicit_use_cache = bool(kwargs.get("use_cache"))
        if kwargs.get("past_key_values") is None and not explicit_use_cache:
            kwargs = dict(kwargs)
            kwargs["use_cache"] = False

        # Check if we can compact valid tokens across the batch (unpadding)
        pack_ctx = None
        input_ids = kwargs.get("input_ids")
        attention_mask = kwargs.get("attention_mask")
        if (
            input_ids is not None
            and attention_mask is not None
            and shift_labels is None
            and kwargs.get("inputs_embeds") is None
            and kwargs.get("past_key_values") is None
            and not explicit_use_cache
            and not kwargs.get("output_hidden_states", False)
            and not kwargs.get("output_attentions", False)
            and attention_mask.ndim == 2
            and input_ids.is_cuda
        ):
            from .padding import maybe_build_packed_batch, packed_context_scope

            pack_ctx = maybe_build_packed_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=kwargs.get("position_ids"),
            )
            if pack_ctx is not None:
                kwargs = dict(kwargs)
                kwargs["input_ids"] = pack_ctx.packed_input_ids
                kwargs["position_ids"] = pack_ctx.packed_position_ids
                kwargs["attention_mask"] = None

        self._holding = True
        self._held = None
        try:
            if pack_ctx is not None:
                from .padding import packed_context_scope

                with packed_context_scope(pack_ctx):
                    outputs = self._call_orig_causal_lm(*args, **kwargs)
            else:
                outputs = self._call_orig_causal_lm(*args, **kwargs)
        finally:
            self._holding = False

        held, self._held = self._held, None
        if held is None or getattr(outputs, "logits", None) is not held[1]:
            return outputs

        hidden = held[0]
        target_full = target_full.to(hidden.device)
        labelled = (target_full != IGNORE_INDEX).sum()

        num_items = kwargs.get("num_items_in_batch")
        if num_items is None:
            denom = labelled.clamp(min=1)
        elif torch.is_tensor(num_items):
            denom = num_items.to(hidden.device).clamp(min=1)
        else:
            denom = max(1, int(num_items))

        if pack_ctx is not None and pack_ctx.valid_idx is not None:
            # hidden is (1, N_valid, D); gather corresponding shifted targets
            target_packed = target_full.reshape(-1).index_select(0, pack_ctx.valid_idx).unsqueeze(0)
            loss, entropy_packed, loss_packed, hits = FusedChunkedHeadLossFunc.apply(
                hidden,
                self.head.weight,
                target_packed,
                self.chunk_tokens,
                denom,
            )
            # Scatter 1D metrics back to (orig_b, orig_s) (only 8 KiB!) for TRL & Rewind parity
            token_entropy = torch.zeros(
                (orig_b * orig_s,), dtype=torch.float32, device=hidden.device
            )
            token_loss = torch.zeros(
                (orig_b * orig_s,), dtype=torch.float32, device=hidden.device
            )
            token_entropy.index_copy_(0, pack_ctx.valid_idx, entropy_packed.reshape(-1))
            token_loss.index_copy_(0, pack_ctx.valid_idx, loss_packed.reshape(-1))
            token_entropy = token_entropy.view(orig_b, orig_s)
            token_loss = token_loss.view(orig_b, orig_s)
        else:
            loss, token_entropy, token_loss, hits = FusedChunkedHeadLossFunc.apply(
                hidden,
                self.head.weight,
                target_full,
                self.chunk_tokens,
                denom,
            )

        try:
            from soup_cli.utils.stream_loss_chunk import ChunkedStep

            self.last_step = ChunkedStep(
                token_entropy=token_entropy,
                token_loss=token_loss,
                correct=hits,
                labelled=labelled,
            )
        except ImportError:
            self.last_step = None

        rest = {k: v for k, v in outputs.items() if k not in ("loss", "logits")}
        return type(outputs)(loss=loss, **rest)

    def take_step(self):
        step, self.last_step = self.last_step, None
        return step

    def remove(self) -> None:
        for module, forward in self._before:
            if forward is None:
                module.__dict__.pop("forward", None)
            else:
                module.forward = forward


def patch_soup_trainer() -> None:
    """Hook TritonChunkedCausalLMLoss and FP16 embedding preservation into Soup."""
    # 1. Prevent peft.prepare_model_for_kbit_training from upcasting frozen [151936, 1536]
    #    embed_tokens / lm_head weights to FP32 (+445 MiB permanent VRAM + 890 MiB load spike
    #    + FP32 residual infection across all 28 layers). Keep 1D norms in FP32.
    try:
        import peft
        import peft.utils
        import peft.utils.other as peft_other

        if not getattr(peft_other, "_soup_kbit_patched", False):
            orig_prep = peft_other.prepare_model_for_kbit_training

            def _fast_prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs=None,
                auto_clear_cache=True,
            ):
                if not isinstance(model, torch.nn.Module):
                    return orig_prep(
                        model,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                        auto_clear_cache=auto_clear_cache,
                    )
                # Temporarily mark large 2D+ frozen fp16/bf16 weights (embed_tokens / lm_head)
                # so orig_prep only upcasts 1D LayerNorm/RMSNorm weights to fp32
                saved_2d: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
                for param in model.parameters():
                    if (
                        param.ndim >= 2
                        and param.dtype in (torch.float16, torch.bfloat16)
                        and param.__class__.__name__ != "Params4bit"
                    ):
                        saved_2d.append((param, param.data))

                if not saved_2d:
                    return orig_prep(
                        model,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                        auto_clear_cache=auto_clear_cache,
                    )

                # Replace param.data with a tiny 0-element fp32 stub during orig_prep
                # so orig_prep NEVER allocates an 890 MiB fp32 copy of embed_tokens!
                stub = torch.empty((0,), dtype=torch.float32, device=saved_2d[0][1].device)
                try:
                    for param, _ in saved_2d:
                        param.data = stub
                    res = orig_prep(
                        model,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                        auto_clear_cache=False,
                    )
                finally:
                    for param, orig_data in saved_2d:
                        param.data = orig_data
                return res

            peft_other.prepare_model_for_kbit_training = _fast_prepare_model_for_kbit_training
            peft.prepare_model_for_kbit_training = _fast_prepare_model_for_kbit_training
            if hasattr(peft.utils, "prepare_model_for_kbit_training"):
                peft.utils.prepare_model_for_kbit_training = _fast_prepare_model_for_kbit_training
            peft_other._soup_kbit_patched = True
    except ImportError:
        pass

    # 2. Attach TritonChunkedCausalLMLoss & decoder checkpointing in SFTTrainerWrapper._setup_transformers
    try:
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from soup_cli.utils.stream_loss_chunk import causal_lm_of
    except ImportError:
        return

    if getattr(SFTTrainerWrapper, "_soup_triton_ce_patched", False):
        return

    original_setup = SFTTrainerWrapper._setup_transformers

    def _setup_transformers(self, cfg, tcfg):
        original_setup(self, cfg, tcfg)
        if self.model is not None and isinstance(self.model, torch.nn.Module):
            causal_lm = causal_lm_of(self.model)
            if hasattr(causal_lm, "get_output_embeddings") and callable(
                causal_lm.get_output_embeddings
            ):
                head = causal_lm.get_output_embeddings()
                if (
                    head is not None
                    and hasattr(head, "weight")
                    and isinstance(head.weight, torch.Tensor)
                    and head.weight.is_cuda
                ):
                    from .padding import install_model_layer_optimizations

                    install_model_layer_optimizations(causal_lm)
                    hook = TritonChunkedCausalLMLoss(
                        causal_lm, chunk_tokens=DEFAULT_CHUNK_TOKENS
                    )
                    self._stream_loss_chunk_hook = hook

    SFTTrainerWrapper._setup_transformers = _setup_transformers
    SFTTrainerWrapper._soup_triton_ce_patched = True
