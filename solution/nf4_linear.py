"""Fused NF4/FP4 Double-Quant Dequantization in Triton + Tensor-Core Linear & LoRA."""

from __future__ import annotations

import torch
from torch.nn import functional as F
import triton
import triton.language as tl

# Per-device reusable workspace buffer for dequantized [N, K] weights (~26.25 MiB max for Qwen2-1.5B)
_DEQUANT_WORKSPACE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_dequant_workspace(numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (device, dtype)
    buf = _DEQUANT_WORKSPACE.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty((numel,), dtype=dtype, device=device)
        _DEQUANT_WORKSPACE[key] = buf
    return buf[:numel]


@triton.jit
def triton_nf4_dequant_kernel(
    qweight_ptr,
    absmax_ptr,
    s2_absmax_ptr,
    s2_code_ptr,
    offset_ptr,
    code_ptr,
    out_ptr,
    n_bytes,
    n_elements,
    HALF_BLOCKSIZE: tl.constexpr,
    S2_BLOCKSIZE: tl.constexpr,
    IS_NESTED: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    pid = tl.program_id(0)
    byte_offs = pid * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    byte_mask = byte_offs < n_bytes

    packed = tl.load(qweight_ptr + byte_offs, mask=byte_mask, other=0).to(tl.int32)
    hi_idx = (packed >> 4) & 0x0F
    lo_idx = packed & 0x0F

    blk_idx = byte_offs // HALF_BLOCKSIZE
    if IS_NESTED:
        qabs = tl.load(absmax_ptr + blk_idx, mask=byte_mask, other=0).to(tl.int32)
        s2_idx = blk_idx // S2_BLOCKSIZE
        s2_abs = tl.load(s2_absmax_ptr + s2_idx, mask=byte_mask, other=0.0).to(tl.float32)
        s2_val = tl.load(s2_code_ptr + qabs, mask=byte_mask, other=0.0).to(tl.float32)
        off_val = tl.load(offset_ptr).to(tl.float32)
        scale = s2_val * s2_abs + off_val
    else:
        scale = tl.load(absmax_ptr + blk_idx, mask=byte_mask, other=0.0).to(tl.float32)

    val_hi = tl.load(code_ptr + hi_idx) * scale
    val_lo = tl.load(code_ptr + lo_idx) * scale

    pair = tl.join(val_hi, val_lo)
    vals = tl.reshape(pair, (BLOCK_BYTES * 2,))

    out_offs = pid * (BLOCK_BYTES * 2) + tl.arange(0, BLOCK_BYTES * 2)
    out_mask = out_offs < n_elements
    tl.store(out_ptr + out_offs, vals.to(out_ptr.dtype.element_ty), mask=out_mask)


def can_triton_dequant(qweight: torch.Tensor, quant_state) -> bool:
    if quant_state is None or not qweight.is_cuda or qweight.dtype != torch.uint8:
        return False
    if quant_state.quant_type not in ("nf4", "fp4"):
        return False
    if quant_state.blocksize is None or quant_state.blocksize < 2 or (quant_state.blocksize % 2 != 0):
        return False
    if quant_state.code is None:
        return False
    if quant_state.nested:
        s2 = quant_state.state2
        if s2 is None or s2.absmax is None or s2.code is None or quant_state.offset is None:
            return False
    return True


def dequantize_nf4_triton(
    qweight: torch.Tensor,
    quant_state,
    dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize a bitsandbytes 4-bit (NF4/FP4, single or double-quant) tensor in Triton."""
    target_dtype = dtype if dtype is not None else quant_state.dtype
    shape = quant_state.shape
    n_elements = int(shape.numel())
    q_flat = qweight.reshape(-1)
    n_bytes = q_flat.numel()

    if out is None:
        out_flat = torch.empty((n_elements,), dtype=target_dtype, device=qweight.device)
    else:
        out_flat = out.reshape(-1)[:n_elements]

    BLOCK_BYTES = 512
    grid = (triton.cdiv(n_bytes, BLOCK_BYTES),)
    half_bs = int(quant_state.blocksize) // 2

    if quant_state.nested:
        s2 = quant_state.state2
        triton_nf4_dequant_kernel[grid](
            q_flat,
            quant_state.absmax,
            s2.absmax,
            s2.code,
            quant_state.offset,
            quant_state.code,
            out_flat,
            n_bytes,
            n_elements,
            HALF_BLOCKSIZE=half_bs,
            S2_BLOCKSIZE=int(s2.blocksize),
            IS_NESTED=True,
            BLOCK_BYTES=BLOCK_BYTES,
            num_warps=4,
        )
    else:
        dummy = quant_state.code
        triton_nf4_dequant_kernel[grid](
            q_flat,
            quant_state.absmax,
            dummy,
            dummy,
            dummy,
            quant_state.code,
            out_flat,
            n_bytes,
            n_elements,
            HALF_BLOCKSIZE=half_bs,
            S2_BLOCKSIZE=256,
            IS_NESTED=False,
            BLOCK_BYTES=BLOCK_BYTES,
            num_warps=4,
        )

    out_2d = out_flat.view(shape)
    if qweight.ndim == 2 and qweight.shape[0] == 1 and qweight.shape[1] > 1:
        return out_2d.t()
    return out_2d


def _dequant_to_workspace(qweight: torch.Tensor, quant_state, dtype: torch.dtype) -> torch.Tensor:
    numel = int(quant_state.shape.numel())
    ws = _get_dequant_workspace(numel, dtype, qweight.device)
    return dequantize_nf4_triton(qweight.view(-1, 1), quant_state, dtype=dtype, out=ws)


class TritonNF4LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        qweight: torch.Tensor,
        bias: torch.Tensor | None,
        quant_state,
        compute_dtype: torch.dtype,
    ):
        orig_x_dtype = x.dtype
        if x.numel() == 0:
            ctx.is_empty = True
            ctx.orig_shape = x.shape
            ctx.orig_x_dtype = orig_x_dtype
            ctx.bias_dtype = bias.dtype if bias is not None else None
            out_features = quant_state.shape[0]
            return torch.empty((*x.shape[:-1], out_features), dtype=orig_x_dtype, device=x.device)

        ctx.is_empty = False
        x_comp = x if orig_x_dtype == compute_dtype else x.to(compute_dtype)
        b_comp = (
            None
            if bias is None
            else (bias if bias.dtype == compute_dtype else bias.to(compute_dtype))
        )

        w_dq = _dequant_to_workspace(qweight, quant_state, compute_dtype)
        out = F.linear(x_comp, w_dq, b_comp)
        if out.dtype != orig_x_dtype:
            out = out.to(orig_x_dtype)

        ctx.qweight = qweight
        ctx.quant_state = quant_state
        ctx.compute_dtype = compute_dtype
        ctx.orig_x_dtype = orig_x_dtype
        ctx.bias_dtype = bias.dtype if (bias is not None and bias.requires_grad) else None
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor | None):
        if grad_output is None:
            return None, None, None, None, None
        if ctx.is_empty:
            dx = torch.zeros(ctx.orig_shape, dtype=ctx.orig_x_dtype, device=grad_output.device)
            db = (
                torch.zeros((grad_output.shape[-1],), dtype=ctx.bias_dtype, device=grad_output.device)
                if ctx.needs_input_grad[2] and ctx.bias_dtype is not None
                else None
            )
            return dx, None, db, None, None

        grad_bias = None
        if ctx.needs_input_grad[2] and ctx.bias_dtype is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0, dtype=ctx.bias_dtype)

        grad_x = None
        if ctx.needs_input_grad[0]:
            compute_dtype = ctx.compute_dtype
            g_comp = (
                grad_output
                if grad_output.dtype == compute_dtype
                else grad_output.to(compute_dtype)
            )
            w_dq = _dequant_to_workspace(ctx.qweight, ctx.quant_state, compute_dtype)
            g_2d = g_comp.reshape(-1, g_comp.shape[-1])
            grad_x = torch.matmul(g_2d, w_dq).view(*g_comp.shape[:-1], w_dq.shape[1])
            if grad_x.dtype != ctx.orig_x_dtype:
                grad_x = grad_x.to(ctx.orig_x_dtype)

        return grad_x, None, grad_bias, None, None


class TritonNF4SwiGLUMLPFunc(torch.autograd.Function):
    """Fused MLP (gate_proj + up_proj + Triton SwiGLU + down_proj) with single-buffer NF4 dequant
    and in-place backward dx accumulation (`addmm_`)."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_qw: torch.Tensor,
        gate_qs,
        up_qw: torch.Tensor,
        up_qs,
        down_qw: torch.Tensor,
        down_qs,
        compute_dtype: torch.dtype,
    ):
        from .swiglu import example_swiglu_fwd_kernel

        orig_shape = x.shape
        orig_dtype = x.dtype
        x_2d = (x if orig_dtype == compute_dtype else x.to(compute_dtype)).reshape(-1, orig_shape[-1]).contiguous()
        m = x_2d.shape[0]
        if m == 0:
            ctx.is_empty = True
            ctx.orig_shape = orig_shape
            ctx.orig_dtype = orig_dtype
            return torch.empty(orig_shape, dtype=orig_dtype, device=x.device)

        ctx.is_empty = False
        w_gate = _dequant_to_workspace(gate_qw, gate_qs, compute_dtype)
        gate = torch.matmul(x_2d, w_gate.t())

        w_up = _dequant_to_workspace(up_qw, up_qs, compute_dtype)
        up = torch.matmul(x_2d, w_up.t())

        act = torch.empty_like(gate)
        n = act.numel()
        BLOCK = 1024
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, act, n, BLOCK=BLOCK)

        w_down = _dequant_to_workspace(down_qw, down_qs, compute_dtype)
        out_2d = torch.matmul(act, w_down.t())

        ctx.save_for_backward(gate, up)
        ctx.weights = (gate_qw, gate_qs, up_qw, up_qs, down_qw, down_qs)
        ctx.compute_dtype = compute_dtype
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = orig_dtype
        ctx.set_materialize_grads(False)

        out = out_2d.view(orig_shape)
        return out if out.dtype == orig_dtype else out.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor | None):
        if grad_out is None:
            return None, None, None, None, None, None, None, None
        if ctx.is_empty:
            return (
                torch.zeros(ctx.orig_shape, dtype=ctx.orig_dtype, device=grad_out.device),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None, None, None

        from .swiglu import example_swiglu_bwd_kernel

        gate, up = ctx.saved_tensors
        gate_qw, gate_qs, up_qw, up_qs, down_qw, down_qs = ctx.weights
        compute_dtype = ctx.compute_dtype

        g_2d = (
            grad_out if grad_out.dtype == compute_dtype else grad_out.to(compute_dtype)
        ).reshape(-1, ctx.orig_shape[-1]).contiguous()

        w_down = _dequant_to_workspace(down_qw, down_qs, compute_dtype)
        d_act = torch.matmul(g_2d, w_down)

        d_gate = torch.empty_like(gate)
        d_up = torch.empty_like(up)
        n = d_act.numel()
        BLOCK = 1024
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](
            d_act, gate, up, d_gate, d_up, n, BLOCK=BLOCK
        )
        del d_act

        w_gate = _dequant_to_workspace(gate_qw, gate_qs, compute_dtype)
        dx_2d = torch.matmul(d_gate, w_gate)
        del d_gate

        w_up = _dequant_to_workspace(up_qw, up_qs, compute_dtype)
        dx_2d.addmm_(d_up, w_up)
        del d_up

        dx = dx_2d.view(ctx.orig_shape)
        if dx.dtype != ctx.orig_dtype:
            dx = dx.to(ctx.orig_dtype)
        return dx, None, None, None, None, None, None, None


def _resolve_bnb_compute_dtype(module, x: torch.Tensor) -> torch.dtype:
    if not getattr(module, "compute_type_is_set", False):
        module.set_compute_type(x)
        module.compute_type_is_set = True
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    if module.compute_dtype is not None:
        return module.compute_dtype
    return x.dtype


def patch_bnb_nf4_linear() -> None:
    """Patch bitsandbytes Linear4bit and PEFT lora.bnb.Linear4bit to use Triton NF4 dequant
    + cuBLAS Tensor Core GEMM + clone-free fused addmm LoRA."""
    try:
        from bitsandbytes.nn.modules import Linear4bit as BnbLinear4bit
        from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module
    except ImportError:
        return

    if not getattr(BnbLinear4bit, "_soup_triton_nf4_patched", False):
        orig_bnb_fwd = BnbLinear4bit.forward

        def _fast_bnb_linear4bit_forward(self, x: torch.Tensor):
            if not x.is_cuda:
                return orig_bnb_fwd(self, x)
            fix_4bit_weight_quant_state_from_module(self)
            qs = getattr(self.weight, "quant_state", None)
            if not can_triton_dequant(self.weight.data, qs) or len(qs.shape) != 2 or x.shape[-1] != qs.shape[1]:
                return orig_bnb_fwd(self, x)
            comp_dtype = _resolve_bnb_compute_dtype(self, x)
            if comp_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                return orig_bnb_fwd(self, x)
            return TritonNF4LinearFunc.apply(x, self.weight.data, self.bias, qs, comp_dtype)

        BnbLinear4bit.forward = _fast_bnb_linear4bit_forward
        BnbLinear4bit._soup_triton_nf4_patched = True

    # Patch PEFT's lora.bnb.Linear4bit to eliminate defensive `result.clone()` and fuse LoRA via `torch.addmm`
    try:
        from peft.tuners.lora.bnb import Linear4bit as PeftLoraLinear4bit
    except ImportError:
        return

    if not getattr(PeftLoraLinear4bit, "_soup_triton_lora_patched", False):
        orig_peft_fwd = PeftLoraLinear4bit.forward

        def _fast_peft_lora_4bit_forward(self, x: torch.Tensor, *args, **kwargs):
            if (
                not x.is_cuda
                or args
                or kwargs
                or self.disable_adapters
                or self.merged
            ):
                return orig_peft_fwd(self, x, *args, **kwargs)

            for active_adapter in self.active_adapters:
                if active_adapter in self.lora_A and active_adapter in self.lora_variant:
                    return orig_peft_fwd(self, x, *args, **kwargs)

            result = self.base_layer(x)
            out_dtype = result.dtype
            orig_shape = result.shape
            res_2d = result.reshape(-1, orig_shape[-1])
            autocast_on = torch.is_autocast_enabled("cuda")

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A:
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = float(self.scaling[active_adapter])

                x_in = x if autocast_on else self._cast_input_dtype(x, lora_A.weight.dtype)
                h = lora_A(dropout(x_in))
                h_2d = h.reshape(-1, h.shape[-1])
                if h_2d.dtype != out_dtype:
                    h_2d = h_2d.to(out_dtype)
                wb = lora_B.weight
                if wb.dtype != out_dtype:
                    wb = wb.to(out_dtype)
                if lora_B.bias is not None:
                    res_2d = res_2d + (lora_B(h).reshape(-1, orig_shape[-1]).to(out_dtype) * scaling)
                else:
                    res_2d = torch.addmm(res_2d, h_2d, wb.t(), beta=1.0, alpha=scaling)

            return res_2d.view(orig_shape)

        PeftLoraLinear4bit.forward = _fast_peft_lora_4bit_forward
        PeftLoraLinear4bit._soup_triton_lora_patched = True
