"""NF4 dequantization of the 4-bit weights, our Triton kernel instead of bitsandbytes' CUDA one.

Every 4-bit linear (196 of them) runs its matmul on an fp16 copy of its weight, decoded right
before the forward matmul and again before the backward one: 392 decodes of 1.3 B weights per
step, about 6.5% of the GPU time of a training step (tools/op_profile.py).

The format is bitsandbytes' NF4 (QLoRA, Dettmers et al. 2023, https://arxiv.org/abs/2305.14314):
each byte holds two 4-bit codes, the high nibble first; a code indexes the 16-value NF4 table
(`quant_state.code`), and every `blocksize` (64) consecutive weights share one fp32 scale
(`absmax`). weight = table[code] * absmax[i // blocksize], computed in fp32 and rounded once to
the output dtype: the same single multiplication bitsandbytes does, so the result is bit for bit
the same (solution/test_nf4.py). The scales must be plain fp32 (not double-quantized): the packed
step decodes the 8-bit scales once for good (packed.py `_flatten_quant_states`), anything else
goes to bitsandbytes.

One program decodes `BYTES` bytes (2 * BYTES weights) with coalesced loads and one interleaved
store; the 16-entry table stays in L1.
"""

import torch
import triton
import triton.language as tl

BYTES = 1024


@triton.jit(do_not_specialize=["n_bytes"])
def nf4_dequant_kernel(packed_ptr, absmax_ptr, code_ptr, out_ptr, n_bytes,
                       HALF_BLOCK: tl.constexpr, BYTES: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BYTES + tl.arange(0, BYTES)
    mask = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    scale = tl.load(absmax_ptr + offs // HALF_BLOCK, mask=mask, other=0.0)
    first = tl.load(code_ptr + (byte >> 4)) * scale
    second = tl.load(code_ptr + (byte & 15)) * scale
    values = tl.interleave(first, second).to(out_ptr.dtype.element_ty)
    out_offs = tl.interleave(2 * offs, 2 * offs + 1)
    tl.store(out_ptr + out_offs, values, mask=out_offs < 2 * n_bytes)


def can_dequantize(packed, state):
    absmax = getattr(state, "absmax", None)
    code = getattr(state, "code", None)
    blocksize = getattr(state, "blocksize", 0)
    return (packed.is_cuda and packed.dtype == torch.uint8
            and getattr(state, "quant_type", None) == "nf4" and not getattr(state, "nested", True)
            and isinstance(absmax, torch.Tensor) and absmax.dtype == torch.float32
            and absmax.is_cuda and isinstance(code, torch.Tensor) and code.numel() == 16
            and blocksize % 2 == 0 and blocksize > 0
            and getattr(state, "dtype", None) in (torch.float16, torch.bfloat16, torch.float32))


def dequantize_nf4(packed, state):
    """The weight `state.shape` in `state.dtype` from its packed NF4 bytes."""
    out = torch.empty(state.shape, dtype=state.dtype, device=packed.device)
    n_bytes = packed.numel()
    code = state.code
    if code.device != packed.device or code.dtype != torch.float32:
        code = code.to(device=packed.device, dtype=torch.float32)
    nf4_dequant_kernel[(triton.cdiv(n_bytes, BYTES),)](
        packed, state.absmax, code, out, n_bytes, HALF_BLOCK=state.blocksize // 2, BYTES=BYTES,
        num_warps=4)
    return out


class NF4Linear(torch.autograd.Function):
    """x @ W^T + bias for a frozen NF4 weight W: decode W (our kernel), then the fp16 matmul.

    The same two steps bitsandbytes' MatMul4Bit takes at these sizes (its dequantize + linear
    path forward, dequantize + matmul backward); the weight is frozen, so the backward is only
    the input gradient, and nothing but the packed bytes (a parameter) is kept for it."""

    @staticmethod
    def forward(ctx, x, packed, bias, state):
        weight = dequantize_nf4(packed, state)
        ctx.packed, ctx.state = packed, state
        ctx.bias_dtype = None if bias is None else bias.dtype
        return torch.nn.functional.linear(x, weight.to(x.dtype), bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            weight = dequantize_nf4(ctx.packed, ctx.state)
            grad_x = torch.matmul(grad_output, weight.to(grad_output.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0, dtype=ctx.bias_dtype)
        return grad_x, None, grad_bias, None


def nf4_linear(x, packed, bias, state):
    return NF4Linear.apply(x, packed.reshape(-1), bias, state)
