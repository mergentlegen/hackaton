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

One lane per packed byte; the 16-entry table and the scales are read through the cache.
"""

import torch
import triton
import triton.language as tl

OUTPUTS = 2048  # weights decoded per program (tools/nf4_bench.py: 254 us for 8960 x 1536 on a
WARPS = 4  # T4, bitsandbytes 225 us; one lane per output was 288-520 us, a select chain slower)


@triton.jit(do_not_specialize=["n_first", "n_out", "first_programs"])
def nf4_dequant_kernel(p0, a0, c0, p1, a1, c1, out_ptr, n_first, n_out, first_programs,
                       BLOCKSIZE: tl.constexpr, OUTPUTS: tl.constexpr):
    # Output rows [0, n_first) come from weight 0, the rest from weight 1 (the gate and up
    # projections decoded into one buffer for one matmul); the first `first_programs` programs
    # decode weight 0. One lane per byte: its two weights are written by two stores at stride 2,
    # which together cover the same cache lines.
    pid = tl.program_id(0)
    if pid < first_programs:
        packed_ptr, absmax_ptr, code_ptr = p0, a0, c0
        start, n_bytes = n_first * 0, n_first // 2  # (an int32 zero, like the other branch)
    else:
        packed_ptr, absmax_ptr, code_ptr = p1, a1, c1
        pid = pid - first_programs
        start, n_bytes = n_first, (n_out - n_first) // 2
    offs = pid.to(tl.int64) * (OUTPUTS // 2) + tl.arange(0, OUTPUTS // 2)
    mask = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    scale = tl.load(absmax_ptr + offs // (BLOCKSIZE // 2), mask=mask, other=0.0)
    first = tl.load(code_ptr + (byte >> 4)) * scale  # high nibble: the first weight
    second = tl.load(code_ptr + (byte & 15)) * scale
    out = out_ptr + start + 2 * offs
    tl.store(out, first.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out + 1, second.to(out_ptr.dtype.element_ty), mask=mask)


def _numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def can_dequantize(packed, state):
    absmax = getattr(state, "absmax", None)
    code = getattr(state, "code", None)
    blocksize = getattr(state, "blocksize", 0)
    return (packed.is_cuda and packed.dtype == torch.uint8
            and getattr(state, "quant_type", None) == "nf4" and not getattr(state, "nested", True)
            and isinstance(absmax, torch.Tensor) and absmax.dtype == torch.float32
            and absmax.is_cuda and isinstance(code, torch.Tensor) and code.numel() == 16
            and blocksize % 2 == 0 and blocksize > 0 and packed.numel() * 2 == _numel(state.shape)
            and getattr(state, "dtype", None) in (torch.float16, torch.bfloat16, torch.float32))


def _code(state, device):
    code = state.code
    if code.device != device or code.dtype != torch.float32:
        code = code.to(device=device, dtype=torch.float32)
    return code


def dequantize_nf4(packed, state, second=None):
    """The weight `state.shape` in `state.dtype` from its packed NF4 bytes; with `second` =
    (packed, state) of a weight with the same number of columns, both stacked by rows."""
    device = packed.device
    pb, sb = second if second is not None else (packed, state)
    rows = state.shape[0] + (sb.shape[0] if second is not None else 0)
    out = torch.empty(rows, state.shape[1], dtype=state.dtype, device=device)
    n_first = _numel(state.shape)
    first_programs = triton.cdiv(n_first, OUTPUTS)
    programs = first_programs + (triton.cdiv(out.numel() - n_first, OUTPUTS) if second else 0)
    nf4_dequant_kernel[(programs,)](
        packed, state.absmax, _code(state, device), pb, sb.absmax, _code(sb, device), out,
        n_first, out.numel(), first_programs, BLOCKSIZE=state.blocksize, OUTPUTS=OUTPUTS,
        num_warps=WARPS)
    return out


def can_stack(pa, sa, pb, sb):
    return (can_dequantize(pa, sa) and can_dequantize(pb, sb) and sa.shape[1] == sb.shape[1]
            and sa.dtype == sb.dtype and sa.blocksize == sb.blocksize
            and _numel(sa.shape) % sa.blocksize == 0)


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


class NF4StackedLinear(torch.autograd.Function):
    """x @ [Wa; Wb]^T for two frozen NF4 weights with the same input (Qwen2's gate and up
    projections): both decoded into one buffer by one launch, one matmul with twice the output
    columns, and in the backward one matmul gives d x = [d_a | d_b] @ [Wa; Wb] (the two products
    summed inside the matmul instead of an extra addition)."""

    @staticmethod
    def forward(ctx, x, pa, pb, sa, sb):
        weight = dequantize_nf4(pa, sa, (pb, sb))
        ctx.parts = (pa, sa, pb, sb)
        return torch.nn.functional.linear(x, weight.to(x.dtype))

    @staticmethod
    def backward(ctx, grad_output):
        pa, sa, pb, sb = ctx.parts
        weight = dequantize_nf4(pa, sa, (pb, sb))
        return torch.matmul(grad_output, weight.to(grad_output.dtype)), None, None, None, None


def nf4_stacked_linear(x, pa, sa, pb, sb):
    return NF4StackedLinear.apply(x, pa.reshape(-1), pb.reshape(-1), sa, sb)
