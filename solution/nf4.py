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

One lane per 32-bit word of packed bytes (8 weights, one vector store); the 16-entry table and
the scales are read through the cache.
"""

import torch
import triton
import triton.language as tl

WORDS = 256  # 32-bit words (8 weights each) per program; tools/nf4_variants.py on a T4,
WARPS = 8  # 8960 x 1536: 165 us (208 GB/s), bitsandbytes 209 us, one lane per byte 254 us


@triton.jit(do_not_specialize=["n_out", "_i1", "_i2", "_i3", "_i4"])
def nf4_dequant_kernel(packed_ptr, absmax_ptr, code_ptr, out_ptr, _p4, _p5, _p6, _p7, _p8, _p9,
                       n_out, _i1, _i2, _i3, _i4, _f, BLOCKSIZE: tl.constexpr,
                       WORDS: tl.constexpr, _c2: tl.constexpr, _c3: tl.constexpr):
    # (the argument layout shared by all our kernels: see kernels.py)
    # One lane per 32-bit word of packed bytes: 8 weights, stored as 8 contiguous values (one
    # vector store). Byte k of a (little-endian) word holds weights 2k (high nibble) and 2k + 1.
    w = tl.program_id(0).to(tl.int64) * WORDS + tl.arange(0, WORDS)
    mask = w < n_out // 8
    word = tl.load(packed_ptr.to(tl.pointer_type(tl.int32)) + w, mask=mask, other=0)
    scale = tl.load(absmax_ptr + w // (BLOCKSIZE // 8), mask=mask, other=0.0)
    k = tl.arange(0, 8)
    shift = (k // 2) * 8 + (1 - k % 2) * 4
    nibble = (word[:, None] >> shift[None, :]) & 15
    value = tl.load(code_ptr + nibble) * scale[:, None]
    at = w[:, None] * 8 + k[None, :]
    tl.store(out_ptr + at, value.to(out_ptr.dtype.element_ty), mask=mask[:, None])


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
            and blocksize % 8 == 0 and packed.numel() * 2 == _numel(state.shape)
            and _numel(state.shape) % 8 == 0 and packed.data_ptr() % 4 == 0
            and getattr(state, "dtype", None) in (torch.float16, torch.bfloat16, torch.float32))


_DEQUANT_BUF = None


def _get_dequant_buf(shape, dtype, device):
    global _DEQUANT_BUF
    numel = _numel(shape)
    if (_DEQUANT_BUF is None or _DEQUANT_BUF.numel() < numel
            or _DEQUANT_BUF.device != device or _DEQUANT_BUF.dtype != dtype):
        _DEQUANT_BUF = torch.empty(max(numel, 13762560), dtype=dtype, device=device)
    return _DEQUANT_BUF[:numel].view(shape)


def dequantize_nf4(packed, state, out=None):
    """The weight `state.shape` in `state.dtype` from its packed NF4 bytes."""
    if out is None:
        out = torch.empty(state.shape, dtype=state.dtype, device=packed.device)
    n_out = out.numel()
    code = state.code
    if code.device != packed.device or code.dtype != torch.float32:
        code = code.to(device=packed.device, dtype=torch.float32)
    nf4_dequant_kernel[(triton.cdiv(n_out // 8, WORDS),)](
        packed, state.absmax, code, out, *[out] * 6, n_out, 0, 0, 0, 0, 0.0,
        BLOCKSIZE=state.blocksize, WORDS=WORDS, _c2=0, _c3=0, num_warps=WARPS)
    return out


class NF4Linear(torch.autograd.Function):
    """x @ W^T + bias for a frozen NF4 weight W: decode W (our kernel), then the fp16 matmul.

    The same two steps bitsandbytes' MatMul4Bit takes at these sizes (its dequantize + linear
    path forward, dequantize + matmul backward); the weight is frozen, so the backward is only
    the input gradient, and nothing but the packed bytes (a parameter) is kept for it."""

    @staticmethod
    def forward(ctx, x, packed, bias, state):
        buf = _get_dequant_buf(state.shape, state.dtype, packed.device)
        weight = dequantize_nf4(packed, state, out=buf)
        ctx.packed, ctx.state = packed, state
        ctx.bias_dtype = None if bias is None else bias.dtype
        return torch.nn.functional.linear(x, weight.to(x.dtype), bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            buf = _get_dequant_buf(ctx.state.shape, ctx.state.dtype, ctx.packed.device)
            weight = dequantize_nf4(ctx.packed, ctx.state, out=buf)
            grad_x = torch.matmul(grad_output, weight.to(grad_output.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0, dtype=ctx.bias_dtype)
        return grad_x, None, grad_bias, None


def nf4_linear(x, packed, bias, state):
    return NF4Linear.apply(x, packed.reshape(-1), bias, state)
