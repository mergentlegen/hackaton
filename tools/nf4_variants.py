"""Analysis only: NF4 decode layouts compared (bitsandbytes, solution/nf4.py, two candidates).

    CUDA_VISIBLE_DEVICES=0 python tools/nf4_variants.py
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from bitsandbytes import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solution import nf4  # noqa: E402


@triton.jit
def joined(packed_ptr, absmax_ptr, code_ptr, out_ptr, n_out, BLOCKSIZE: tl.constexpr,
           BYTES: tl.constexpr):
    """One lane per byte; the two values joined and stored as one contiguous pair."""
    offs = tl.program_id(0).to(tl.int64) * BYTES + tl.arange(0, BYTES)
    mask = offs < n_out // 2
    byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    scale = tl.load(absmax_ptr + offs // (BLOCKSIZE // 2), mask=mask, other=0.0)
    first = tl.load(code_ptr + (byte >> 4)) * scale
    second = tl.load(code_ptr + (byte & 15)) * scale
    pair = tl.join(first, second).to(out_ptr.dtype.element_ty)  # (BYTES, 2)
    at = tl.join(2 * offs, 2 * offs + 1)
    keep = tl.join(mask, mask)
    tl.store(out_ptr + at, pair, mask=keep)


@triton.jit
def words(packed_ptr, absmax_ptr, code_ptr, out_ptr, n_out, BLOCKSIZE: tl.constexpr,
          WORDS: tl.constexpr):
    """One lane per 32-bit word (4 bytes, 8 weights), stored as 8 contiguous values."""
    w = tl.program_id(0).to(tl.int64) * WORDS + tl.arange(0, WORDS)
    mask = w < n_out // 8
    word = tl.load(packed_ptr.to(tl.pointer_type(tl.int32)) + w, mask=mask, other=0)
    scale = tl.load(absmax_ptr + w // (BLOCKSIZE // 8), mask=mask, other=0.0)
    shifts = tl.arange(0, 8)
    # byte k of the word (little endian) holds weights 2k (high nibble) and 2k + 1 (low nibble)
    nib_shift = (shifts // 2) * 8 + (1 - shifts % 2) * 4
    nibbles = (word[:, None] >> nib_shift[None, :]) & 15
    values = (tl.load(code_ptr + nibbles) * scale[:, None]).to(out_ptr.dtype.element_ty)
    at = w[:, None] * 8 + shifts[None, :]
    tl.store(out_ptr + at, values, mask=mask[:, None])


def timed(fn, reps=50):
    fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / reps


def main():
    weight = (torch.randn(8960, 1536, device="cuda") * 0.02).half()
    packed, state = F.quantize_4bit(weight, blocksize=64, compress_statistics=False,
                                    quant_type="nf4")
    reference = F.dequantize_4bit(packed, state)
    n = weight.numel()
    out = torch.empty_like(reference)
    code = state.code.float()

    def report(name, fn):
        ms = timed(fn)
        fn()
        print(f"{name:24s} {ms * 1000:7.1f} us  {n * 2.5 / ms / 1e6:6.1f} GB/s  "
              f"same={torch.equal(out if name != 'bitsandbytes' else reference, reference)}")

    report("bitsandbytes", lambda: F.dequantize_4bit(packed, state))
    report("solution", lambda: out.copy_(nf4.dequantize_nf4(packed, state)))
    for bytes_, warps in ((512, 4), (1024, 4), (1024, 8), (2048, 8)):
        report(f"joined {bytes_}/{warps}", lambda: joined[(triton.cdiv(n // 2, bytes_),)](
            packed, state.absmax, code, out, n, BLOCKSIZE=64, BYTES=bytes_, num_warps=warps))
    for words_, warps in ((128, 4), (256, 4), (256, 8), (512, 8)):
        report(f"words {words_}/{warps}", lambda: words[(triton.cdiv(n // 8, words_),)](
            packed, state.absmax, code, out, n, BLOCKSIZE=64, WORDS=words_, num_warps=warps))


if __name__ == "__main__":
    main()
