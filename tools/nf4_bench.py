"""Analysis only: NF4 decode time, bitsandbytes against our kernel with a few launch settings.

    CUDA_VISIBLE_DEVICES=0 python tools/nf4_bench.py
"""

import sys
from pathlib import Path

import torch
from bitsandbytes import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solution import nf4  # noqa: E402


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
    ms = timed(lambda: F.dequantize_4bit(packed, state))
    gbs = weight.numel() * 2.5 / ms / 1e6
    print(f"bitsandbytes   {ms * 1000:7.1f} us  {gbs:6.1f} GB/s")
    reference = F.dequantize_4bit(packed, state)
    for variant in range(4):
        for outputs in (512, 1024, 2048):
            for warps in (4, 8):
                nf4.VARIANT, nf4.OUTPUTS, nf4.WARPS = variant, outputs, warps
                ms = timed(lambda: nf4.dequantize_nf4(packed, state))
                same = torch.equal(nf4.dequantize_nf4(packed, state), reference)
                print(f"ours v{variant} {outputs:5d}/{warps}  {ms * 1000:7.1f} us  "
                      f"{weight.numel() * 2.5 / ms / 1e6:6.1f} GB/s  same={same}")

if __name__ == "__main__":
    main()
