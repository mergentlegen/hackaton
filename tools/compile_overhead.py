"""Analysis only: the fixed cost of compiling one more Triton kernel on a cold cache.

    CUDA_VISIBLE_DEVICES=0 python tools/compile_overhead.py

Times the first call of a few tiny, distinct kernels (and a second call of each): the first
one also pays Triton's one-time setup, the others show what every extra kernel costs.
"""

import os
import tempfile
import time

os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp()

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def k1(x, n, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(x + o, tl.load(x + o, mask=o < n) + 1, mask=o < n)


@triton.jit
def k2(x, n, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(x + o, tl.load(x + o, mask=o < n) * 2, mask=o < n)


@triton.jit
def k3(x, y, n, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(y + o, tl.load(x + o, mask=o < n) - 1, mask=o < n)


@triton.jit
def k4(x, y, n, m, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    v = tl.load(x + o, mask=o < n)
    if m == 0:
        v = v + 3
    else:
        v = tl.exp(v)
    tl.store(y + o, v, mask=o < n)


@triton.jit
def k5(x, n, B: tl.constexpr, C: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(x + o, tl.load(x + o, mask=o < n) * C, mask=o < n)


@triton.jit(do_not_specialize=["n"])
def k6(x, n, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(x + o, tl.load(x + o, mask=o < n) + 5, mask=o < n)


@triton.jit
def k7(x, y, n, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(y + o, (tl.load(x + o, mask=o < n) * 3).to(y.dtype.element_ty), mask=o < n)


def main():
    x = torch.zeros(4096, device="cuda")
    y = torch.zeros_like(x)
    h = torch.zeros(4096, device="cuda", dtype=torch.float16)
    for name, call in (("k1 (first: + Triton setup)", lambda: k1[(4,)](x, 4096, B=1024)),
                       ("k2", lambda: k2[(4,)](x, 4096, B=1024)),
                       ("k3 (other signature)", lambda: k3[(4,)](x, y, 4096, B=1024)),
                       ("k4 (two branches)", lambda: k4[(4,)](x, y, 4096, 1, B=1024)),
                       ("k4 again, other branch value", lambda: k4[(4,)](x, y, 4096, 0, B=1024)),
                       ("k5 (k1's args + a constexpr)", lambda: k5[(4,)](x, 4096, B=1024, C=3)),
                       ("k6 (k1's args, n unspecialized)", lambda: k6[(4,)](x, 4096, B=1024)),
                       ("k7 (k3's args, fp16 output)", lambda: k7[(4,)](x, h, 4096, B=1024))):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        first = time.perf_counter() - start
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        print(f"{name:32s} first {first:6.3f}s  second {time.perf_counter() - start:7.4f}s")


if __name__ == "__main__":
    main()
