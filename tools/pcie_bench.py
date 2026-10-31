"""Analysis only: pinned host <-> GPU copy bandwidth on this machine, alone and next to compute.

    CUDA_VISIBLE_DEVICES=0 python tools/pcie_bench.py
"""

import time

import torch


def timed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps


def main():
    size = 256 << 20
    gpu = torch.empty(size, dtype=torch.uint8, device="cuda")
    t0 = time.perf_counter()
    host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    print(f"pin 256 MiB: {(time.perf_counter() - t0) * 1e3:.0f} ms")
    stream = torch.cuda.Stream()
    d2h = timed(lambda: host.copy_(gpu, non_blocking=True))
    h2d = timed(lambda: gpu.copy_(host, non_blocking=True))
    print(f"D2H {size / d2h / 1e9:.1f} GB/s, H2D {size / h2d / 1e9:.1f} GB/s")

    a = torch.randn(2048, 8960, device="cuda").half()
    b = torch.randn(8960, 1536, device="cuda").half()
    compute = timed(lambda: [a @ b for _ in range(20)])

    def both():
        with torch.cuda.stream(stream):
            host.copy_(gpu, non_blocking=True)
            gpu.copy_(host, non_blocking=True)
        for _ in range(20):
            a @ b

    together = timed(both)
    print(f"20 matmuls {compute * 1e3:.1f} ms, 2 x 256 MiB copies alone {(d2h + h2d) * 1e3:.1f} ms,"
          f" together {together * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
