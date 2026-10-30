"""Analysis only: how long each of our Triton kernels takes on its first call (compile included).

    python tools/compile_times.py

The measured runs start with empty Triton caches and count the compile time, so every kernel we
add costs its compile time once. This prints the first call (compile + run) and the second call
(run only) of each kernel path, using a fresh cache directory.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from solution.attention import qkv_rotate  # noqa: E402
from solution.kernels import fused_cross_entropy, silu_mul  # noqa: E402
from solution.logit_stats import entropy_from_logits, row_cross_entropy  # noqa: E402
from solution.rmsnorm import rms_norm  # noqa: E402

DEV = "cuda"
TOTAL = {"first": 0.0, "second": 0.0}


def timed(name, fn):
    out = []
    for label in ("first", "second"):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - start)
        TOTAL[label] += out[-1]
    print(f"{name:28s} first {out[0]:6.2f}s   second {out[1]*1000:7.1f}ms   compile ~{out[0]-out[1]:5.2f}s")


def main():
    torch.zeros(1, device=DEV)
    batch, seq = 4, 300
    gate = torch.randn(batch, seq, 8960, device=DEV, dtype=torch.float16, requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    timed("silu_mul fwd", lambda: silu_mul(gate, up))
    out = silu_mul(gate, up)
    grad = torch.randn_like(out)
    timed("silu_mul bwd", lambda: torch.autograd.grad(out, (gate, up), grad, retain_graph=True))

    logits = torch.randn(batch * seq, 151936, device=DEV, dtype=torch.float16, requires_grad=True)
    targets = torch.randint(0, 151936, (batch * seq,), device=DEV)
    denom = torch.tensor(1000.0, device=DEV)
    timed("ce fwd", lambda: fused_cross_entropy(logits, targets, denom))
    loss = fused_cross_entropy(logits, targets, denom)
    timed("ce bwd", lambda: torch.autograd.grad(loss, logits, retain_graph=True))
    timed("entropy", lambda: entropy_from_logits(logits.detach()))
    labels = targets.view(batch, seq)
    timed("row_losses (ce fwd reuse)", lambda: row_cross_entropy(logits.detach().view(batch, seq, -1), labels))

    x = torch.randn(batch, seq, 1536, device=DEV, requires_grad=True)
    weight = torch.ones(1536, device=DEV)
    timed("rmsnorm fwd", lambda: rms_norm(x, weight, 1e-6))
    y = rms_norm(x, weight, 1e-6)
    dy = torch.randn_like(y)
    timed("rmsnorm bwd", lambda: torch.autograd.grad(y, x, dy, retain_graph=True))

    q = torch.randn(batch, seq, 1536, device=DEV, requires_grad=True)
    k = torch.randn(batch, seq, 256, device=DEV, requires_grad=True)
    v = torch.randn(batch, seq, 256, device=DEV, requires_grad=True)
    cos = torch.randn(batch, seq, 128, device=DEV)
    sin = torch.randn(batch, seq, 128, device=DEV)
    timed("qkv_rotate fwd", lambda: qkv_rotate(q, k, v, cos, sin, 12, 2))
    outs = qkv_rotate(q, k, v, cos, sin, 12, 2)
    grads = [torch.randn_like(o) for o in outs]
    timed("qkv_rotate bwd", lambda: torch.autograd.grad(outs, (q, k, v), grads, retain_graph=True))
    print(f"TOTAL first {TOTAL['first']:.2f}s, second {TOTAL['second']:.2f}s, "
          f"compile ~{TOTAL['first'] - TOTAL['second']:.2f}s")


if __name__ == "__main__":
    main()
