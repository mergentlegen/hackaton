"""Print the key numbers of the measurement files in one place.

    python tools/summarize_runs.py                      # looks in runs/
    python tools/summarize_runs.py --runs runs_base     # another output folder

Reads, when they exist: baseline_probe.json (tools/baseline_probe.py), result.json (run_task.py),
checks.json (check_rules.py) and any profile.json (torch.profiler table of GPU kernels).
"""

import argparse
import json
import statistics
from pathlib import Path

GROUPS = (  # substring of the CUDA kernel name -> bucket (first match wins)
    ("triton_", "triton (ours or inductor)"),
    ("dequant", "NF4 dequantize"),
    ("gemm", "matmul"), ("cutlass", "matmul"), ("xmma", "matmul"), ("cublas", "matmul"),
    ("gemv", "matmul"),
    ("attention", "attention"), ("fmha", "attention"), ("efficient", "attention"),
    ("softmax", "softmax / cross entropy"), ("cross_entropy", "softmax / cross entropy"),
    ("nll", "softmax / cross entropy"),
    ("elementwise", "elementwise / casts"), ("vectorized", "elementwise / casts"),
    ("copy", "elementwise / casts"), ("cast", "elementwise / casts"),
    ("reduce", "reductions"), ("index", "indexing"), ("memcpy", "memcpy"),
)


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def show_probe(folder):
    probe = load(folder / "baseline_probe.json")
    if not probe:
        return
    print("=== baseline_probe.json (plain Soup) ===")
    print("phases:", json.dumps(probe.get("phases")))
    print("summary:", json.dumps(probe.get("summary")))
    print("padding:", json.dumps(probe.get("padding")))
    print("dtypes:", json.dumps(probe.get("dtypes")))
    print("torch peak reserved MiB:", probe.get("torch_peak_reserved_mib"))
    steps = probe.get("steps", [])
    if steps:
        print("per-step peak alloc MiB (first 8):", [s["peak_alloc_mib"] for s in steps[:8]])


def show_result(folder):
    result = load(folder / "result.json")
    if not result:
        return
    print("=== result.json ===")
    print("versions:", json.dumps(result.get("versions")))
    print("summary:", json.dumps(result.get("summary", {}).get("ratios")))
    for side in ("base", "solution"):
        for run in result.get("runs", {}).get(side, []):
            m = run.get("metrics", {})
            own = m.get("own_kernel_ms", {})
            loop = m.get("timed_loop_s") or 0
            share = 100 * sum(own.values()) / 1000 / loop if loop else 0
            print(f"{run['run']:12s} exit={run['exit_code']} wall={run['wall_s']}s "
                  f"peak={run['gpu_peak_mib']}MiB train_loop={m.get('train_loop_s')}s "
                  f"median_step={m.get('median_step_s')}s own_kernel_share={share:.1f}% "
                  f"torch_reserved={m.get('torch_max_reserved_mib')}MiB "
                  f"outside_loop={round(run['wall_s'] - (m.get('train_loop_s') or 0), 1)}s")
    for side, ev in result.get("evals", {}).items():
        print(f"eval {side}: mean_loss={ev.get('mean_loss')}")
    base = [r["wall_s"] for r in result.get("runs", {}).get("base", [])]
    if len(base) > 1:
        print(f"base wall times {base}: spread {100 * (max(base) - min(base)) / statistics.mean(base):.1f}%")
    checks = load(folder / "checks.json")
    if checks:
        for c in checks["checks"]:
            print(("PASS " if c["ok"] else "FAIL ") + c["check"] + " :: " + str(c["detail"])[:200])


def show_profile(path):
    data = load(path)
    if not data:
        return
    total = data["total_gpu_us"] or 1
    print(f"=== {path} (GPU time of {data['active_steps']} steps: {total / 1000:.0f} ms) ===")
    buckets = {}
    for k in data["kernels"]:
        name = k["name"].lower()
        bucket = next((label for key, label in GROUPS if key in name), "other")
        buckets[bucket] = buckets.get(bucket, 0) + k["time_us"]
    for bucket, us in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {100 * us / total:5.1f}%  {bucket}")
    print("  top kernels:")
    for k in data["kernels"][:12]:
        print(f"  {100 * k['time_us'] / total:5.1f}%  x{k['count']:<6d} {k['name'][:90]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs")
    folder = Path(parser.parse_args().runs)
    show_probe(folder)
    show_result(folder)
    for path in sorted(folder.glob("**/profile.json")) + sorted(folder.glob("bp_profile.json")):
        show_profile(path)


if __name__ == "__main__":
    main()
