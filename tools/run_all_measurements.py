"""Run every measurement of the plan, one after another, on a Kaggle T4, and write ONE report.

    python tools/run_all_measurements.py

Steps (each in its own process, GPU checked idle before it, nvidia-smi peak recorded):
  warmup      download the model once so no timed run pays for it
  pytest      our kernels against PyTorch on the real GPU
  probe       plain Soup: phases, padding, dtypes, per-step peak memory
  base_prof   plain Soup under torch.profiler
  base_x2     two plain-Soup runs (spread between repeats)
  sol_try     the solution alone with the profiler
  full        base, solution, solution, base like the jury, then check_rules and score
A failing step never stops the next ones. The result is runs/MEASUREMENT_REPORT.txt, copied to
/kaggle/working/ together with a zip of the raw JSON files. Takes about 35-45 minutes.
Do not run anything else on the GPUs meanwhile.
"""

import datetime
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
from common import MODEL, OFFLINE_ENV, library_versions, run_logged  # noqa: E402

RUNS = REPO / "runs"
LOGS = REPO / "runs_log"
LOGS.mkdir(exist_ok=True)
RUNS.mkdir(exist_ok=True)
PY = sys.executable
results = []


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False).stdout
    return sum(int(x) for x in out.split() if x.strip().isdigit())


def wait_idle(limit_s=90):
    start = time.time()
    while gpu_used_mib() > 900 and time.time() - start < limit_s:
        time.sleep(3)
    return gpu_used_mib()


def step(name, cmd, env=None, timeout=2700):
    idle = wait_idle()
    print(f"[measure] {name}: start (GPU used {idle} MiB) ...", flush=True)
    log = LOGS / f"{name}.log"
    code, wall, peak, start = run_logged(cmd, log, timeout, env or OFFLINE_ENV)
    results.append({"step": name, "exit": code, "wall_s": wall, "nvidia_smi_peak_mib": peak,
                    "gpu_used_at_start_mib": start, "log": str(log.relative_to(REPO))})
    print(f"[measure] {name}: exit={code} wall={wall}s peak={peak} MiB", flush=True)
    return code


def tail(path, n=60):
    try:
        return "\n".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def main():
    started = datetime.datetime.now().isoformat(timespec="seconds")
    if not (REPO / "data" / "train.jsonl").exists():
        step("prepare_data", [PY, "task/prepare_data.py"], dict(OFFLINE_ENV, HF_HUB_OFFLINE="0"))
    step("warmup", [PY, "-c",
                    "import soup_cli.cli, transformers, peft, trl, bitsandbytes, triton; "
                    "from huggingface_hub import snapshot_download; "
                    f"print(snapshot_download({MODEL!r}))"], dict(OFFLINE_ENV, HF_HUB_OFFLINE="0"))
    step("pytest", [PY, "-m", "pytest", "solution", "-q", "-p", "no:cacheprovider"])
    probe_env = dict(OFFLINE_ENV, TRITON_CACHE_DIR=str(RUNS / "probe_triton"))
    step("probe", [PY, "tools/baseline_probe.py"], probe_env)
    step("base_prof", [PY, "harness/_train.py", "--config", "runs/baseline_probe/task.yaml",
                       "--metrics", "runs/bp_metrics.json", "--profile", "runs/bp_profile.json"],
         dict(OFFLINE_ENV, TRITON_CACHE_DIR=str(RUNS / "bp_triton"),
              SOUP_DB_PATH=str(RUNS / "bp.db")))
    step("base_x2", [PY, "harness/run_task.py", "--only", "base", "--repeats", "2",
                     "--out", "runs_base"])
    step("sol_try", [PY, "harness/run_task.py", "--only", "solution", "--out", "runs_try",
                     "--profile"])
    step("full", [PY, "harness/run_task.py", "--repeats", "2"], timeout=3600)
    step("check_rules", [PY, "harness/check_rules.py"], timeout=3600)
    step("score", [PY, "harness/score.py", "runs/result.json"])

    lines = [f"MEASUREMENT REPORT  started {started}  finished "
             f"{datetime.datetime.now().isoformat(timespec='seconds')}",
             json.dumps(library_versions(), indent=1), "", "=== steps ==="]
    for r in results:
        lines.append(f"{r['step']:13s} exit={r['exit']:<4} wall={r['wall_s']:8.1f}s "
                     f"nvidia-smi peak={r['nvidia_smi_peak_mib']} MiB "
                     f"gpu at start={r['gpu_used_at_start_mib']} MiB")
    for r in results:
        if r["exit"] != 0:
            lines += ["", f"=== FAILED {r['step']} (last 60 lines of {r['log']}) ===",
                      tail(REPO / r["log"])]
    lines += ["", "=== pytest (our kernels on the GPU) ===", tail(LOGS / "pytest.log", 30)]
    for folder in ("runs", "runs_base", "runs_try"):
        summary = subprocess.run([PY, "tools/summarize_runs.py", "--runs", folder], cwd=REPO,
                                 capture_output=True, text=True, check=False)
        lines += ["", f"=== summarize {folder} ===", summary.stdout, summary.stderr[-800:]]
    lines += ["", "=== score ===", tail(LOGS / "score.log", 10),
              "", "=== check_rules ===", tail(LOGS / "check_rules.log", 25)]
    report = "\n".join(lines)
    (RUNS / "MEASUREMENT_REPORT.txt").write_text(report, encoding="utf-8")
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO
    shutil.copy(RUNS / "MEASUREMENT_REPORT.txt", out / "MEASUREMENT_REPORT.txt")
    keep = [RUNS / "result.json", RUNS / "checks.json", RUNS / "baseline_probe.json",
            RUNS / "bp_profile.json", RUNS / "bp_metrics.json", REPO / "runs_base" / "result.json",
            REPO / "runs_try" / "result.json", REPO / "runs_try" / "solution_profile" / "profile.json"]
    with zipfile.ZipFile(out / "measurement_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in keep + sorted(LOGS.glob("*.log")):
            if path.exists():
                bundle.write(path, path.relative_to(REPO))
    print("\n" + report)
    print(f"\n[measure] report: {out / 'MEASUREMENT_REPORT.txt'}  bundle: {out / 'measurement_bundle.zip'}")


if __name__ == "__main__":
    main()
