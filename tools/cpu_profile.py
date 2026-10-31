"""Analysis only: where the CPU (Python) time of a training step goes.

    python tools/cpu_profile.py --config runs/base_1/task.yaml [--solution]

Runs cProfile on the main thread over steps 4-19 of `soup train` (then stops) and prints the
functions with the most own time and the most cumulative time. The GPU is mostly waiting for the
CPU in this task, so this shows what to cut.
"""

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

sys.modules["unsloth"] = None
sys.modules["unsloth_zoo"] = None

import torch  # noqa: E402
import transformers  # noqa: E402

FIRST, LAST = 4, 19


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--solution", action="store_true")
    args = parser.parse_args()
    torch.empty(1, device="cuda")
    if args.solution:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import solution

        solution.apply()

    profile = cProfile.Profile()

    class Probe(transformers.TrainerCallback):
        steps = 0

        def on_step_begin(self, a, state, control, **kw):
            Probe.steps += 1
            if Probe.steps == FIRST:
                torch.cuda.synchronize()
                profile.enable()

        def on_step_end(self, a, state, control, **kw):
            if Probe.steps == LAST:
                torch.cuda.synchronize()
                profile.disable()
                control.should_training_stop = True
                for pattern in ("'item'", "named_modules", "acquire", "commit", "wait"):
                    out = io.StringIO()
                    pstats.Stats(profile, stream=out).sort_stats("tottime").print_callers(pattern)
                    lines = [line for line in out.getvalue().splitlines() if line.strip()]
                    print(f"CPUPROF ---- callers of {pattern}")
                    for line in lines[2:14]:
                        print("CPUPROF", line[:200])
                for order, rows in (("tottime", 12),):
                    out = io.StringIO()
                    stats = pstats.Stats(profile, stream=out).sort_stats(order)
                    stats.print_stats(rows)
                    lines = [line for line in out.getvalue().splitlines() if line.strip()]
                    print(f"CPUPROF ---- by {order}, steps {FIRST}-{LAST}")
                    for line in lines[4:]:
                        print("CPUPROF", line[:170])

    original_init = transformers.Trainer.__init__

    def init(self, *a, **k):
        original_init(self, *a, **k)
        self.add_callback(Probe())

    transformers.Trainer.__init__ = init
    sys.argv = ["soup", "train", "--config", args.config, "--yes"]
    from soup_cli.cli import run

    try:
        run()
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
