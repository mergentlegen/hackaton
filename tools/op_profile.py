"""Analysis only: which torch ops (with their tensor shapes) take the GPU time in a step.

    python tools/op_profile.py --config runs/base_1/task.yaml [--solution]

Profiles steps 7-10 of `soup train` with torch.profiler (record_shapes) and prints the ops with
the most GPU time, grouped by input shapes, so casts and copies show which tensors they move.
"""

import argparse
import sys
from pathlib import Path

sys.modules["unsloth"] = None
sys.modules["unsloth_zoo"] = None

import torch  # noqa: E402
import transformers  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

FIRST, LAST = 7, 10


def gaps(prof):
    """GPU idle time between consecutive kernels inside the profiled window, largest first."""
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
               and e.time_range.end > e.time_range.start]
    kernels.sort(key=lambda e: e.time_range.start)
    if not kernels:
        return
    begin, end = kernels[0].time_range.start, kernels[-1].time_range.end
    busy, last_end, found = 0.0, begin, []
    for k in kernels:
        start, stop = k.time_range.start, k.time_range.end
        if start > last_end:
            found.append((start - last_end, (last_end - begin) / 1e3, prev.name, k.name))
        busy += max(0.0, stop - max(start, last_end))
        if stop > last_end:
            last_end, prev = stop, k
    span = end - begin
    n = LAST - FIRST + 1
    print(f"OPPROF window {span / 1e3 / n:.0f} ms/step, GPU busy {busy / 1e3 / n:.0f} ms/step, "
          f"idle {(span - busy) / 1e3 / n:.0f} ms/step in {len(found)} gaps")
    big = sorted(found, key=lambda g: -g[0])[:14]
    for length, at, before, after in big:
        print(f"OPPROF gap {length / 1e3:6.1f} ms at {at:7.1f} ms | after {before[:44]} | before {after[:44]}")
    small = sum(g[0] for g in found if g[0] < 500)
    print(f"OPPROF gaps under 0.5 ms add up to {small / 1e3 / n:.0f} ms/step")


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

    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)

    class Probe(transformers.TrainerCallback):
        steps = 0

        def on_step_begin(self, a, state, control, **kw):
            Probe.steps += 1
            if Probe.steps == FIRST:
                torch.cuda.synchronize()
                prof.start()

        def on_step_end(self, a, state, control, **kw):
            if Probe.steps == LAST:
                torch.cuda.synchronize()
                prof.stop()
                control.should_training_stop = True
                total = sum(e.self_device_time_total for e in prof.key_averages()
                            if e.device_type == torch.autograd.DeviceType.CUDA)
                rows = prof.key_averages(group_by_input_shape=True)
                rows = sorted(rows, key=lambda e: -e.self_device_time_total)
                print(f"OPPROF GPU kernels total {total / 1e3:.0f} ms over {LAST - FIRST + 1} steps")
                shown = 0
                for e in rows:
                    if e.self_device_time_total <= 0:
                        continue
                    shapes = str(e.input_shapes)[:70]
                    print(f"OPPROF {100 * e.self_device_time_total / total:5.1f}% "
                          f"{e.count // (LAST - FIRST + 1):5d}/step {e.key[:46]:46s} {shapes}")
                    shown += 1
                    if shown >= 40:
                        break
                gaps(prof)

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
