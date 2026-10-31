"""Analysis only: where the wall time of one `soup train` goes outside the training steps.

    python tools/phase_times.py --config runs/base_1/task.yaml [--solution]

Prints `PHASE <name> <seconds since this process started>` at the main milestones: imports, the
solution applied, the model loaded and quantized, LoRA attached, the trainer ready, the first and
the last step, the end of training, the saves, and the return of `soup train`.
"""

import argparse
import sys
import time

T0 = time.time()
sys.modules["unsloth"] = None
sys.modules["unsloth_zoo"] = None


def mark(name):
    print(f"PHASE {name:34s} {time.time() - T0:7.2f}", flush=True)


import torch  # noqa: E402
import transformers  # noqa: E402

mark("imports: torch + transformers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--solution", action="store_true")
    args = parser.parse_args()
    torch.empty(1, device="cuda")
    mark("cuda context")
    if args.solution:
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import solution

        solution.apply()
        mark("solution.apply() done")

    wrapped = {}

    def wrap(owner, name, label):
        original = getattr(owner, name)

        def call(*a, **k):
            mark(label + " begin")
            result = original(*a, **k)
            mark(label + " end")
            return result

        wrapped[label] = original
        setattr(owner, name, call)

    import peft

    wrap(peft, "prepare_model_for_kbit_training", "prepare_kbit")
    wrap(peft, "get_peft_model", "get_peft_model")
    original_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    def from_pretrained(*a, **k):
        mark("from_pretrained begin")
        result = original_pretrained(*a, **k)
        mark("from_pretrained end")
        return result

    transformers.AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained)

    class Phases(transformers.TrainerCallback):
        seen = 0

        def on_train_begin(self, a, state, control, **kw):
            mark("train begin")

        def on_step_end(self, a, state, control, **kw):
            Phases.seen += 1
            if Phases.seen in (1, 64):
                mark(f"step {Phases.seen} end")

        def on_train_end(self, a, state, control, **kw):
            mark("train end")

        def on_save(self, a, state, control, **kw):
            mark("save")

    original_init = transformers.Trainer.__init__

    def init(self, *a, **k):
        mark("Trainer init begin")
        original_init(self, *a, **k)
        self.add_callback(Phases())
        mark("Trainer init end")

    transformers.Trainer.__init__ = init
    sys.argv = ["soup", "train", "--config", args.config, "--yes"]
    from soup_cli.cli import run

    mark("soup_cli imported")
    try:
        run()
    except SystemExit:
        pass
    mark("soup run returned")


if __name__ == "__main__":
    main()
