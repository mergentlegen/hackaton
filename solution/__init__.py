"""Pad-free fused Triton training path for Qwen2 LoRA on a T4 (see README.md).

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` only replaces Python callables
(Qwen2ForCausalLM.forward, TRL's entropy helper, Soup's rewind helper); it does not touch the
task, the batches, the optimizer or any Soup setting.
"""


def apply():
    from . import qwen_fast

    installed = qwen_fast.install()
    print(f"[solution] fused Triton path installed: {', '.join(installed)}", flush=True)
