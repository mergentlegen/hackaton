"""Analysis only (not part of the solution): dtypes and memory of Soup's training step on a T4.

    python tools/probe.py --config runs/base_1/task.yaml

Runs 3 steps of the base `soup train` (no solution patches) and prints `PROBE ...` lines:
parameter dtypes, the dtype and shape flowing through the first layer, the memory the forward pass
keeps for the backward, and the peak of each step.
"""

import argparse
import collections
import re
import sys

sys.modules["unsloth"] = None
sys.modules["unsloth_zoo"] = None

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402

STOP_AFTER = 3
WATCH_LAYERS = (0, 14)
STATE = {"step": 0, "base": 0.0}
SEEN = set()
STACK = []
SAVED = collections.defaultdict(float)
PARAM_PTRS = set()
SEEN_PTR = set()


def say(*parts):
    print("PROBE", *parts, flush=True)


def desc(value):
    if torch.is_tensor(value):
        return f"{tuple(value.shape)}:{str(value.dtype)[6:]}"
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(desc(item) for item in value) + ")"
    return type(value).__name__


def mib():
    return torch.cuda.memory_allocated() / 2**20


def recording():
    return STATE["step"] == 2 and torch.is_grad_enabled()


def label():
    return STACK[-1] if STACK else "functional"


def pack(tensor):
    try:
        storage = tensor.untyped_storage()
        ptr = storage.data_ptr()
        if ptr not in PARAM_PTRS and ptr not in SEEN_PTR:
            SEEN_PTR.add(ptr)
            SAVED[(label(), str(tensor.dtype)[6:], tuple(tensor.shape))] += storage.nbytes() / 2**20
    except Exception:  # noqa: BLE001 - analysis only
        pass
    return tensor


def unpack(tensor):
    return tensor


def report_saved():
    by_label = collections.defaultdict(float)
    for (name, _dtype, _shape), size in SAVED.items():
        by_label[name] += size
    say(f"saved total_MiB={sum(by_label.values()):.0f}")
    for name, size in sorted(by_label.items(), key=lambda kv: -kv[1])[:14]:
        say(f"saved by module {name}: {size:.0f} MiB")
    for (name, dtype, shape), size in sorted(SAVED.items(), key=lambda kv: -kv[1])[:30]:
        say(f"saved {size:7.1f} MiB  {dtype:8s} {shape}  <- {name}")


def wrap_model_memory():
    original_layer = mq.Qwen2DecoderLayer.forward
    original_body = mq.Qwen2Model.forward
    original_lm = mq.Qwen2ForCausalLM.forward

    def layer(self, *args, **kwargs):
        idx = getattr(self.self_attn, "layer_idx", None)
        watch = recording() and idx in WATCH_LAYERS
        before = mib()
        out = original_layer(self, *args, **kwargs)
        if watch:
            say(f"layer{idx} kept_by_forward_MiB={mib() - before:.1f} in={desc(args[0] if args else kwargs.get('hidden_states'))} out={desc(out)}")
        return out

    def body(self, *args, **kwargs):
        watch = recording()
        before = mib()
        out = original_body(self, *args, **kwargs)
        if watch:
            say(f"body kept_by_forward_MiB={mib() - before:.1f} (all 28 layers + embeddings + final norm)")
        return out

    def lm(self, *args, **kwargs):
        watch = recording()
        before = mib()
        STATE["lm_start"] = before
        if watch:
            STACK.clear()
            SEEN_PTR.clear()
            SAVED.clear()
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                out = original_lm(self, *args, **kwargs)
            report_saved()
        else:
            out = original_lm(self, *args, **kwargs)
        if watch:
            say(f"causal_lm total_MiB={mib() - before:.1f} logits={desc(out.logits)} loss={desc(out.loss)}")
        return out

    mq.Qwen2DecoderLayer.forward = layer
    mq.Qwen2Model.forward = body
    mq.Qwen2ForCausalLM.forward = lm


def register_hooks(model):
    wanted = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
              "input_layernorm", "post_attention_layernorm", "self_attn", "mlp", "lora_A.default",
              "lora_B.default", "lora_dropout.default", "embed_tokens", "lm_head", "model.norm")

    def make_hook(name):
        def hook(module, args, output):
            if name.endswith("lm_head") and recording():
                say(f"after_lm_head used_since_causal_lm_start_MiB={mib() - STATE['lm_start']:.1f}")
            if name in SEEN or STATE["step"] != 1:
                return
            SEEN.add(name)
            say(f"io {name}: in={desc(args[0]) if args else '-'} out={desc(output)}")
        return hook

    for name, module in model.named_modules():
        if ".layers.0." in name + "." and name.endswith(wanted):
            module.register_forward_hook(make_hook(name))
        elif name.endswith(("embed_tokens", "lm_head", "model.norm")):
            module.register_forward_hook(make_hook(name))


def register_stack_hooks(model):
    for name, module in model.named_modules():
        short = re.sub(r"layers\.\d+", "layers.*", name).replace("base_model.model.", "")
        module.register_forward_pre_hook(lambda mod, args, n=(short or "<root>"): STACK.append(n))
        module.register_forward_hook(lambda mod, args, out: (STACK.pop(), None)[1])
    for param in list(model.parameters()) + list(model.buffers()):
        PARAM_PTRS.add(param.untyped_storage().data_ptr())


def report_parameters(model):
    counts = collections.defaultdict(lambda: [0, 0])
    for _name, param in model.named_parameters():
        key = f"{str(param.dtype)[6:]}/trainable={param.requires_grad}"
        counts[key][0] += 1
        counts[key][1] += param.numel()
    for key, (n, numel) in sorted(counts.items()):
        say(f"params {key}: tensors={n} numel={numel / 1e6:.1f}M")
    fp32 = sorted({".".join(name.split(".")[-2:]) for name, p in model.named_parameters()
                   if p.dtype == torch.float32 and not p.requires_grad})
    say("fp32 frozen params (suffixes):", fp32[:20])
    bufs = collections.Counter(str(b.dtype)[6:] for _n, b in model.named_buffers())
    say("buffers by dtype:", dict(bufs))


class Probe(transformers.TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        STATE["step"] += 1
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        STATE["base"] = mib()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        say(f"step {STATE['step']}: before={STATE['base']:.0f} MiB, after={mib():.0f} MiB, "
            f"peak={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
        if STATE["step"] >= STOP_AFTER:
            control.should_training_stop = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    sys.argv = ["soup", "train", "--config", args.config, "--yes"]
    torch.empty(1, device="cuda")
    wrap_model_memory()

    original_init = transformers.Trainer.__init__

    def init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        self.add_callback(Probe())
        say("trainer", type(self).__name__, "model", type(self.model).__name__)
        report_parameters(self.model)
        register_hooks(self.model)
        register_stack_hooks(self.model)

    transformers.Trainer.__init__ = init
    from soup_cli.cli import run

    try:
        run()
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
