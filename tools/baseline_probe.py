"""Diagnostics of PLAIN Soup (no solution) on the task: where time and memory go.

    python task/prepare_data.py
    python tools/baseline_probe.py            # -> runs/baseline_probe.json and a printed summary

Not part of the submission and not used by the harness. It records
  * the phases of the command: imports, trainer setup, training loop, saving;
  * per step: real tokens, padded tokens, step time, peak allocated memory;
  * the padding side and the dtypes the base model really computes in;
  * the peak memory of each step, so the step that sets the peak is known.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

T0 = time.perf_counter()
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
from common import ENV  # noqa: E402

os.environ.update({k: ENV[k] for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                                       "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM",
                                       "SOUP_NO_AUDIT_LOG")})
out_dir = REPO / "runs" / "baseline_probe"
out_dir.mkdir(parents=True, exist_ok=True)
os.environ["SOUP_DB_PATH"] = str(out_dir / "experiments.db")

import torch  # noqa: E402

sys.modules["unsloth"] = None
sys.modules["unsloth_zoo"] = None
torch.empty(1, device="cuda")
import transformers  # noqa: E402
import yaml  # noqa: E402

t_imports = time.perf_counter() - T0
task = yaml.safe_load((REPO / "task" / "task.yaml").read_text(encoding="utf-8"))
task["data"]["train"] = str(REPO / "data" / "train.jsonl")
task["output"] = str(out_dir / "adapter")
config_path = out_dir / "task.yaml"
config_path.write_text(yaml.safe_dump(task, sort_keys=False, allow_unicode=True), encoding="utf-8")

rec = {"phases": {"imports_s": round(t_imports, 2)}, "steps": [], "batches": [], "dtypes": {}}
mark = {}


def hook_once(name, module):
    def hook(_m, _i, out):
        tensor = out[0] if isinstance(out, tuple) else out
        rec["dtypes"].setdefault(name, str(getattr(tensor, "dtype", type(tensor))))
    module.register_forward_hook(hook)


class Probe(transformers.TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kw):
        torch.cuda.synchronize()
        mark["train_begin"] = time.perf_counter()
        rec["phases"]["setup_until_train_s"] = round(mark["train_begin"] - T0 - t_imports, 2)
        rec["phases"]["gpu_mem_at_train_start_mib"] = round(torch.cuda.memory_allocated() / 2**20)
        names = {"embed_tokens": "embed_out", "layers.0.input_layernorm": "norm1_out",
                 "layers.0.self_attn.q_proj": "q_proj_out", "layers.0.self_attn.k_proj": "k_proj_out",
                 "layers.0.mlp.gate_proj": "gate_out", "layers.0.mlp.act_fn": "act_out",
                 "layers.0.mlp.down_proj": "down_out", "lm_head": "logits"}
        for module_name, module in model.named_modules():
            for suffix, label in names.items():
                if module_name.endswith(suffix):
                    hook_once(label, module)
        dt = rec["dtypes"]
        for module_name, p in model.named_parameters():
            if module_name.endswith("embed_tokens.weight"):
                dt["param_embed"] = str(p.dtype)
            elif module_name.endswith("layers.0.input_layernorm.weight"):
                dt["param_norm"] = str(p.dtype)
            elif module_name.endswith("layers.0.self_attn.q_proj.lora_A.default.weight"):
                dt["param_lora_A"] = str(p.dtype)
                dt["lora_A_requires_grad"] = p.requires_grad
            elif module_name.endswith("layers.0.self_attn.q_proj.base_layer.bias"):
                dt["param_q_bias"] = str(p.dtype)
        dt["lm_head_tied"] = bool(model.get_input_embeddings().weight is model.get_output_embeddings().weight) \
            if hasattr(model, "get_input_embeddings") else None

    def on_step_begin(self, args, state, control, **kw):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        rec["steps"].append({"s": round(time.perf_counter() - self.t, 4),
                             "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                             "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)})

    def on_train_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        mark["train_end"] = time.perf_counter()
        rec["phases"]["train_loop_s"] = round(mark["train_end"] - mark["train_begin"], 2)


original_init = transformers.Trainer.__init__


def init(self, *a, **kw):
    original_init(self, *a, **kw)
    self.add_callback(Probe())


transformers.Trainer.__init__ = init
original_step = transformers.Trainer.training_step


def training_step(self, model, inputs, *a, **kw):
    mask = inputs.get("attention_mask")
    if mask is not None:
        b, t = mask.shape
        lens = mask.sum(1)
        ar = torch.arange(t, device=mask.device)
        right = bool((mask.bool() == (ar[None, :] < lens[:, None])).all())
        labels = inputs.get("labels")
        rec["batches"].append({
            "rows": int(b), "padded_len": int(t), "real_tokens": int(lens.sum()),
            "padded_tokens": int(b * t), "right_padded": right,
            "supervised": int((labels[:, 1:] != -100).sum()) if labels is not None else None})
    return original_step(self, model, inputs, *a, **kw)


transformers.Trainer.training_step = training_step

sys.argv = ["soup", "train", "--config", str(config_path), "--yes"]
code = 0
try:
    from soup_cli.cli import run

    run()
except SystemExit as exc:
    code = 0 if exc.code is None else exc.code
t_end = time.perf_counter()
rec["phases"]["after_train_until_exit_s"] = round(t_end - mark.get("train_end", t_end), 2)
rec["phases"]["total_in_process_s"] = round(t_end - T0, 2)
rec["exit_code"] = code
rec["torch_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)

steps = rec["steps"]
batches = rec["batches"]
if steps:
    times = [s["s"] for s in steps]
    rec["summary"] = {
        "step_s_median": statistics.median(times), "step_s_first3": times[:3],
        "step_s_mean_after_3": round(statistics.mean(times[3:]), 4) if len(times) > 3 else None,
        "max_peak_alloc_mib": max(s["peak_alloc_mib"] for s in steps),
        "step_of_max_peak": 1 + max(range(len(steps)), key=lambda i: steps[i]["peak_alloc_mib"]),
    }
if batches:
    real = sum(b["real_tokens"] for b in batches)
    padded = sum(b["padded_tokens"] for b in batches)
    rec["padding"] = {
        "real_tokens": real, "padded_tokens": padded,
        "padded_over_real": round(padded / real, 3),
        "wasted_fraction": round(1 - real / padded, 3),
        "all_right_padded": all(b["right_padded"] for b in batches),
        "max_real_tokens_in_a_batch": max(b["real_tokens"] for b in batches),
        "max_padded_len": max(b["padded_len"] for b in batches),
        "supervised_over_real": round(sum(b["supervised"] or 0 for b in batches) / real, 3),
    }
(REPO / "runs").mkdir(exist_ok=True)
(REPO / "runs" / "baseline_probe.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
print("\n=== baseline probe ===")
print(json.dumps({k: rec[k] for k in ("phases", "dtypes", "summary", "padding", "torch_peak_reserved_mib")
                  if k in rec}, indent=2))
sys.exit(code)
