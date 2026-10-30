"""Solution for Soup Kernel Challenge (Decentrathon 6.0).

Optimizations applied via `apply()`:
1. Fused RMSNorm & Fused Residual-Add + RMSNorm in Triton (`solution/rmsnorm.py`)
2. Zero-copy stride-native Fused Q+K RoPE in Triton (`solution/rope.py`)
3. Fused NF4/FP4 Double-Quant Dequantization in Triton + clone-free `addmm` LoRA (`solution/nf4_linear.py`)
4. Fused Chunked Head + 2-Pass Online-Softmax Cross-Entropy, Entropy & Accuracy in Triton (`solution/cross_entropy.py`)
5. In-batch valid-token compaction (unpadding), T4-optimal GQA SDPA, FP16 frozen embedding preservation,
   and low-overhead custom decoder-layer activation checkpointing (`solution/padding.py`)
6. Fused SwiGLU in Triton (`solution/swiglu.py`)
"""

from .cross_entropy import patch_soup_trainer
from .nf4_linear import patch_bnb_nf4_linear
from .rmsnorm import patch_qwen2_rmsnorm
from .rope import patch_qwen2_rope
from .swiglu import patch_qwen2_mlp


def apply() -> None:
    """Install all custom Triton kernel patches and Soup trainer hooks."""
    patch_qwen2_mlp()
    patch_qwen2_rmsnorm()
    patch_qwen2_rope()
    patch_bnb_nf4_linear()
    patch_soup_trainer()
