"""Solution for Soup Kernel Challenge (Decentrathon 6.0).

Optimizations applied via `apply()`:
1. Unified Fused RMSNorm & Fused Residual-Add + RMSNorm in Triton (`solution/rmsnorm.py`)
2. Unified Zero-Copy Stride-Native BSHD/BHSD Fused Q+K RoPE in Triton (`solution/rope.py`)
3. Fused NF4/FP4 Double-Quant Dequantization + Packed SwiGLU MLP + Fused QKV Base Projection + Clone-Free LoRA (`solution/nf4_linear.py`)
4. Fused Chunked Head + 2-Pass Online-Softmax Cross-Entropy, Entropy & Accuracy in Triton (`solution/cross_entropy.py`)
5. In-Batch Valid-Token Compaction, Direct C++ Varlen SM75 CUTLASS Attention, FP16 Frozen Embeddings,
   and Token-Aware Selective Attention Checkpointing (`solution/padding.py`)
"""

from .cross_entropy import patch_soup_trainer
from .nf4_linear import patch_bnb_nf4_linear
from .rmsnorm import patch_qwen2_rmsnorm
from .rope import patch_qwen2_rope


def apply() -> None:
    """Install all custom Triton kernel patches and Soup trainer hooks."""
    patch_qwen2_rmsnorm()
    patch_qwen2_rope()
    patch_bnb_nf4_linear()
    patch_soup_trainer()
