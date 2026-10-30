"""Per-token statistics of the fp16 logits that Soup and TRL log every step.

TRL's `entropy_from_logits` runs a chunked fp16 log-softmax, an exp and a multiply over the whole
(tokens, vocab) matrix, and Soup's `row_losses` runs one cross-entropy per row, both after the
loss. Each is one pass per token row here (the kernel of the loss, kernels.ce_stats), in fp32
inside the kernel, and they return the same quantities.
"""

import torch

from .kernels import ce_stats
from .packed_logits import PackedLogits


def _is_root(logits):
    return isinstance(logits, PackedLogits) and logits.lo == 0 and logits.hi == logits.seq


def entropy_from_logits(logits, chunk_size=128):
    """Shannon entropy (nats) of softmax(logits) over the last dim; the shape without it."""
    if _is_root(logits):  # packed rows: padding positions get 0, the caller masks them
        no_target = torch.full((logits.rows.shape[0],), -1, dtype=torch.int64,
                               device=logits.rows.device)
        _lse, _loss, entropy = ce_stats(logits.packed, no_target, True)
        out = torch.zeros(logits.batch * logits.seq, dtype=torch.float32, device=entropy.device)
        out[logits.rows] = entropy
        return out.view(logits.batch, logits.seq)
    if isinstance(logits, PackedLogits):
        logits = logits.dense()
    vocab = logits.shape[-1]
    flat = logits.reshape(-1, vocab).contiguous()
    no_target = torch.full((flat.shape[0],), -1, dtype=torch.int64, device=logits.device)
    _lse, _loss, entropy = ce_stats(flat, no_target, True)
    return entropy.reshape(logits.shape[:-1])


def row_cross_entropy(logits, labels):
    """Mean cross-entropy and token count of every row of (batch, length, vocab) logits against
    the labels shifted by one (-100 is ignored), as two plain Python lists."""
    if _is_root(logits):
        batch, length = logits.batch, logits.seq
        shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].reshape(-1)
        shifted = shifted.to(logits.rows.device)
        _lse, losses, _ = ce_stats(logits.packed, shifted[logits.rows], False)
        per_token = torch.zeros(batch * length, dtype=torch.float32, device=losses.device)
        per_token[logits.rows] = losses
        counts = (shifted >= 0).reshape(batch, length).sum(dim=1)
        means = per_token.reshape(batch, length).sum(dim=1) / counts.clamp(min=1)
        return means.tolist(), counts.tolist()
    if isinstance(logits, PackedLogits):
        logits = logits.dense()
    batch, length, vocab = logits.shape
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    targets = shifted.reshape(-1).to(logits.device)
    _lse, losses, _ = ce_stats(logits.reshape(-1, vocab).contiguous(), targets, False)
    counts = (targets >= 0).reshape(batch, length).sum(dim=1)
    means = losses.reshape(batch, length).sum(dim=1) / counts.clamp(min=1)
    return means.tolist(), counts.tolist()
