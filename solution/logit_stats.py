"""Per-token statistics of the fp16 logits that Soup and TRL log every step.

TRL's `entropy_from_logits` runs a chunked fp16 log-softmax, an exp and a multiply over the whole
(tokens, vocab) matrix, and Soup's `row_losses` runs one cross-entropy per row, both after the
loss. Each is one pass per token row here, in fp32 inside the kernel, and they return the same
quantities.
"""

import torch
import triton
import triton.language as tl

from .kernels import CE_BLOCK, NEG, ce_forward_kernel


@triton.jit
def entropy_kernel(logits_ptr, out_ptr, vocab, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * vocab
    offs = tl.arange(0, BLOCK)
    run_max = tl.full([BLOCK], NEG, tl.float32)
    run_sum = tl.zeros([BLOCK], tl.float32)
    run_dot = tl.zeros([BLOCK], tl.float32)
    for start in range(0, vocab, BLOCK):
        idx = start + offs
        mask = idx < vocab
        x = tl.load(base + idx, mask=mask, other=NEG).to(tl.float32)
        new_max = tl.maximum(run_max, x)
        rescale = tl.exp(run_max - new_max)
        weight = tl.exp(x - new_max)
        run_sum = run_sum * rescale + weight
        run_dot = run_dot * rescale + weight * tl.where(mask, x, 0.0)
        run_max = new_max
    row_max = tl.max(run_max, axis=0)
    align = tl.exp(run_max - row_max)
    total = tl.sum(run_sum * align, axis=0)
    dot = tl.sum(run_dot * align, axis=0)
    # entropy of softmax(x) = logsumexp(x) - E[x]
    tl.store(out_ptr + row, row_max + tl.log(total) - dot / total)


def entropy_from_logits(logits, chunk_size=128):
    """Shannon entropy (nats) of softmax(logits) over the last dim; the shape without it."""
    vocab = logits.shape[-1]
    flat = logits.reshape(-1, vocab).contiguous()
    out = torch.empty(flat.shape[0], dtype=torch.float32, device=logits.device)
    entropy_kernel[(flat.shape[0],)](flat, out, vocab, BLOCK=CE_BLOCK, num_warps=8)
    return out.reshape(logits.shape[:-1])


def row_cross_entropy(logits, labels):
    """Mean cross-entropy and token count of every row of (batch, length, vocab) logits against
    the labels shifted by one (-100 is ignored), as two plain Python lists."""
    batch, length, vocab = logits.shape
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    targets = shifted.reshape(-1).to(logits.device)
    flat = logits.reshape(-1, vocab).contiguous()
    rows = flat.shape[0]
    lse = torch.empty(rows, dtype=torch.float32, device=logits.device)
    losses = torch.empty_like(lse)
    ce_forward_kernel[(rows,)](flat, targets, lse, losses, vocab, BLOCK=CE_BLOCK, num_warps=8)
    counts = (targets >= 0).reshape(batch, length).sum(dim=1)
    means = losses.reshape(batch, length).sum(dim=1) / counts.clamp(min=1)
    return means.tolist(), counts.tolist()
