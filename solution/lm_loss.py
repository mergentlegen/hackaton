"""lm_head and the loss in chunks of rows, without the (tokens, vocab) logits.

The logits of the packed step are (N, 151936) fp16: 297 KB per token, the largest tensor left on
the GPU once the layers' activations are offloaded (offload.py). The lm_head weight is frozen
(the tied embedding table), so the backward of lm_head + loss only needs d loss / d hidden, and
that is linear in the incoming gradient:

    d loss / d hidden = grad_output / denom * (softmax(logits) - onehot(target)) @ W

so it is computed in the forward, chunk by chunk, with grad_output applied in the backward.
Per chunk of rows: logits = hidden @ W^T (the matmul lm_head runs, fp16), one pass of
the statistics pass of `ce_kernel` (lse, loss, entropy, argmax: what the loss and TRL / Soup's
metrics need), one gradient pass writing softmax - onehot over the chunk's logits in place, and one
matmul into d hidden. Only one chunk of logits exists at a time.

The fp16 gradient of the logits is written as (softmax - onehot) * 1024 (a power of two, so the
factor is exact, divided out again in fp32 in the backward): small probabilities (1e-6 and
below) stay normal fp16 numbers instead of subnormals, and the largest value, 1024 times a few
weights, is far from the fp16 limit. The reference rounds grad_output / denom * (softmax -
onehot), about 80 times, to fp16, so this is at least as precise.
"""

import torch

from .kernels import ce_grad, ce_stats

CHUNK_ROWS = 256  # 74 MiB of fp16 logits per chunk
GRAD_SCALE = 1024.0


_LOGITS_BUF = None

def _get_logits_buf(n_rows, vocab, device, dtype):
    global _LOGITS_BUF
    if (_LOGITS_BUF is None or _LOGITS_BUF.shape[0] < n_rows or _LOGITS_BUF.shape[1] < vocab
            or _LOGITS_BUF.device != device or _LOGITS_BUF.dtype != dtype):
        _LOGITS_BUF = torch.empty(max(n_rows, CHUNK_ROWS), vocab, device=device, dtype=dtype)
    return _LOGITS_BUF[:n_rows]

def _chunks(hidden, weight, targets, lse, losses, entropy, argmax, unit, grad_hidden):
    rows = hidden.shape[0]
    vocab = weight.shape[0]
    weight_t = weight.t()
    for start in range(0, rows, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, rows)
        part = slice(start, end)
        n_rows = end - start
        logits = _get_logits_buf(n_rows, vocab, hidden.device, hidden.dtype)
        torch.mm(hidden[part], weight_t, out=logits)
        ce_stats(logits, targets[part], True,
                 out=(lse[part], losses[part], entropy[part], argmax[part]))
        ce_grad(logits, targets[part], lse[part], unit, logits)  # 1024 (p - onehot), in place
        torch.mm(logits, weight, out=grad_hidden[part])


class LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, denom):
        rows, device = hidden.shape[0], hidden.device
        lse = torch.empty(rows, dtype=torch.float32, device=device)
        losses, entropy = torch.empty_like(lse), torch.empty_like(lse)
        argmax = torch.empty(rows, dtype=torch.int64, device=device)
        unit = torch.full((1,), GRAD_SCALE, dtype=torch.float32, device=device)
        grad_hidden = torch.empty_like(hidden)
        with torch.autocast("cuda", enabled=False):  # fp16 inputs already: same matmuls
            _chunks(hidden, weight, targets, lse, losses, entropy, argmax, unit, grad_hidden)
        ctx.save_for_backward(grad_hidden, denom)
        ctx.mark_non_differentiable(losses, entropy, argmax)
        return losses.sum() / denom, losses, entropy, argmax

    @staticmethod
    def backward(ctx, grad_output, _losses, _entropy, _argmax):
        grad_hidden, denom = ctx.saved_tensors
        scale = (grad_output.float() / (denom * GRAD_SCALE)).reshape(1, 1)
        return (grad_hidden.float() * scale).to(grad_hidden.dtype), None, None, None


def linear_cross_entropy(hidden, weight, targets, denom):
    """(loss, per-row losses, entropy, argmax) of hidden @ weight^T against `targets`."""
    return LinearCrossEntropy.apply(hidden.contiguous(), weight, targets, denom)
