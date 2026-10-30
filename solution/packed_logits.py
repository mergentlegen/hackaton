"""The logits of the packed step, handed to TRL and Soup without being padded back.

The step computes logits only for the N real tokens, (N, vocab). TRL and Soup read them after the
loss for their metrics as a padded (batch, length, vocab) tensor: an entropy per token (masked by
the attention mask), an argmax per token for the accuracy (labels -100 are ignored), a loss per
row. Building that tensor costs a (batch * length, vocab) fp16 allocation (600+ MiB), a scatter
into it, and TRL then slices and copies it once more to shift it by one position.

`PackedLogits` stands in for that tensor. It answers exactly what those metrics ask (shape,
`[..., :-1, :]`, `.contiguous()`, `.argmax(-1)`) from the packed rows, and for anything else it
builds the real padded tensor on demand (`dense()`), so an access we did not foresee still works.
The packed logits are the same memory the loss saved for its backward, so this adds nothing.
"""

import torch


class PackedLogits:
    def __init__(self, packed, rows, batch, seq, lo=0, hi=None):
        self.packed = packed  # (N, vocab), detached
        self.rows = rows  # (N,) flat index b * seq + t of each real token
        self.batch, self.seq = batch, seq
        self.lo, self.hi = lo, seq if hi is None else hi  # window of positions still visible
        self._dense = None

    # ---- what a tensor of shape (batch, window, vocab) answers ----------------------------
    @property
    def shape(self):
        return torch.Size((self.batch, self.hi - self.lo, self.packed.shape[-1]))

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    @property
    def ndim(self):
        return 3

    def dim(self):
        return 3

    @property
    def dtype(self):
        return self.packed.dtype

    @property
    def device(self):
        return self.packed.device

    requires_grad = False

    def detach(self):
        return self

    def contiguous(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        """`logits[..., a:b, :]` or `logits[:, a:b, :]`: a window over the positions."""
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            at = key.index(Ellipsis)
            key = key[:at] + (slice(None),) * (4 - len(key)) + key[at + 1:]
        if (len(key) == 3 and key[0] == slice(None) and key[2] == slice(None)
                and isinstance(key[1], slice) and key[1].step in (None, 1)):
            window = range(self.lo, self.hi)[key[1]]
            if len(window) and window.step == 1:
                return PackedLogits(self.packed, self.rows, self.batch, self.seq, window.start,
                                    window.stop)
        return self.dense()[key]

    def argmax(self, dim=None, keepdim=False):
        if dim not in (-1, 2) or keepdim:
            return self.dense().argmax(dim=dim, keepdim=keepdim)
        best = self.packed.argmax(dim=-1)
        full = torch.full((self.batch * self.seq,), -1, dtype=best.dtype, device=best.device)
        full[self.rows] = best
        return full.view(self.batch, self.seq)[:, self.lo:self.hi]

    # ---- anything else: the real padded tensor -------------------------------------------
    def dense(self):
        if self._dense is None:
            vocab = self.packed.shape[-1]
            full = torch.zeros(self.batch * self.seq, vocab, dtype=self.packed.dtype,
                               device=self.packed.device)
            full.index_copy_(0, self.rows, self.packed)
            self._dense = full.view(self.batch, self.seq, vocab)[:, self.lo:self.hi]
        return self._dense

    def __getattr__(self, name):
        if name.startswith("__") or name in ("packed", "rows", "batch", "seq", "lo", "hi",
                                             "_dense"):
            raise AttributeError(name)
        return getattr(self.dense(), name)

    def __len__(self):
        return self.batch
