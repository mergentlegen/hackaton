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

Under fp16 mixed precision, accelerate converts every fp16 tensor of the model output to fp32
(`convert_to_fp32` -> `.float()`) before the trainer sees it: for the real padded tensor that is a
(batch, length, vocab) fp32 copy, 1.5-2.4 GiB on this task. `float()` here returns the same
packed logits presented as fp32 instead: fp16 -> fp32 is exact, and every metric above already
computes in fp32 from the fp16 values, so the numbers are those of the fp32 copy.
"""

import torch


class PackedLogits:
    """`packed` (N, vocab) logits, or `source` = (hidden, weight, vocab) with `stats` = per-row
    (entropy, argmax, losses, targets) already computed by the chunked loss (lm_loss.py): then the
    logits themselves are only computed (hidden @ weight^T) if something asks for them."""

    def __init__(self, packed, rows, batch, seq, lo=0, hi=None, dtype=None, source=None,
                 stats=None):
        self._packed = packed  # (N, vocab), detached
        self.source, self.stats = source, stats
        self.rows = rows  # (N,) flat index b * seq + t of each real token
        self.batch, self.seq = batch, seq
        self.lo, self.hi = lo, seq if hi is None else hi  # window of positions still visible
        self._dtype = (packed.dtype if packed is not None else source[0].dtype) \
            if dtype is None else dtype  # the dtype it presents
        self._dense = None

    @property
    def packed(self):
        if self._packed is None:
            hidden, weight, _vocab = self.source
            with torch.no_grad():
                self._packed = torch.nn.functional.linear(hidden, weight)
        return self._packed

    def _same(self, lo, hi, dtype):
        out = PackedLogits(self._packed, self.rows, self.batch, self.seq, lo, hi, dtype,
                           self.source, self.stats)
        return out

    # ---- what a tensor of shape (batch, window, vocab) answers ----------------------------
    @property
    def shape(self):
        vocab = self.source[2] if self._packed is None else self._packed.shape[-1]
        return torch.Size((self.batch, self.hi - self.lo, vocab))

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    @property
    def ndim(self):
        return 3

    def dim(self):
        return 3

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self.rows.device

    requires_grad = False

    def detach(self):
        return self

    def contiguous(self, *args, **kwargs):
        return self

    def _as(self, dtype):
        if dtype == self._dtype:
            return self
        return self._same(self.lo, self.hi, dtype)

    def float(self):
        """fp32 view: the values are exact (see the module docstring), nothing is copied."""
        return self._as(torch.float32) if self._dtype in (torch.float16, torch.float32) \
            else self.dense().float()

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
                return self._same(window.start, window.stop, self._dtype)
        return self.dense()[key]

    def argmax(self, dim=None, keepdim=False):
        if dim not in (-1, 2) or keepdim:
            return self.dense().argmax(dim=dim, keepdim=keepdim)
        best = self.stats[1] if self.stats is not None else self.packed.argmax(dim=-1)
        full = torch.full((self.batch * self.seq,), -1, dtype=best.dtype, device=best.device)
        full[self.rows] = best
        return full.view(self.batch, self.seq)[:, self.lo:self.hi]

    # ---- anything else: the real padded tensor -------------------------------------------
    def dense(self):
        if self._dense is None:
            vocab = self.packed.shape[-1]
            full = torch.zeros(self.batch * self.seq, vocab, dtype=self._dtype,
                               device=self.packed.device)
            full.index_copy_(0, self.rows, self.packed.to(self._dtype))
            self._dense = full.view(self.batch, self.seq, vocab)[:, self.lo:self.hi]
        return self._dense

    def __getattr__(self, name):
        if name.startswith("__") or name in ("_packed", "source", "stats", "rows", "batch",
                                             "seq", "lo", "hi", "_dtype", "_dense"):
            raise AttributeError(name)
        return getattr(self.dense(), name)

    def __len__(self):
        return self.batch
