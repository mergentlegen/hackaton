"""Activation offloading to pinned host memory, overlapped with compute (README: "recomputation
or offloading in your own code" is allowed; its time counts).

The largest tensors the packed step keeps for its backward are the two fp16 (tokens, 8960) MLP
intermediates (gate and up, 36 KB per token and layer) and the fp32 inputs of the two norms of
every layer (12 KB). Together that is 60% of the activation memory. Our autograd functions
(SiluMul, RMSNorm) hand those tensors to `OFFLOADER.save()` instead of `save_for_backward`:

forward:  a tensor is copied to pinned host memory on a separate copy stream, once `KEEP` newer
          items exist (the last few stay on the GPU: the backward needs them first); the GPU
          memory is released when the copy is done (`record_stream`).
backward: `load()` returns the tensors, and starts copying the next `PREFETCH` items (in
          backward order) back to the GPU, so that they arrive while the GPU computes the
          layer in between. The compute stream waits only for the copy of what it uses.

The copies run on the GPU's copy engines, in parallel with the kernels; the values are copied
bit for bit. Host memory is a few pinned chunks reused every step (bump allocation), so pinning
happens only when a step needs more than any step before.

Only active inside the packed training step (packed.py `begin()` / `end()`), with grad enabled.
"""

import torch

KEEP = 3  # items (about one layer) left on the GPU at the end of the forward
PREFETCH = 6  # items (about two layers) copied back ahead of the backward
CHUNK = 256 << 20  # bytes of pinned host memory per chunk
ALIGN = 512


class _Item:
    __slots__ = ("index", "tensors", "host", "restored", "ready")

    def __init__(self, index, tensors):
        self.index, self.tensors = index, tensors
        self.host = self.restored = self.ready = None


class Offloader:
    def __init__(self):
        self.active = False
        self.items = []
        self.stream = None
        self.chunks = []
        self.chunk = self.offset = 0

    # ---- step boundaries (packed.py) ----------------------------------------------------
    def begin(self, device):
        self.items = []
        self.chunk = self.offset = 0
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device)
        self.active = True

    def end(self):
        self.active = False

    def wants(self, *tensors):
        """Offload these (for a backward that will need them)?"""
        return self.active and torch.is_grad_enabled() and all(t.is_cuda for t in tensors)

    # ---- forward ------------------------------------------------------------------------
    def save(self, *tensors):
        item = _Item(len(self.items), tensors)
        self.items.append(item)
        if len(self.items) > KEEP:
            self._to_host(self.items[-KEEP - 1])
        return item

    def _host_buffer(self, like):
        nbytes = -(-like.numel() * like.element_size() // ALIGN) * ALIGN
        while True:
            if self.chunk < len(self.chunks):
                chunk = self.chunks[self.chunk]
                if self.offset + nbytes <= chunk.numel():
                    view = chunk[self.offset:self.offset + nbytes]
                    self.offset += nbytes
                    return view[:like.numel() * like.element_size()].view(like.dtype).view(
                        like.shape)
                self.chunk, self.offset = self.chunk + 1, 0
                continue
            self.chunks.append(torch.empty(max(nbytes, CHUNK), dtype=torch.uint8,
                                           pin_memory=True))

    def _to_host(self, item):
        if item.tensors is None:
            return
        compute = torch.cuda.current_stream()
        produced = compute.record_event()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(produced)
            host = []
            for tensor in item.tensors:
                buffer = self._host_buffer(tensor)
                buffer.copy_(tensor, non_blocking=True)
                tensor.record_stream(self.stream)  # its memory is free once the copy is done
                host.append(buffer)
        item.host, item.tensors = host, None

    # ---- backward -----------------------------------------------------------------------
    def _to_device(self, item):
        if item.host is None or item.restored is not None:
            return
        compute = torch.cuda.current_stream()
        restored = [torch.empty(h.shape, dtype=h.dtype, device=compute.device) for h in item.host]
        free = compute.record_event()  # the memory just allocated is free from here on
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(free)
            for gpu, host in zip(restored, item.host):
                gpu.copy_(host, non_blocking=True)
            item.ready = self.stream.record_event()
        item.restored = restored

    def load(self, item):
        if item.tensors is not None:  # never left the GPU
            tensors, item.tensors = item.tensors, None
            return tensors
        self._to_device(item)
        for index in range(item.index - 1, max(item.index - 1 - PREFETCH, -1), -1):
            self._to_device(self.items[index])
        torch.cuda.current_stream().wait_event(item.ready)
        tensors, item.restored, item.host = item.restored, None, None
        return tensors


OFFLOADER = Offloader()
