"""Activation offloading to pinned host memory, overlapped with compute (README: "recomputation
or offloading in your own code" is allowed; its time counts).

Everything the decoder layers of the packed step keep for their backward goes to host memory:
- our autograd functions hand their tensors to `OFFLOADER.save()` instead of `save_for_backward`
  (SiluMul: the fp16 MLP gate and up; RMSNorm: the fp32 input; PackedAttention: the q, k, v
  projections, attention itself is recomputed in the backward);
- every other tensor autograd saves inside the layers (the LoRA dropout masks and the dropped-out
  inputs of lora_A) goes through `pack` / `unpack` (torch.autograd.graph.saved_tensors_hooks).

forward:  an item is copied to pinned host memory on a separate copy stream once `KEEP_BYTES` of
          newer items exist (the newest stay on the GPU: the backward needs them first); its GPU
          memory is released when the copy is done (`record_stream`).
backward: `load()` returns an item and keeps `PREFETCH_BYTES` of the next items (in backward
          order) on their way back, so they arrive while the GPU computes; the compute stream
          only waits for the copy of what it uses.

T4 on Kaggle: 13 GB/s each way, and the copies run next to the matmuls without slowing them
(tools/pcie_bench.py). Values are copied bit for bit. Host memory is a few pinned chunks reused
every step (bump allocation): pinning happens only when a step needs more than any before.
"""

import torch

KEEP_BYTES = 96 << 20  # newest bytes left on the GPU at the end of the forward
PREFETCH_BYTES = 160 << 20  # bytes on their way back ahead of the backward
MIN_BYTES = 256 << 10  # smaller saved tensors stay where they are
CHUNK = 256 << 20  # bytes of pinned host memory per chunk
ALIGN = 512


class _Item:
    __slots__ = ("index", "nbytes", "tensors", "host", "restored", "ready")

    def __init__(self, index, tensors):
        self.index, self.tensors = index, tensors
        self.nbytes = sum(t.numel() * t.element_size() for t in tensors)
        self.host = self.restored = self.ready = None


class Offloader:
    def __init__(self):
        self.active = False
        self.items = []
        self.stream = None
        self.chunks = []
        self.chunk = self.offset = 0
        self.evict_next = self.resident = 0
        self.fetch_next = -1

    # ---- step boundaries (packed.py) ----------------------------------------------------
    def begin(self, device):
        self.items = []
        self.chunk = self.offset = 0
        self.evict_next = self.resident = 0
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device)
        self.active = True

    def end(self):
        self.active = False
        self.fetch_next = len(self.items) - 1

    def wants(self, *tensors):
        """Offload these (for a backward that will need them)?"""
        return self.active and torch.is_grad_enabled() and all(t.is_cuda for t in tensors)

    # ---- saved_tensors_hooks ------------------------------------------------------------
    def pack(self, tensor):
        if (not self.active or not tensor.is_cuda or isinstance(tensor, torch.nn.Parameter)
                or tensor.numel() * tensor.element_size() < MIN_BYTES):
            return tensor
        return self.save(tensor)

    def unpack(self, packed):
        return self.load(packed)[0] if isinstance(packed, _Item) else packed

    # ---- forward ------------------------------------------------------------------------
    def save(self, *tensors):
        item = _Item(len(self.items), tensors)
        self.items.append(item)
        self.resident += item.nbytes
        while self.resident > KEEP_BYTES and self.evict_next < len(self.items) - 1:
            old = self.items[self.evict_next]
            self.evict_next += 1
            self.resident -= old.nbytes
            self._to_host(old)
        return item

    def _host_buffer(self, like):
        size = like.numel() * like.element_size()
        nbytes = -(-size // ALIGN) * ALIGN
        while True:
            if self.chunk < len(self.chunks):
                chunk = self.chunks[self.chunk]
                if self.offset + nbytes <= chunk.numel():
                    view = chunk[self.offset:self.offset + size]
                    self.offset += nbytes
                    return view.view(like.dtype).view(like.shape)
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
        if item.index <= self.fetch_next + 1:
            self.fetch_next = min(self.fetch_next, item.index - 1)
        ahead = sum(other.nbytes for other in self.items[self.fetch_next + 1:item.index]
                    if other.restored is not None)
        while self.fetch_next >= 0 and ahead < PREFETCH_BYTES:
            other = self.items[self.fetch_next]
            self.fetch_next -= 1
            if other.host is not None and other.restored is None:
                self._to_device(other)
                ahead += other.nbytes
        torch.cuda.current_stream().wait_event(item.ready)
        tensors, item.restored, item.host = item.restored, None, None
        return tensors


OFFLOADER = Offloader()
