"""Our NF4 decode against bitsandbytes' (bit for bit), and the 4-bit linear against bnb's."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def quantized(rows, cols, nested):
    from bitsandbytes import functional as F

    torch.manual_seed(0)
    weight = (torch.randn(rows, cols, device="cuda") * 0.02).half()
    packed, state = F.quantize_4bit(weight, blocksize=64, compress_statistics=nested,
                                    quant_type="nf4")
    return packed, state


def flat(state):
    """What packed.py does once: the double-quantized scales decoded to plain fp32."""
    from bitsandbytes import functional as F

    if state.nested:
        absmax = F.dequantize_blockwise(state.absmax, state.state2)
        absmax += state.offset
        state.absmax, state.nested = absmax.float(), False
    return state


@pytest.mark.parametrize("rows,cols,nested", [(256, 1536, False), (1536, 8960, True),
                                              (8960, 1536, True), (100, 192, False)])
def test_decode_is_bit_identical(rows, cols, nested):
    from bitsandbytes import functional as F

    from solution.nf4 import can_dequantize, dequantize_nf4

    packed, state = quantized(rows, cols, nested)
    state = flat(state)
    assert can_dequantize(packed, state)
    reference = F.dequantize_4bit(packed, state)
    mine = dequantize_nf4(packed, state)
    assert mine.dtype == reference.dtype and mine.shape == reference.shape
    assert torch.equal(mine, reference)


def test_linear_matches_bitsandbytes_forward_and_backward():
    import bitsandbytes as bnb

    from solution.nf4 import nf4_linear

    packed, state = quantized(512, 1536, True)
    state = flat(state)
    bias = (torch.randn(512, device="cuda") * 0.1).half()
    x = torch.randn(300, 1536, device="cuda").half().requires_grad_()
    grad = torch.randn(300, 512, device="cuda").half()

    ref = bnb.matmul_4bit(x, packed.t(), quant_state=state, bias=bias)
    ref.backward(grad)
    ref_grad = x.grad.clone()
    x.grad = None
    out = nf4_linear(x, packed.t(), bias, state)
    out.backward(grad)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, ref_grad, rtol=2e-3, atol=2e-3)


def test_stacked_gate_up_matches_two_bitsandbytes_linears_and_swiglu():
    import bitsandbytes as bnb

    from solution.kernels import silu_mul, silu_mul_packed
    from solution.nf4 import can_stack, nf4_stacked_linear

    pg, sg = quantized(1024, 1536, True)
    pu, su = quantized(1024, 1536, True)
    pu = pu.clone()
    sg, su = flat(sg), flat(su)
    torch.manual_seed(1)
    su.absmax = su.absmax * 1.5  # a different second weight
    assert can_stack(pg, sg, pu, su)
    x = torch.randn(300, 1536, device="cuda").half().requires_grad_()
    grad = torch.randn(300, 1024, device="cuda").half()

    ref = silu_mul(bnb.matmul_4bit(x, pg.t(), quant_state=sg),
                   bnb.matmul_4bit(x, pu.t(), quant_state=su))
    ref.backward(grad)
    ref_grad = x.grad.clone()
    x.grad = None
    out = silu_mul_packed(nf4_stacked_linear(x, pg, sg, pu, su))
    out.backward(grad)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    cos = torch.nn.functional.cosine_similarity(x.grad.float().flatten(), ref_grad.float().flatten(),
                                                dim=0)
    assert cos > 0.9999
    torch.testing.assert_close(x.grad, ref_grad, rtol=1e-2, atol=1e-2)
