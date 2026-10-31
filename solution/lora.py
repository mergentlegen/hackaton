"""The LoRA branch of q_proj / v_proj as one autograd function.

peft's 4-bit LoRA layer (peft.tuners.lora.bnb.Linear4bit) computes, for every layer and step,

    result = base(x).clone()          # a "defensive" copy of the (tokens, out) base output
    result = result + B(A(dropout(x))) * scaling

as six separate autograd ops (clone, dropout, two linears with their autocast weight casts, the
scaling multiply, the add), each with its own backward node, twice per decoder layer. Here the
same arithmetic is: the base output without the copy, the very same dropout module (so the
random masks are those of the reference run, drawn from the same generator in the same order),
then one function that does z = x_d @ A^T and out = result + scaling * (z @ B^T) (one addmm, the
scaling applied inside it), and its backward by hand with the same fp16 matmuls:

    d result = g                              d z   = (scaling * g) @ B
    d B      = (scaling * g)^T @ z            d A   = (d z)^T @ x_d
    d x_d    = d z @ A

The adapter weights get their gradients in their own dtype, like autocast's cast backward gives.
Anything else (DoRA, several adapters, merged weights, no autocast) goes through peft.
"""

import torch


class LoraBranch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, result, x_dropped, weight_a, weight_b, scaling):
        dtype = result.dtype
        a, b = weight_a.to(dtype), weight_b.to(dtype)
        rows = x_dropped.reshape(-1, x_dropped.shape[-1])
        z = rows @ a.t()
        out = torch.addmm(result.reshape(-1, result.shape[-1]), z, b.t(), alpha=scaling)
        ctx.save_for_backward(rows, z, weight_a, weight_b)
        ctx.scaling, ctx.shapes = scaling, (x_dropped.shape, result.shape)
        return out.view(result.shape)

    @staticmethod
    def backward(ctx, grad):
        rows, z, weight_a, weight_b = ctx.saved_tensors
        x_shape, out_shape = ctx.shapes
        dtype = z.dtype
        a, b = weight_a.to(dtype), weight_b.to(dtype)
        g = grad.reshape(-1, out_shape[-1])
        g_scaled = g * ctx.scaling
        grad_b = (g_scaled.t() @ z).to(weight_b.dtype)
        grad_z = g_scaled @ b
        grad_a = (grad_z.t() @ rows).to(weight_a.dtype)
        grad_x = (grad_z @ a).view(x_shape)
        return grad, grad_x, grad_a, grad_b, None


def _autocast_cuda():
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:  # older torch
        return torch.is_autocast_enabled()


def fused_forward(layer, x):
    """peft's forward for the plain case, through LoraBranch; None when it does not apply."""
    if (layer.disable_adapters or layer.merged or not _autocast_cuda()
            or not isinstance(x, torch.Tensor) or not x.is_cuda):
        return None
    adapters = [name for name in layer.active_adapters if name in layer.lora_A.keys()]
    if len(adapters) != 1:
        return None
    name = adapters[0]
    lora_a, lora_b = layer.lora_A[name], layer.lora_B[name]
    dropout = layer.lora_dropout[name]
    if (name in getattr(layer, "lora_variant", {}) or getattr(layer, "use_dora", {}).get(name, False)
            or not isinstance(lora_a, torch.nn.Linear) or not isinstance(lora_b, torch.nn.Linear)
            or lora_a.bias is not None or lora_b.bias is not None
            or not isinstance(dropout, (torch.nn.Dropout, torch.nn.Identity))):
        return None
    result = layer.base_layer(x)
    if result.dtype not in (torch.float16, torch.bfloat16):
        return None  # (never in the task: the 4-bit base returns the fp16 compute dtype)
    return LoraBranch.apply(result, dropout(x), lora_a.weight, lora_b.weight,
                            float(layer.scaling[name]))
