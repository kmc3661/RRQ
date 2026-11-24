"""FP-fixed population diagnostics. No selection of response-only Text tokens."""
import torch

def newly_zeroed(original, quantized):
    """Vector-average percent: only originally nonzero elements in denominator."""
    x, q = original.reshape(-1, original.shape[-1]), quantized.reshape(-1, original.shape[-1])
    valid = x.ne(0)
    count = valid.sum(-1)
    keep = count > 0
    if not keep.any(): raise ValueError("No vectors with nonzero inputs")
    frac = (valid & q.eq(0)).sum(-1)[keep].double() / count[keep]
    return float(100 * frac.mean())

def text_mask(valid, vision):
    if valid.shape != vision.shape: raise ValueError("Mask shape mismatch")
    return valid.bool() & ~vision.bool()

def nmse(original, quantized):
    den = original.double().square().sum()
    if den == 0: raise ValueError("Zero-energy input")
    return float((original.double()-quantized.double()).square().sum()/den)

def symmetric_group(x, bits, group_size):
    view = x.reshape(-1, group_size)
    scale = view.abs().amax(-1, keepdim=True).clamp_min(1e-5) / ((1 << (bits-1)) - 1)
    codes = (view/scale).round().clamp(-(1 << (bits-1)), (1 << (bits-1))-1)
    return (codes*scale).reshape_as(x)
