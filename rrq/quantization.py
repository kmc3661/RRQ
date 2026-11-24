"""Final-format reference math; BF16 intermediate roundings are intentional."""
from dataclasses import dataclass
import torch

@dataclass
class WeightEncoding:
    codes: torch.Tensor
    multiplier: torch.Tensor
    offset: torch.Tensor
    row_scale: torch.Tensor
    value: torch.Tensor
    group_value: torch.Tensor
    def packed(self):
        return (self.codes[:, 0::2] | (self.codes[:, 1::2] << 4)).contiguous()

@torch.no_grad()
def hadamard(x, block_size=32):
    if block_size != 32 or x.shape[-1] % 32:
        raise ValueError("The released final setting uses independent 32-element blocks")
    v = x.float().reshape(-1, 32)
    for shift in (1, 2, 4, 8, 16):
        a = v.reshape(-1, 32 // (2 * shift), 2, shift)
        lo, hi = a[:, :, 0, :], a[:, :, 1, :]
        v = torch.stack((lo + hi, lo - hi), dim=2).reshape(-1, 32)
    return (v * 0.1767766952966369).to(x.dtype).reshape_as(x)

@torch.no_grad()
def group_weight(w, group_size=128):
    if w.dtype != torch.bfloat16 or w.ndim != 2 or w.shape[-1] % group_size:
        raise ValueError("Expected BF16 [output, input] with divisible input width")
    x = w.reshape(-1, group_size)
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = (hi - lo).clamp(min=1e-5) / 15
    zero = (-torch.round(lo / scale)).clamp(0, 15)
    codes = (torch.round(x / scale) + zero).clamp(0, 15).to(torch.uint8)
    value = ((codes.to(w.dtype) - zero) * scale).reshape_as(w)
    return codes.reshape_as(w), scale, zero, value

@torch.no_grad()
def encode_weight(w, iterations=3):
    if iterations not in (0, 3):
        raise ValueError("Use 3 for the final method, 0 for the refit control")
    _, _, _, g4 = group_weight(w)
    maximum = g4.abs().amax(-1).clamp_min(1e-5)
    c = (maximum / 127).to(torch.bfloat16).half()
    n, k = w.shape
    target = (g4.float() / c.float()[:, None]).reshape(n, -1, 128)
    lo, hi = target.amin(-1), target.amax(-1)
    mult = ((hi - lo) / 15).round().clamp(1, 17)
    offset = ((lo + hi - 15 * mult) / 2).round().clamp_min(-128).minimum(127 - 15 * mult)
    def codes(m, b):
        return ((target - b[:, :, None]) / m[:, :, None]).round().clamp(0, 15)
    q = codes(mult, offset)
    bestq, bestm, bestb = q.to(torch.uint8), mult, offset
    besterr = (q * mult[:, :, None] + offset[:, :, None] - target).square().sum(-1)
    for _ in range(iterations):
        qm, tm = q.mean(-1), target.mean(-1)
        var = (q.square().mean(-1) - qm.square()).clamp_min(1e-12)
        pm = ((q * target).mean(-1) - qm * tm).div(var).round().clamp(1, 17)
        pb = (tm - pm * qm).round().clamp_min(-128).minimum(127 - 15 * pm)
        pq = codes(pm, pb)
        pe = (pq * pm[:, :, None] + pb[:, :, None] - target).square().sum(-1)
        accept = pe < besterr
        bestq = torch.where(accept[:, :, None], pq.to(torch.uint8), bestq)
        bestm, bestb = torch.where(accept, pm, bestm), torch.where(accept, pb, bestb)
        besterr = torch.minimum(besterr, pe)
        q, mult, offset = bestq.float(), bestm, bestb
    integer = (bestq.int() * bestm.int()[:, :, None] + bestb.int()[:, :, None]).reshape(n, k)
    if not (integer.min() >= -128 and integer.max() <= 127):
        raise ValueError("Integer weight exceeded the signed-byte correction bound")
    value = (integer.float() * c.float()[:, None]).to(w.dtype)
    return WeightEncoding(bestq.reshape(n, k), bestm.to(torch.int8), bestb.to(torch.int8), c, value, g4)

@torch.no_grad()
def encode_activation(v, bits=4):
    if bits not in (4, 6, 8) or v.dtype != torch.bfloat16 or v.shape[-1] % 128:
        raise ValueError("Expected BF16 A4/A6/A8, input width divisible by 128")
    v = v.reshape(-1, v.shape[-1])
    x = v.float().reshape(len(v), -1, 128)
    limit = (1 << (bits - 1)) - 1
    local = x.abs().amax(-1).clamp_min(1e-5) / limit
    a = (local.amax(-1) / 8).clamp_min(2 ** -24)
    h = a.half()
    anchor = torch.where(h.float() < a, torch.nextafter(h, torch.full_like(h, float("inf"))), h)
    mu = (local / anchor.float()[:, None]).ceil().clamp(1, 8).to(torch.uint8)
    step = anchor.float()[:, None] * mu.float()
    q = (x / step[:, :, None]).bfloat16().float().round().clamp(-(1 << (bits - 1)), limit).to(torch.int8).reshape_as(v)
    return q, anchor, mu

@torch.no_grad()
def unpack_activation4(packed, width):
    q = torch.stack([(packed >> j) & 15 for j in range(0, 32, 4)], -1).reshape(-1, width)
    return torch.where(q >= 8, q - 16, q).to(torch.int8)

@torch.no_grad()
def integer_linear(q, weight, anchor, mu):
    """Transparent integer-dot oracle. Never use for speed claims."""
    m, k = q.shape
    w = (weight.codes.int().reshape(weight.codes.shape[0], -1, 128) *
         weight.multiplier.int()[:, :, None] + weight.offset.int()[:, :, None])
    total = torch.zeros((m, w.shape[0]), dtype=torch.int64, device=q.device)
    for g in range(k // 128):
        # FP32 dot is exact for these small integer products; accumulate in INT64.
        part = q[:, g*128:(g+1)*128].float() @ w[:, g].float().T
        total += part.to(torch.int64) * mu[:, g, None].long()
    return (total.float() * (anchor.float()[:, None] * weight.row_scale.float()[None, :])).half().bfloat16()
