import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from .quantization import hadamard, encode_weight, encode_activation, integer_linear
from .shared_input import SharedInput

class QuantizedLinear(nn.Module):
    """Final W4/A4,A6,A8 format. Native execution is qualified for A4/A8."""
    def __init__(self, fc, bits=4, backend="reference", model="qwen3", refit_iterations=3):
        super().__init__()
        if fc.bias is not None:
            raise ValueError("The released decoder Linear scope is bias-free")
        if backend not in ("reference", "accuracy", "native") or bits not in (4, 6, 8):
            raise ValueError((backend, bits))
        if backend == "native" and bits == 6:
            raise ValueError("A6 accuracy uses logical signed-six codes; no native INT6 timing claim")
        self.in_features, self.out_features = fc.in_features, fc.out_features
        self.bits, self.backend, self.cache, self.calls = bits, backend, None, 0
        enc = encode_weight(hadamard(fc.weight.detach()), iterations=refit_iterations)
        for name, value in [("packed_weight", enc.packed()), ("mult", enc.multiplier.T.contiguous().to(torch.uint8)),
                            ("offset", enc.offset.T.contiguous()), ("row_scale", enc.row_scale)]:
            self.register_buffer(name, value)
        if backend == "reference":
            # Only the debugging/oracle backend retains an expanded representation.
            self.register_buffer("codes", enc.codes)
        if backend == "native" and bits == 4:
            from .native import extension
            self.ext = extension("matmul4")
            rules = json.loads((Path(__file__).parent / "policies" / (model + ".json")).read_text())["shapes"]
            self.rule = rules.get(f"{self.out_features}x{self.in_features}", {"decode": {"backend": "legacy"}})
            self.register_buffer("native_row_scale", (enc.row_scale.float() * 128).half())
            if not torch.equal(self.native_row_scale.float() / 128, enc.row_scale.float()):
                raise ValueError("A4 row-scale normalization overflowed")

    def act_quant(self, x):
        x = x.contiguous().reshape(-1, self.in_features)
        if self.backend == "native":
            from . import native
            return native.encode_packed4(x, True) if self.bits == 4 else native.encode(x, self.bits, True)
        return encode_activation(hadamard(x), self.bits)

    @torch.no_grad()
    def forward(self, x):
        q, a, mu = self.cache.quantized(self, x) if self.cache is not None else self.act_quant(x)
        if self.backend == "reference":
            weight = SimpleNamespace(codes=self.codes, multiplier=self.mult.T, offset=self.offset.T, row_scale=self.row_scale)
            y = integer_linear(q, weight, a, mu)
        else:
            import triton
            m, n, k = q.shape[0], self.out_features, self.in_features
            y = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
            if self.bits == 8 or self.backend == "accuracy":
                from .kernels.gemm8 import _gemm
                from .kernels.gemv8 import _gemv
                if m == 1:
                    _gemv[(triton.cdiv(n, 8),)](q, self.packed_weight, self.mult, self.offset, self.row_scale, a, mu, y, n, k, 8, 8, num_warps=4, enable_fp_fusion=False)
                else:
                    bm = 64 if m >= 64 else 32
                    _gemm[(triton.cdiv(m, bm), triton.cdiv(n, 64))](q, self.packed_weight, self.mult, self.offset, self.row_scale, a, mu, y, m, n, k, bm, 64, num_warps=4, num_stages=2, enable_fp_fusion=False)
            else:
                from .kernels.matmul4 import _packed_gemv
                if m == 1 and self.rule.get("decode", {}).get("backend") == "parallel_packed4":
                    rule = self.rule["decode"]; bn, bg = rule["BN"], rule["BG"]
                    _packed_gemv[(triton.cdiv(n, bn),)](q, self.packed_weight, self.mult, self.offset, self.native_row_scale, a, mu, y, n, k, bn, bg, num_warps=4, enable_fp_fusion=False)
                else:
                    tn = self.rule.get("decode", {}).get("TN", 0) if m == 1 else (8 if m >= 64 else 4)
                    choices = self.rule.get("prefill", [])
                    if m > 1 and choices: tn = min(choices, key=lambda v: abs(v["M"] - m))["TN"]
                    y = self.ext.gemm(q, self.packed_weight, self.mult, self.offset, self.native_row_scale, a, mu, tn, False)[0]
        self.calls += 1
        return y.reshape(*x.shape[:-1], self.out_features)

PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

def install_layer(layer, bits=4, backend="reference", model="qwen3"):
    for path in PROJECTIONS:
        parent, name = path.rsplit(".", 1)
        owner = layer.get_submodule(parent); fc = getattr(owner, name)
        if not isinstance(fc, nn.Linear): raise TypeError((path, type(fc)))
        setattr(owner, name, QuantizedLinear(fc, bits, backend, model))
    for owner, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj")), (layer.mlp, ("gate_proj", "up_proj"))):
        cache = SharedInput(len(names))
        for name in names: getattr(owner, name).cache = cache
    return len(PROJECTIONS)

def install(model, bits=4, backend="native", name="qwen3"):
    decoder = model.model.language_model
    if len(decoder.layers) != 36:
        raise ValueError("Released policies are qualified for the three dense 36-layer 8B models")
    return sum(install_layer(layer, bits, backend, name) for layer in decoder.layers)
