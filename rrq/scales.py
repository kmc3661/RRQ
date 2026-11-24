"""Non-pickle scale artifacts and equivalent rescaling."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

def validate_scales(module, entries):
    layers = module.model.language_model.layers
    expected = set()
    for i in range(len(layers)):
        prefix = f'model.language_model.layers.{i}.'
        expected.update(prefix + n for n in ('input_layernorm', 'post_attention_layernorm', 'mlp.up_proj'))
    if len(entries) != len(expected) or {e[0] for e in entries} != expected:
        raise ValueError('Scale file does not cover exactly three calibration sites per decoder layer')
    sites = {
        'input_layernorm': ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'),
        'post_attention_layernorm': ('mlp.gate_proj', 'mlp.up_proj'),
        'mlp.up_proj': ('mlp.down_proj',),
    }
    for previous, names, values in entries:
        layer_prefix, site = previous.split('.layers.')[0] + '.layers.', previous.split('.layers.')[1]
        index, suffix = site.split('.', 1)
        prefix = layer_prefix + index + '.'
        if tuple(names) != tuple(prefix+n for n in sites[suffix]):
            raise ValueError(f'Unexpected scale targets: {previous}')
        if values.ndim != 1 or not torch.isfinite(values).all() or not (values > 0).all():
            raise ValueError(f'Invalid scale values: {previous}')
        if any(module.get_submodule(n).in_features != values.numel() for n in names):
            raise ValueError(f'Scale width does not match model: {previous}')
    return entries

@torch.no_grad()
def apply_scales(module, entries):
    for prev_name, names, scales in entries:
        prev = module.get_submodule(prev_name)
        fcs = [module.get_submodule(n) for n in names]
        if not torch.isfinite(scales).all() or not (scales > 0).all(): raise ValueError("Invalid scales")
        if isinstance(prev, nn.Linear):
            if len(fcs) != 1: raise ValueError("FC->FC requires one target")
            s = scales.to(prev.weight.device)
            prev.weight[-s.numel():].div_(s[:, None])
            if prev.bias is not None: prev.bias[-s.numel():].div_(s)
        elif isinstance(prev, nn.LayerNorm) or "rmsnorm" in type(prev).__name__.lower():
            prev.weight.div_(scales.to(prev.weight.device))
            if getattr(prev, "bias", None) is not None: prev.bias.div_(scales.to(prev.bias.device))
        else: raise TypeError(type(prev))
        for fc in fcs: fc.weight.mul_(scales.to(fc.weight.device)[None, :])

def save(path, entries):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    meta = []
    arrays = {}
    for i, (prev, names, scale) in enumerate(entries):
        key = f"scale_{i:03d}"; meta.append({"previous": prev, "linears": list(names), "array": key})
        arrays[key] = scale.detach().float().cpu().numpy()
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps({"dtype": "bfloat16", "entries": meta}, indent=2) + "\n")

def load(path):
    path = Path(path); meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as data:
        return [(v["previous"], tuple(v["linears"]), torch.from_numpy(data[v["array"]].copy()).to(torch.bfloat16)) for v in meta["entries"]]
