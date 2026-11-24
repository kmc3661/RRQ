"""Non-pickle scale artifacts and equivalent rescaling."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

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
