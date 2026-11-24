"""Portable source builds. No machine-specific binaries or library paths."""
import os, sys
from pathlib import Path
from functools import lru_cache
import torch

@lru_cache(None)
def extension(kind):
    from torch.utils.cpp_extension import load
    if kind not in {"rotation", "activation", "activation4", "matmul4"}:
        raise ValueError(kind)
    if not torch.cuda.is_available():
        raise RuntimeError("Optimized low-bit execution requires an NVIDIA CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 6):
        raise RuntimeError("Released optimized kernel policies are qualified for SM86. Requalify kernels on other architectures; use the reference backend for portability.")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("MAX_JOBS", "1")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    source = Path(__file__).parent / "csrc" / kind
    return load(name="rrq_" + kind,
                sources=[str(source / "bind.cpp"), str(source / "kernel.cu")],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--fmad=false"],
                verbose=False)

def encode(x, bits=8, rotate=True):
    if bits not in (4, 8):
        raise ValueError("Optimized activation encoding supports A4/A8, not INT6 throughput")
    return tuple(extension("activation").encode(x.contiguous(), bits, rotate, False))

def encode_packed4(x, rotate=True):
    return tuple(extension("activation4").encode(x.contiguous(), 4, rotate, False))
