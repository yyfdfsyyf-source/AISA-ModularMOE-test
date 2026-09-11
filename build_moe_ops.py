"""JIT 编译 moe_ops 扩展（CPU 版）。
GTX 1060 上如需 CUDA：安装 CUDA toolkit 后设 USE_CUDA=1，并在 .cu 中启用内核。
"""
import os
import torch
from torch.utils.cpp_extension import load

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
sources = [os.path.join(src_dir, "moe_ops.cpp")]
use_cuda = os.environ.get("USE_CUDA", "0") == "1"
extra_cflags = ["-O3", "-fopenmp"]
extra_ldflags = ["-fopenmp"]
if use_cuda and torch.cuda.is_available():
    sources.append(os.path.join(src_dir, "moe_ops.cu"))
    extra_cflags.append("-DUSE_CUDA")
    print("[build] CUDA builds enabled")
else:
    print("[build] CPU-only build (no CUDA toolchain in this env / USE_CUDA not set)")

moe_ops = load(
    name="moe_ops",
    sources=sources,
    extra_cflags=extra_cflags,
    extra_ldflags=extra_ldflags,
    extra_cuda_cflags=["-O3"] if use_cuda else [],
    verbose=False,
)