"""Compute backends: numpy (CPU) and cupy (GPU).

Import-time: ensure that pip-installed nvidia-* CUDA libraries are on the
loader path, so that cupy finds cuBLAS / cuSOLVER / cuFFT without the user
having to set LD_LIBRARY_PATH manually.
"""
from __future__ import annotations
import os
import sys
import numpy as _np


def _setup_cuda_libpath() -> None:
    """Preload the pip-installed nvidia CUDA runtime shared libraries.

    LD_LIBRARY_PATH mutation at import time does not affect the current
    process — ctypes.CDLL with the explicit file path does.  This makes
    `cupy` work in an environment where CUDA was installed via pip
    (nvidia-cublas-cu12, nvidia-cusolver-cu12, ...).
    """
    if sys.platform != "linux":
        return
    try:
        import nvidia
    except Exception:
        return
    bases = list(getattr(nvidia, "__path__", []))
    if not bases:
        return

    import ctypes
    # order matters: cusolver depends on cublas etc.
    to_load = [
        ("nvjitlink",     "libnvJitLink.so.12"),
        ("cuda_nvrtc",    "libnvrtc.so.12"),
        ("cuda_runtime",  "libcudart.so.12"),
        ("cublas",        "libcublasLt.so.12"),
        ("cublas",        "libcublas.so.12"),
        ("cusparse",      "libcusparse.so.12"),
        ("cusolver",      "libcusolver.so.11"),
        ("cufft",         "libcufft.so.11"),
    ]
    for pkg, sofile in to_load:
        for base in bases:
            path = os.path.join(base, pkg, "lib", sofile)
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


_setup_cuda_libpath()


def get_backend(prefer_gpu: bool = False):
    """Return (array module, on_gpu_bool).

    If prefer_gpu is True and cupy is importable, returns cupy; else numpy.
    """
    if prefer_gpu:
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            return cp, True
        except Exception as exc:
            print(f"  [tbexciton90.backends] cupy unavailable ({exc}); falling back to numpy")
    return _np, False


def to_numpy(x):
    """Move a possibly-GPU array back to numpy for I/O and downstream tools."""
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return _np.asarray(x)
