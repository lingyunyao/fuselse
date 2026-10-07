"""
ctypes wrapper for the AVX2 SIMD IntLSE kernel.

Loads `lse_kernel_simd.so` from the same directory and exposes
NumPy-friendly LSE functions:
  - lse_intlse_simd(v):       1-D LSE -> scalar
  - lse_intlse_2d_simd(v):    2-D batched LSE -> (B,) ndarray
  - *_neg_inf(v):             native -inf-aware variants
  - logsumexp_intlse(a, axis):drop-in for scipy.special.logsumexp

No torch dependency.  Build via build.sh.
"""
import os
import ctypes
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(_HERE, "lse_kernel_simd.so")

_lib = ctypes.CDLL(_SO)
_lib.lse_q16_1d_simd.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.lse_q16_1d_simd.restype  = ctypes.c_float
_lib.lse_q16_2d_simd.argtypes = [ctypes.POINTER(ctypes.c_float),
                                 ctypes.c_int, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_float)]
_lib.lse_q16_2d_simd.restype  = None
_lib.lse_q16_1d_simd_neg_inf.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
_lib.lse_q16_1d_simd_neg_inf.restype = ctypes.c_float
_lib.lse_q16_2d_simd_neg_inf.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
_lib.lse_q16_2d_simd_neg_inf.restype = None


def _as_f32_contig(v):
    return np.ascontiguousarray(np.asarray(v, dtype=np.float32))


def lse_intlse_simd(v):
    """1-D AVX2 SIMD IntLSE.  Input: numpy float array of shape (N,).
       Returns: float (the log-sum-exp)."""
    a = _as_f32_contig(v)
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {a.shape}")
    return float(_lib.lse_q16_1d_simd(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), a.shape[0]))


def lse_intlse_2d_simd(v):
    """2-D batched AVX2 SIMD IntLSE.  Input: (B, N) float array.
       Returns: (B,) float32 ndarray, reduces over axis=-1."""
    a = _as_f32_contig(v)
    if a.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {a.shape}")
    B, N = a.shape
    out = np.empty(B, dtype=np.float32)
    _lib.lse_q16_2d_simd(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, N,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return out


def lse_intlse_simd_neg_inf(v):
    """1-D IntLSE that treats -inf as the reduction identity."""
    a = _as_f32_contig(v)
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {a.shape}")
    return float(
        _lib.lse_q16_1d_simd_neg_inf(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a.shape[0],
        )
    )


def lse_intlse_2d_simd_neg_inf(v):
    """2-D batched IntLSE that treats -inf as the reduction identity."""
    a = _as_f32_contig(v)
    if a.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {a.shape}")
    batch_size, row_size = a.shape
    out = np.empty(batch_size, dtype=np.float32)
    _lib.lse_q16_2d_simd_neg_inf(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        batch_size,
        row_size,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


def logsumexp_intlse(a, axis=None, keepdims=False):
    """Drop-in replacement for scipy.special.logsumexp / numpy LSE.
    Currently supports axis=None (full reduction) and axis=-1.
    No torch dependency."""
    a = np.asarray(a, dtype=np.float32)
    maximum = np.max(a) if a.size else -np.inf
    if np.isnan(maximum) or np.isposinf(maximum):
        if axis is None:
            result = np.logaddexp.reduce(a.ravel())
            if keepdims:
                return np.full((1,) * a.ndim, result, dtype=np.float32)
            return result
        return np.logaddexp.reduce(a, axis=axis, keepdims=keepdims)
    if axis is None:
        return lse_intlse_simd_neg_inf(a.ravel())
    if axis == -1 or axis == a.ndim - 1:
        if a.ndim == 1:
            r = lse_intlse_simd_neg_inf(a)
            return r if not keepdims else np.array([r], dtype=np.float32)
        if a.ndim == 2:
            r = lse_intlse_2d_simd_neg_inf(a)
            return r if not keepdims else r[..., None]
        flat = a.reshape(-1, a.shape[-1])
        r = lse_intlse_2d_simd_neg_inf(flat)
        out = r.reshape(a.shape[:-1])
        return out if not keepdims else out[..., None]
    raise NotImplementedError(f"axis={axis} not yet supported")


# Backward-compat aliases (so old experiment scripts importing these names work)
lse_intlse = lse_intlse_simd
HAS_SIMD = True
