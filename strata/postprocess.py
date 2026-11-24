"""Post-processing of generated motion (Supp. C "sampling"): light head smoothing (sigma_h = 0.75 frames) and
re-projection with Pi_delta.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from strata import decoder as _decoder
from strata import network as _network


SIGMA_H = 0.75  # sigma_h in frames (Supp. C, sampling)


HEAD = slice(_network.TRANS.start, _network.ROT.stop)


def gs6(r):
    a, b = r[..., :3], r[..., 3:]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (a * b).sum(-1, keepdims=True) * a
    return np.concatenate([a, b / np.linalg.norm(b, axis=-1, keepdims=True)], -1)


def smooth_head(X, sigma=SIGMA_H):
    """Head smoothing after sampling (Supp. C "sampling"): Gaussian with sigma_h = 0.75 frames along time on the
    head translation and 6D rotation columns, then rotation re-orthonormalised; blendshapes and gaze untouched."""
    Y = np.array(X, dtype=np.float64, copy=True)
    Y[..., HEAD] = gaussian_filter1d(Y[..., HEAD], sigma, axis=-2, mode="nearest")
    Y[..., _network.ROT] = gs6(Y[..., _network.ROT])
    return Y.astype(np.asarray(X).dtype)


def reproject(X, proj):
    """Re-projection after head smoothing (Supp. C "sampling"): every sample projected again with Pi_delta."""
    X = np.asarray(X)
    return np.stack([_decoder.Pi_delta(np.asarray(x, np.float64), proj) for x in X]).astype(X.dtype)
