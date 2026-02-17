"""Smoothing operations -- pure numpy, no xarray."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def infer_dt(time: NDArray[np.float64]) -> float:
    """Infer uniform sampling interval from a 1-D time array.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
        Monotonically increasing, uniformly spaced time values.

    Returns
    -------
    float
        The sampling interval dt.

    Raises
    ------
    ValueError
        If fewer than 2 values, non-uniform spacing, or non-increasing.
    """
    values = np.asarray(time, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Need at least two coordinate values to infer dt.")
    diffs = np.diff(values)
    dt = float(diffs[0])
    if not np.allclose(diffs, dt):
        raise ValueError("Coordinate values must be uniformly spaced.")
    if dt <= 0:
        raise ValueError("Coordinate values must be increasing.")
    return dt


def make_kernel(
    method: str,
    dt: float,
    *,
    sigma: float | None = None,
    window: float | None = None,
) -> NDArray[np.float64]:
    """Build a normalized 1-D smoothing kernel.

    Parameters
    ----------
    method : str
        One of "gaussian", "boxcar" (aliases: "mean", "moving").
    dt : float
        Sampling interval in seconds.
    sigma : float or None
        Gaussian sigma in seconds. Required for method="gaussian" unless
        window is provided (sigma defaults to window/6).
    window : float or None
        Total window width in seconds. Required for method="boxcar".
        For Gaussian, defaults to 6*sigma.

    Returns
    -------
    ndarray, shape (kernel_size,)
        Normalized kernel that sums to 1.
    """
    method = method.lower()
    if method == "gaussian":
        if sigma is None and window is None:
            raise ValueError("gaussian smoothing requires sigma or window.")
        if sigma is None:
            sigma = float(window) / 6.0
        if window is None:
            window = 6.0 * float(sigma)
        sigma_bins = float(sigma) / dt
        if sigma_bins <= 0:
            return np.asarray([1.0])
        half = int(np.ceil(0.5 * float(window) / dt))
        size = max(1, 2 * half + 1)
        x = np.arange(size, dtype=float) - size // 2
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= kernel.sum()
        return kernel

    if method in ("boxcar", "mean", "moving"):
        if window is None:
            raise ValueError("boxcar smoothing requires window.")
        size = int(np.ceil(float(window) / dt))
        size = max(1, size)
        kernel = np.ones(size, dtype=float)
        kernel /= kernel.sum()
        return kernel

    raise ValueError(f"Unknown smoothing method {method!r}.")


def convolve_1d(
    y: NDArray[np.float64],
    kernel: NDArray[np.float64],
    *,
    boundary: str = "reflect",
    causal: bool = False,
) -> NDArray[np.float64]:
    """Convolve a 1-D signal with a kernel, handling boundaries.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Input signal.
    kernel : ndarray, shape (k,)
        Convolution kernel (should sum to 1 for smoothing).
    boundary : str
        Padding mode: "reflect", "nearest"/"edge", or "constant".
    causal : bool
        If True, use causal (left-sided) convolution with edge
        normalization.

    Returns
    -------
    ndarray, shape (n,)
        Smoothed signal.
    """
    if kernel.size == 1:
        return y.copy()
    if causal:
        out = np.convolve(y, kernel, mode="full")
        # Normalize by the effective kernel mass at each sample so the
        # leading edge uses an average over available history instead of
        # zero-padding.
        norm = np.convolve(np.ones(y.shape[0], dtype=float), kernel, mode="full")
        out = out[: y.shape[0]]
        norm = norm[: y.shape[0]]
        return out / np.maximum(norm, np.finfo(float).eps)
    boundary = boundary.lower()
    pad = kernel.size // 2
    if pad == 0:
        return y.copy()
    if boundary == "reflect":
        y_pad = np.pad(y, pad_width=pad, mode="reflect")
    elif boundary in ("nearest", "edge"):
        y_pad = np.pad(y, pad_width=pad, mode="edge")
    elif boundary == "constant":
        y_pad = np.pad(y, pad_width=pad, mode="constant")
    else:
        raise ValueError(f"Unknown boundary mode {boundary!r}.")
    return np.convolve(y_pad, kernel, mode="valid")


def smooth(
    data: NDArray[np.float64],
    dt: float,
    *,
    method: str = "gaussian",
    sigma: float | None = None,
    window: float | None = None,
    boundary: str = "reflect",
) -> NDArray[np.float64]:
    """Smooth dense data along the last axis (time).

    Accepts arrays of any dimensionality; smoothing is applied
    independently along the last axis. For the standard
    (trial, unit, time) layout, this means smoothing along time.

    Parameters
    ----------
    data : ndarray, shape (..., n_time)
        Dense data array. Last axis is time.
    dt : float
        Sampling interval in seconds.
    method : str
        "gaussian" or "boxcar".
    sigma : float or None
        Gaussian sigma in seconds.
    window : float or None
        Window width in seconds (boxcar) or total Gaussian window.
    boundary : str
        Boundary mode for non-causal smoothing.

    Returns
    -------
    ndarray, same shape as data
        Smoothed data.
    """
    if sigma is not None:
        method = "gaussian"
    elif method.lower() == "gaussian" and sigma is None and window is None:
        raise ValueError("Gaussian smoothing requires sigma parameter.")

    kernel = make_kernel(method, dt, sigma=sigma, window=window)
    # Boxcar uses causal (left-sided) convolution with edge normalization
    # so the moving average only looks backward in time. Gaussian uses
    # symmetric boundary padding so the kernel is centered on each sample.
    causal = method.lower() in ("boxcar", "mean", "moving")

    if kernel.size == 1:
        return data.copy()

    # Reshape to 2D: (batch, n_time), apply, reshape back.
    orig_shape = data.shape
    n_time = orig_shape[-1]
    flat = data.reshape(-1, n_time)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = convolve_1d(flat[i], kernel, boundary=boundary, causal=causal)
    return out.reshape(orig_shape)
