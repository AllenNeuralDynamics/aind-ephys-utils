"""Smoothing operations for binned or continuous signals."""

from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from ..core.conventions import C


def smooth(
    da: xr.DataArray,
    *,
    dim: str = C.time,
    method: str = "gaussian",
    sigma: Optional[float] = None,
    window: Optional[float] = None,
    boundary: str = "reflect",
) -> xr.DataArray:
    """
    Smooth a signal along a dimension.

    Parameters
    ----------
    da:
        Input DataArray.
    dim:
        Dimension to smooth (defaults to time).
    method:
        Smoothing method (e.g. "gaussian").
    sigma:
        Gaussian sigma in seconds (optional).
    window:
        Window size in seconds (optional).
    boundary:
        Boundary handling mode.
    """
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not found in DataArray dims.")
    if dim not in da.coords:
        raise ValueError(f"dim {dim!r} has no coordinate values.")

    dt = _infer_dt(da[dim].values)
    kernel = _make_kernel(
        method=method,
        sigma=sigma,
        window=window,
        dt=dt,
    )

    out = xr.apply_ufunc(
        _convolve_1d,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        kwargs={"kernel": kernel, "boundary": boundary},
    )
    out.attrs = dict(da.attrs)
    return out


def _infer_dt(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Need at least two coordinate values to infer dt.")
    diffs = np.diff(values)
    dt = float(diffs[0])
    if not np.allclose(diffs, dt):
        raise ValueError("Coordinate values must be uniformly spaced.")
    if dt <= 0:
        raise ValueError("Coordinate values must be increasing.")
    return dt


def _make_kernel(
    *,
    method: str,
    sigma: Optional[float],
    window: Optional[float],
    dt: float,
) -> np.ndarray:
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


def _convolve_1d(
    y: np.ndarray, *, kernel: np.ndarray, boundary: str
) -> np.ndarray:
    if kernel.size == 1:
        return y
    boundary = boundary.lower()
    pad = kernel.size // 2
    if pad == 0:
        return y
    if boundary == "reflect":
        y_pad = np.pad(y, pad_width=pad, mode="reflect")
    elif boundary in ("nearest", "edge"):
        y_pad = np.pad(y, pad_width=pad, mode="edge")
    elif boundary == "constant":
        y_pad = np.pad(y, pad_width=pad, mode="constant")
    else:
        raise ValueError(f"Unknown boundary mode {boundary!r}.")
    return np.convolve(y_pad, kernel, mode="valid")
