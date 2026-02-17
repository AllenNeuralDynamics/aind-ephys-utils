"""Baseline correction -- pure numpy, no xarray."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def baseline(
    data: NDArray[np.float64],
    time: NDArray[np.float64],
    window: tuple[float, float],
    *,
    mode: str = "subtract",
) -> NDArray[np.float64]:
    """Apply baseline correction over a time window.

    Operates along the last axis (time). For the standard
    ``(trial, unit, time)`` layout, this means the baseline
    statistics are computed per trial-unit pair.

    Parameters
    ----------
    data : ndarray, shape (..., n_time)
        Dense data array.
    time : ndarray, shape (n_time,)
        Time coordinate values.
    window : (float, float)
        ``(tmin, tmax)`` baseline window. Selects time points where
        ``tmin <= time <= tmax``.
    mode : str
        ``"subtract"`` (default), ``"divide"``, or ``"zscore"``.

    Returns
    -------
    ndarray, same shape as *data*
        Baseline-corrected data.

    Raises
    ------
    ValueError
        If window is invalid, produces no samples, or mode is unknown.
    """
    tmin, tmax = window
    if tmin >= tmax:
        raise ValueError(f"window must have tmin < tmax, got {window}.")

    mask = (time >= tmin) & (time <= tmax)
    if not mask.any():
        raise ValueError("Baseline window produced no samples.")

    # baseline_data shape: (..., n_baseline_bins)
    baseline_data = data[..., mask]
    mean = baseline_data.mean(axis=-1, keepdims=True)

    mode = mode.lower()
    if mode == "subtract":
        return data - mean
    elif mode == "divide":
        return data / mean
    elif mode == "zscore":
        std = baseline_data.std(axis=-1, keepdims=True)
        # Guard against division by zero.
        safe_std = np.where(std == 0, 1.0, std)
        out = (data - mean) / safe_std
        # Where std was 0, the signal is constant → baselined value is 0.
        return np.where(std == 0, 0.0, out)
    else:
        raise ValueError(f"Unknown baseline mode {mode!r}.")
