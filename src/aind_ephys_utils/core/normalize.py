"""Normalization operations -- pure numpy, no xarray."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Fixed mapping from semantic names to axis indices.
# Convention: data is always (trial, unit, time).
# The xarray layer uses ``dim=`` (xarray convention); this layer uses
# ``across=`` to avoid confusion with numpy axis integers.
AXIS_MAP: dict[str, int] = {"trials": 0, "units": 1, "time": 2}


def normalize(
    data: NDArray[np.float64],
    *,
    across: str | tuple[str, ...] = "trials",
    method: str = "zscore",
) -> NDArray[np.float64]:
    """Normalize data across one or more axes.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_units, n_time)
        Dense data array.
    across : str or tuple of str
        Axis name(s) to normalize across. Valid names: ``"trials"``,
        ``"units"``, ``"time"``, or a tuple combining them.
    method : str
        ``"zscore"``, ``"minmax"``, or ``"robust"``.

    Returns
    -------
    ndarray, same shape as *data*
        Normalized data. Zero-variance slices are set to 0.
    """
    if isinstance(across, str):
        across = (across,)

    try:
        axes = tuple(AXIS_MAP[a] for a in across)
    except KeyError:
        bad = [a for a in across if a not in AXIS_MAP]
        raise ValueError(
            f"Unknown axis name(s): {bad}. Valid names: {list(AXIS_MAP.keys())}."
        )

    method = method.lower()
    if method == "zscore":
        mean = np.mean(data, axis=axes, keepdims=True)
        std = np.std(data, axis=axes, keepdims=True)
        safe_std = np.where(std == 0, 1.0, std)
        out = (data - mean) / safe_std
        return np.where(std == 0, 0.0, out)
    elif method == "minmax":
        vmin = np.min(data, axis=axes, keepdims=True)
        vmax = np.max(data, axis=axes, keepdims=True)
        denom = vmax - vmin
        safe_denom = np.where(denom == 0, 1.0, denom)
        out = (data - vmin) / safe_denom
        return np.where(denom == 0, 0.0, out)
    elif method == "robust":
        q25 = np.quantile(data, 0.25, axis=axes, keepdims=True)
        q75 = np.quantile(data, 0.75, axis=axes, keepdims=True)
        median = np.quantile(data, 0.5, axis=axes, keepdims=True)
        iqr = q75 - q25
        safe_iqr = np.where(iqr == 0, 1.0, iqr)
        out = (data - median) / safe_iqr
        return np.where(iqr == 0, 0.0, out)
    else:
        raise ValueError(f"Unknown normalization method {method!r}.")
