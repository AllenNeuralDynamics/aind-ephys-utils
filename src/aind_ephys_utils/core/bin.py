"""Binning operations -- pure numpy, no xarray."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_bin_edges(
    dt: float,
    window: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute bin edges and centers for a given window and bin width.

    Parameters
    ----------
    dt : float
        Bin width in seconds. Must be positive.
    window : (float, float)
        ``(tmin, tmax)`` time window.

    Returns
    -------
    edges : ndarray, shape (n_bins + 1,)
        Bin edges.
    centers : ndarray, shape (n_bins,)
        Bin centers.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    tmin, tmax = window
    if tmin >= tmax:
        raise ValueError(f"window must have tmin < tmax, got {window}.")

    ratio = (tmax - tmin) / dt
    if np.isclose(ratio, round(ratio)):
        n_bins = int(round(ratio))
    else:
        n_bins = int(np.ceil(ratio))
    n_bins = max(1, n_bins)

    edges = tmin + dt * np.arange(n_bins + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def bin_spikes(
    spike_times: list[list[NDArray[np.float64]]],
    dt: float,
    window: tuple[float, float],
    *,
    output: str = "rate",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin ragged spikes into a dense (trial, unit, time) array.

    Parameters
    ----------
    spike_times : list of list of ndarray
        ``spike_times[trial][unit]`` is a 1-D float array of spike
        times.
    dt : float
        Bin width in seconds.
    window : (float, float)
        ``(tmin, tmax)`` time window for binning.
    output : str
        ``"rate"`` (spikes/s) or ``"count"`` (raw counts).

    Returns
    -------
    data : ndarray, shape (n_trials, n_units, n_bins)
        Binned spike data.
    centers : ndarray, shape (n_bins,)
        Time bin centers.

    Raises
    ------
    ValueError
        If *dt* <= 0, window is invalid, or *output* is not recognized.
    """
    output = output.lower()
    if output not in ("rate", "count"):
        raise ValueError("output must be 'rate' or 'count'.")

    edges, centers = make_bin_edges(dt, window)

    n_trials = len(spike_times)
    n_units = len(spike_times[0]) if n_trials > 0 else 0
    n_bins = len(centers)

    data = np.zeros((n_trials, n_units, n_bins), dtype=np.float64)
    for t in range(n_trials):
        for u in range(n_units):
            arr = _ensure_1d_float(spike_times[t][u])
            if arr.size > 0:
                data[t, u], _ = np.histogram(arr, bins=edges)

    if output == "rate":
        data = data / dt

    return data, centers


def _ensure_1d_float(x: object) -> NDArray[np.float64]:
    """Coerce a spike-time entry to a 1-D float array."""
    if x is None:
        return np.array([], dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(
            f"Spike time entries must be 0-D or 1-D, got shape {arr.shape}."
        )
    return arr
