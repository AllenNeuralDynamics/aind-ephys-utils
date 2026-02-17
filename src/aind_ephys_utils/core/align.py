"""Spike alignment -- pure numpy, no xarray."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def align_unit(
    spike_times: NDArray[np.float64],
    anchor_times: NDArray[np.float64],
    window: tuple[float, float],
) -> list[NDArray[np.float64]]:
    """Align one unit's spikes to per-trial anchor times.

    Uses vectorized ``searchsorted`` to find spike boundaries for all
    trials simultaneously, then slices per trial.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Sorted spike times for a single unit (session time).
    anchor_times : ndarray, shape (n_trials,)
        Per-trial anchor times (e.g. stimulus onset times).
    window : (float, float)
        ``(tmin, tmax)`` relative to anchor. E.g. ``(-0.5, 1.0)``.

    Returns
    -------
    list of ndarray
        Length ``n_trials``. Each element is a 1-D float array of spike
        times relative to the anchor.
    """
    tmin, tmax = window
    n_trials = len(anchor_times)

    if spike_times is None:
        return [np.array([], dtype=float) for _ in range(n_trials)]

    arr = np.asarray(spike_times, dtype=float)
    if arr.size == 0:
        return [np.array([], dtype=float) for _ in range(n_trials)]

    # Vectorized searchsorted: find all boundaries at once.
    lo_bounds = anchor_times + tmin
    hi_bounds = anchor_times + tmax
    all_lo = np.searchsorted(arr, lo_bounds)
    all_hi = np.searchsorted(arr, hi_bounds)

    # Slice and shift for each trial (can't vectorize due to ragged output).
    result = []
    for i in range(n_trials):
        result.append(arr[all_lo[i] : all_hi[i]] - anchor_times[i])

    return result


def align(
    spike_times: list[NDArray[np.float64]],
    anchor_times: NDArray[np.float64],
    window: tuple[float, float],
) -> list[list[NDArray[np.float64]]]:
    """Align spikes for all units to per-trial anchor times.

    Parameters
    ----------
    spike_times : list of ndarray
        Length ``n_units``. ``spike_times[u]`` is a sorted 1-D float
        array of spike times for unit *u* (session time).
    anchor_times : ndarray, shape (n_trials,)
        Per-trial anchor times.
    window : (float, float)
        ``(tmin, tmax)`` relative to anchor.

    Returns
    -------
    list of list of ndarray
        Outer list has length ``n_trials``, inner list has length
        ``n_units``. Each element is a 1-D float array of aligned
        spike times.
    """
    tmin, tmax = window
    if tmin >= tmax:
        raise ValueError(f"window must be (tmin, tmax) with tmin < tmax, got {window}.")

    anchor_times = np.asarray(anchor_times, dtype=float)
    n_units = len(spike_times)
    n_trials = len(anchor_times)

    # Align per unit: unit_results[u] is list of n_trials arrays.
    unit_results = [
        align_unit(spike_times[u], anchor_times, window) for u in range(n_units)
    ]

    # Transpose to [trial][unit].
    result: list[list[NDArray[np.float64]]] = [
        [unit_results[u][t] for u in range(n_units)] for t in range(n_trials)
    ]
    return result
