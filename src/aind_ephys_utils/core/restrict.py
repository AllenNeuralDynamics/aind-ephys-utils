"""Restrict operations -- pure numpy, no xarray."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def restrict_dense(
    data: NDArray[np.float64],
    time: NDArray[np.float64],
    window: tuple[float, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Restrict dense data to a time window.

    Parameters
    ----------
    data : ndarray, shape (..., n_time)
        Dense data with time along the last axis.
    time : ndarray, shape (n_time,)
        Time coordinate values.
    window : (float, float)
        ``(tmin, tmax)`` interval to keep (inclusive).

    Returns
    -------
    data_restricted : ndarray, shape (..., n_kept)
        Restricted data.
    time_restricted : ndarray, shape (n_kept,)
        Restricted time values.
    """
    tmin, tmax = window
    # Unlike baseline/bin/align which require positive-width windows,
    # restrict allows tmin == tmax to select a single time point.
    if tmin > tmax:
        raise ValueError(f"window must have tmin <= tmax, got {window}.")
    mask = (time >= tmin) & (time <= tmax)
    return data[..., mask], time[mask]


def restrict_ragged(
    spike_times: list[list[NDArray[np.float64]]],
    window: tuple[float, float],
) -> list[list[NDArray[np.float64]]]:
    """Restrict ragged spikes to a time window.

    Parameters
    ----------
    spike_times : list of list of ndarray
        ``spike_times[trial][unit]`` is a sorted 1-D float array.
    window : (float, float)
        ``(tmin, tmax)`` interval to keep.

    Returns
    -------
    list of list of ndarray
        Same structure, with spikes outside window removed.
    """
    tmin, tmax = window
    if tmin > tmax:
        raise ValueError(f"window must have tmin <= tmax, got {window}.")
    result = []
    for trial_spikes in spike_times:
        trial_result = []
        for arr in trial_spikes:
            arr = np.asarray(arr, dtype=float)
            left = np.searchsorted(arr, tmin, side="left")
            right = np.searchsorted(arr, tmax, side="right")
            trial_result.append(arr[left:right])
        result.append(trial_result)
    return result
