"""xarray <-> dataclass type conversions.

Provides conversion between :class:`~aind_ephys_utils.types.BinnedSpikes`
/ :class:`~aind_ephys_utils.types.RaggedSpikes` and ``xr.DataArray``.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..standards.conventions import C
from ..types import BinnedSpikes, RaggedSpikes


def binned_to_xarray(binned: BinnedSpikes) -> xr.DataArray:
    """Convert :class:`BinnedSpikes` to ``xr.DataArray``.

    Parameters
    ----------
    binned : BinnedSpikes

    Returns
    -------
    xr.DataArray
        Dims ``(trial, unit, time)``. Coordinates include trial/unit/time
        index coords plus all entries from ``trial_meta`` and
        ``unit_meta``.
    """
    coords: dict = {
        C.trial: np.arange(binned.n_trials),
        C.unit: np.arange(binned.n_units),
        C.time: binned.time,
    }
    # Use unit_id as the unit coordinate if available.
    if "unit_id" in binned.unit_meta:
        coords[C.unit] = binned.unit_meta["unit_id"]

    # Attach remaining metadata as non-index coords.
    for key, arr in binned.trial_meta.items():
        coords[key] = (C.trial, arr)
    for key, arr in binned.unit_meta.items():
        if key != "unit_id":
            coords[key] = (C.unit, arr)

    return xr.DataArray(
        data=binned.data,
        dims=(C.trial, C.unit, C.time),
        coords=coords,
    )


def xarray_to_binned(da: xr.DataArray) -> BinnedSpikes:
    """Convert ``xr.DataArray`` to :class:`BinnedSpikes`.

    Parameters
    ----------
    da : xr.DataArray
        Must have dims that include ``"trial"``, ``"unit"``, ``"time"``.
        Transposed to ``(trial, unit, time)`` if needed.

    Returns
    -------
    BinnedSpikes
    """
    da = da.transpose(C.trial, C.unit, C.time)
    data = da.values.astype(np.float64)
    time = np.asarray(da[C.time].values, dtype=np.float64)

    trial_meta: dict[str, np.ndarray] = {}
    unit_meta: dict[str, np.ndarray] = {}
    for name, coord in da.coords.items():
        if name in (C.trial, C.unit, C.time):
            continue
        if coord.dims == (C.trial,):
            trial_meta[str(name)] = np.asarray(coord.values)
        elif coord.dims == (C.unit,):
            unit_meta[str(name)] = np.asarray(coord.values)

    return BinnedSpikes(
        data=data,
        time=time,
        trial_meta=trial_meta,
        unit_meta=unit_meta,
    )


def ragged_to_xarray(ragged: RaggedSpikes) -> xr.DataArray:
    """Convert :class:`RaggedSpikes` to ``xr.DataArray`` (object dtype).

    Parameters
    ----------
    ragged : RaggedSpikes

    Returns
    -------
    xr.DataArray
        Dims ``(trial, unit)``, dtype=object. Each cell contains a 1-D
        float array of spike times.
    """
    data = np.empty((ragged.n_trials, ragged.n_units), dtype=object)
    for t in range(ragged.n_trials):
        for u in range(ragged.n_units):
            data[t, u] = ragged.spike_times[t][u]

    coords: dict = {
        C.trial: np.arange(ragged.n_trials),
        C.unit: np.arange(ragged.n_units),
    }
    if "unit_id" in ragged.unit_meta:
        coords[C.unit] = ragged.unit_meta["unit_id"]

    for key, arr in ragged.trial_meta.items():
        coords[key] = (C.trial, arr)
    for key, arr in ragged.unit_meta.items():
        if key != "unit_id":
            coords[key] = (C.unit, arr)

    da = xr.DataArray(data=data, dims=(C.trial, C.unit), coords=coords)
    if ragged.time_window is not None:
        da.attrs[C.attr_valid_intervals] = [ragged.time_window]
    return da


def xarray_to_ragged(da: xr.DataArray) -> RaggedSpikes:
    """Convert ``xr.DataArray`` (object dtype) to :class:`RaggedSpikes`.

    Parameters
    ----------
    da : xr.DataArray
        Must have dims ``(trial, unit)`` and ``dtype=object``.

    Returns
    -------
    RaggedSpikes
    """
    da = da.transpose(C.trial, C.unit)
    n_trials, n_units = da.shape

    spike_times: list[list[np.ndarray]] = []
    for t in range(n_trials):
        trial_spikes = []
        for u in range(n_units):
            val = da.values[t, u]
            if val is None:
                trial_spikes.append(np.array([], dtype=np.float64))
            else:
                trial_spikes.append(np.asarray(val, dtype=np.float64))
        spike_times.append(trial_spikes)

    # Extract metadata from non-index coords.
    trial_meta: dict[str, np.ndarray] = {}
    unit_meta: dict[str, np.ndarray] = {}
    for name, coord in da.coords.items():
        if name in (C.trial, C.unit):
            continue
        if coord.dims == (C.trial,):
            trial_meta[str(name)] = np.asarray(coord.values)
        elif coord.dims == (C.unit,):
            unit_meta[str(name)] = np.asarray(coord.values)

    window = None
    if C.attr_valid_intervals in da.attrs:
        intervals = da.attrs[C.attr_valid_intervals]
        if intervals:
            window = tuple(intervals[0])

    return RaggedSpikes(
        spike_times=spike_times,
        n_trials=n_trials,
        n_units=n_units,
        time_window=window,
        trial_meta=trial_meta,
        unit_meta=unit_meta,
    )
