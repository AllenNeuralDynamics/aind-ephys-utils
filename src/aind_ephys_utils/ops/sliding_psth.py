"""Sliding-window PSTH operation over ragged spikes.

This is the xarray "skin" over the raw-array core in
:mod:`aind_ephys_utils.spiketrain.psth`.  It unwraps a ragged spikes
DataArray, calls the pure-numpy ``sliding_window_psth`` per unit, and
rewraps the result as a dense ``(unit, time)`` firing-rate DataArray.
The skin imports the core, never the other way around.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from ..spiketrain.psth import sliding_window_psth
from ..standards.conventions import C
from ._internal.utils import (
    DataInput,
    from_dataarray_output,
    preserve_coords,
    to_dataarray_input,
)
from .bin import _ensure_1d_float_array, _infer_tlim


def sliding_psth(  # noqa: C901
    data: DataInput,
    *,
    window_size: float,
    bin_size: float = 0.001,
    window: Optional[Tuple[float, float]] = None,
    causal: bool = False,
    time_unit: str = "s",
    dims: Optional[Sequence[str]] = None,
    coords: Optional[Dict[str, object]] = None,
    return_type: str = "auto",
) -> Union[xr.DataArray, object]:
    """Sliding-window firing-rate PSTH from ragged spikes.

    Pools spikes across the ``trial`` dimension within an overlapping
    boxcar window, producing a dense firing-rate estimate (spikes/s).
    Unlike :func:`aind_ephys_utils.ops.bin` (disjoint fixed bins), the
    window slides, so adjacent time points share spikes.

    Parameters
    ----------
    data:
        Ragged spike DataArray/list or object NumPy array with dims drawn
        from ``{"trial", "unit"}``.  Spike times are relative to each
        trial's alignment event.
    window_size:
        Sliding window width in seconds.
    bin_size:
        Underlying fine bin width in seconds.
    window:
        Optional ``(tmin, tmax)`` time range.  If ``None``, taken from the
        ``ephys.valid_intervals`` attribute or inferred from the data.
    causal:
        If ``True``, use a causal (trailing) window instead of centered.
    time_unit:
        Unit for time values, recorded in attrs.
    dims:
        Optional dimension names when ``data`` is a dense NumPy array.
    coords:
        Optional coordinate mapping when constructing from dense NumPy.
    return_type:
        Output type policy: ``"auto"``, ``"xarray"``, or ``"numpy"``.

    Returns
    -------
    xr.DataArray or object
        Firing-rate PSTH with a ``time`` dimension (plus ``unit`` when the
        input has one).
    """
    da, input_kind = to_dataarray_input(data, dims=dims, coords=coords)

    if window_size <= 0:
        raise ValueError("window_size must be a positive number of seconds.")
    if da.dtype != object:
        raise ValueError(
            "sliding_psth expects ragged spikes with dtype=object."
        )
    allowed_dims = {C.trial, C.unit}
    if not set(da.dims).issubset(allowed_dims):
        raise ValueError(
            "sliding_psth expects only 'trial'/'unit' dimensions for "
            "ragged spikes."
        )

    if window is None:
        valid = da.attrs.get(C.attr_valid_intervals, None)
        if valid is not None:
            tmin, tmax = valid[0]
        else:
            tmin, tmax = _infer_tlim(da)
    else:
        tmin, tmax = window
    if tmin >= tmax:
        raise ValueError(
            f"window must be (min, max) with min < max, got {(tmin, tmax)}."
        )
    time_range = (float(tmin), float(tmax))

    def _psth_of(trials: list[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Run the core sliding-window PSTH on one unit's trials."""
        time_bins, rate, _ = sliding_window_psth(
            trials,
            window_size=window_size,
            bin_size=bin_size,
            time_range=time_range,
            causal=causal,
        )
        return time_bins, rate

    has_trial = C.trial in da.dims
    has_unit = C.unit in da.dims

    if has_trial and has_unit:
        da_tu = da.transpose(C.trial, C.unit)
        n_trials, n_units = da_tu.shape
        rows = []
        time_bins = np.empty(0)
        for j in range(n_units):
            trials = [
                _ensure_1d_float_array(da_tu.data[i, j])
                for i in range(n_trials)
            ]
            time_bins, rate = _psth_of(trials)
            rows.append(rate)
        arr = np.stack(rows, axis=0)
        out_dims: Tuple[str, ...] = (C.unit, C.time)
        out_coords = {C.unit: da_tu[C.unit], C.time: time_bins}
    elif has_unit:
        da_u = da.transpose(C.unit)
        rows = []
        time_bins = np.empty(0)
        for j in range(da_u.shape[0]):
            time_bins, rate = _psth_of([_ensure_1d_float_array(da_u.data[j])])
            rows.append(rate)
        arr = np.stack(rows, axis=0)
        out_dims = (C.unit, C.time)
        out_coords = {C.unit: da_u[C.unit], C.time: time_bins}
    elif has_trial:
        da_t = da.transpose(C.trial)
        trials = [
            _ensure_1d_float_array(da_t.data[i]) for i in range(da_t.shape[0])
        ]
        time_bins, arr = _psth_of(trials)
        out_dims = (C.time,)
        out_coords = {C.time: time_bins}
    else:
        raise ValueError(
            "sliding_psth requires at least a 'trial' or 'unit' dimension."
        )

    out = xr.DataArray(
        arr,
        dims=out_dims,
        coords=out_coords,
        name=da.name,
        attrs=dict(da.attrs),
    )
    out = preserve_coords(da, out)
    out.attrs[C.attr_kind] = "binned"
    out.attrs[C.attr_time_unit] = time_unit
    return from_dataarray_output(
        out, input_kind=input_kind, return_type=return_type
    )
