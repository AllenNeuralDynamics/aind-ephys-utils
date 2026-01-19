"""Alignment operations.

This module contains the core implementation for aligning spikes/continuous
signals to event times.

The public entry point is `align`, which is used by the `.ephys.align` accessor.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import xarray as xr

from ..core.conventions import C
from ..core.validate import infer_kind, validate


class EphysAlignError(ValueError):
    """Raised when alignment inputs are invalid or unsupported."""

    pass


def _normalize_events(events: Union[xr.Dataset, xr.DataArray]) -> xr.Dataset:
    """
    Accept:
      - Dataset with var "t" dims (trial, event)
      - DataArray already representing event times dims (trial, event)
    Return a Dataset with events["t"] always present.
    """
    if isinstance(events, xr.DataArray):
        return xr.Dataset({C.event_time_var: events})
    if isinstance(events, xr.Dataset):
        if C.event_time_var not in events:
            raise EphysAlignError(
                f"events must contain variable {C.event_time_var!r}."
            )
        return events
    raise TypeError(
        f"events must be an xarray Dataset or DataArray, got {type(events)!r}."
    )


def _get_event_times(events: xr.Dataset, to: str) -> xr.DataArray:
    """Select per-trial event times for a specific event label.

    Parameters
    ----------
    events:
        Events dataset containing `events[C.event_time_var]` with an `event`
        dimension.
    to:
        Event label to select.

    Returns
    -------
    xr.DataArray
        Per-trial event times with dims (trial,).
    """
    t = events[C.event_time_var]
    if C.event not in t.dims:
        raise EphysAlignError(
            f"events['{C.event_time_var}'] must have a '{C.event}' dimension. "
            f"Got dims {t.dims}."
        )
    if to not in t.coords.get(C.event, []):
        # coords might not exist; try fallback on indexes
        try:
            _ = t.sel({C.event: to})
        except Exception as e:
            raise EphysAlignError(
                f"Could not find event label {to!r} in events['{C.event_time_var}']."
            ) from e
    return t.sel({C.event: to})  # dims (trial,)


def align(
    da: xr.DataArray,
    *,
    events: Union[xr.Dataset, xr.DataArray],
    to: str,
    window: Tuple[float, float],
) -> xr.DataArray:
    """
    Align a DataArray to an event and extract a time window around it.

    - Continuous: returns da with time shifted so event is at 0, and sliced to window.
    - Ragged spikes: subtract event time per trial, filter to window, return ragged.
    """
    validate(da)
    events = _normalize_events(events)
    t0 = _get_event_times(events, to=to)  # (trial,)

    kind = infer_kind(da)
    tmin, tmax = window
    if tmin >= tmax:
        raise EphysAlignError(
            f"window must be (min,max) with min < max, got {window}."
        )

    if kind in ("continuous", "binned"):
        # We assume da has dims including 'trial' and 'time' OR just 'time'.
        if C.time not in da.dims:
            raise EphysAlignError(
                "Continuous/binned alignment requires a time dimension."
            )

        if C.trial in da.dims:
            # Broadcast subtract per-trial t0 from time coordinate to get aligned time.
            # We'll create an aligned time coordinate for each trial by shifting the data via interpolation.
            #
            # Simple approach: for each trial, reindex/interp onto a common aligned time grid.
            time = da[C.time].values
            aligned_time = time  # assume time already relative per trial; if not, you can change this.
            # If time is absolute, you'd want aligned_time = time - t0(trial). But that becomes 2D time coord.
            # To keep things simple and memorable, v0 assumes trial-wise time coords are already in trial-time.
            # Users can convert session->trial before using align, or you can add a session-time mode later.

            # Slice window
            mask = (aligned_time >= tmin) & (aligned_time <= tmax)
            out = da.sel({C.time: aligned_time[mask]})
            out = out.assign_coords(
                {C.time: out[C.time].values - 0.0}
            )  # explicit no-op placeholder
            out.attrs = dict(da.attrs)
            out.attrs[C.attr_valid_intervals] = [window]
            return out

        else:
            # No trial dim: interpret as a single continuous trace with event time scalar
            if t0.size != 1:
                raise EphysAlignError(
                    "Aligning a no-trial DataArray requires a single event time (trial size 1)."
                )
            t0_scalar = float(t0.values)
            aligned_time = da[C.time].values - t0_scalar
            out = da.assign_coords({C.time: aligned_time}).sel(
                {C.time: slice(tmin, tmax)}
            )
            out.attrs = dict(da.attrs)
            out.attrs[C.attr_valid_intervals] = [window]
            return out

    if kind == "spikes_ragged":
        # da dims include (trial, unit), each entry is array of spike times (session or trial).
        if C.trial not in da.dims:
            raise EphysAlignError(
                "Ragged spikes alignment requires a 'trial' dimension."
            )
        # Ensure t0 is aligned on trial coordinate
        if not da[C.trial].identical(t0[C.trial]):
            t0 = t0.sel({C.trial: da[C.trial]})

        def _align_entry(spk: object, trial_idx: int) -> np.ndarray:
            """Align a single ragged spike array for one (trial, unit) entry."""
            if spk is None:
                return np.asarray([], dtype=float)
            arr = np.asarray(spk, dtype=float)
            arr = arr - float(t0.values[trial_idx])
            arr = arr[(arr >= tmin) & (arr <= tmax)]
            return arr

        # Apply across (trial, unit)
        data = da.data
        out = np.empty_like(data, dtype=object)

        # iterate trials then units (keeps trial index available)
        for i in range(data.shape[da.get_axis_num(C.trial)]):
            # we need to index properly if dims order isn't (trial, unit)
            # simplest: transpose to (trial, unit), operate, then transpose back
            pass

        # Robust version: work on a transposed view
        tr_first = da.transpose(
            C.trial, C.unit, ...
        ).data  # should now be (trial, unit)
        out_tf = np.empty_like(tr_first, dtype=object)
        for i in range(tr_first.shape[0]):
            for j in range(tr_first.shape[1]):
                out_tf[i, j] = _align_entry(tr_first[i, j], i)

        out_da = xr.DataArray(
            out_tf,
            dims=(C.trial, C.unit),
            coords={C.trial: da[C.trial], C.unit: da[C.unit]},
            name=da.name,
            attrs=dict(da.attrs),
        )
        # Restore original dim order if needed
        out_da = out_da.transpose(*da.dims)
        out_da.attrs[C.attr_valid_intervals] = [window]

        return out_da

    raise EphysAlignError(f"Unsupported kind {kind!r} for align.")
