"""pandas/polars -> xarray ingestion for ephys analysis.

This module implements `from_dataframe`, an adapter that converts one or two
DataFrames into canonical xarray objects.

Supported patterns
------------------
1) `from_dataframe(units_df)`:
   Build a ragged spikes DataArray with dims (unit, trial) where trial=[0] and
   spike times are expressed in session time.

2) `from_dataframe(trials_df)`:
   If the DataFrame does not look like a units/spikes table, interpret it as a
   trials/events table and build an events DataArray with dims (trial, event,
   bound).

3) `from_dataframe(units_df, trials_df)`:
   Segment per-unit session spike times by trial boundaries and express spikes
   in trial time relative to an anchor (by default, the values in the
   `start_time` column).

The resulting DataArrays are validated by `aind_ephys_utils.standards.validate` when
`validate=True`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ..standards.conventions import C
from ..standards.validate import validate as validate_xarray
from ..ops.align import align as ops_align
from ..ops.bin import bin as ops_bin


class FromDataFrameError(ValueError):
    """Raised when `from_dataframe` cannot interpret or validate the inputs."""

    pass


# -----------------------------
# Helpers
# -----------------------------
def _get_ids(df: pd.DataFrame, id_col: Optional[str], kind: str) -> np.ndarray:
    """
    Default to index unless id_col is provided.
    """
    if id_col is None:
        try:
            return df.index.to_numpy()
        except AttributeError:  # polars has no index
            return np.arange(len(df))
    if id_col not in df.columns:
        raise FromDataFrameError(
            f"{kind}_id_col={id_col!r} not found in DataFrame columns."
        )
    return df[id_col].to_numpy()


def _default_coords(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Return column names to attach as coordinates.

    Parameters
    ----------
    df:
        Source DataFrame.
    exclude:
        Column names to exclude (ids, spike times, event columns, etc.).

    Returns
    -------
    list[str]
        Remaining column names.
    """
    return [c for c in df.columns if c not in exclude]


def _ensure_1d_float_array(x: Any, *, ctx: str) -> np.ndarray:
    """Convert a spike-time entry into a 1D float array.

    Parameters
    ----------
    x:
        Array-like spike times in seconds, or None.
    ctx:
        Context string used to make error messages more actionable.

    Returns
    -------
    np.ndarray
        1D float array of spike times.
    """
    if x is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise FromDataFrameError(
            f"{ctx}: expected 1D array, got shape {arr.shape}."
        )
    return arr


def _infer_wide_event_cols(
    trials_df: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    """
    Infer:
      - event_cols: {"go_cue": "go_cue_time"} from *_time columns
      - epoch_cols: {"delay": ("delay_start","delay_end")} from *_start/_end pairs
    """
    cols = list(trials_df.columns)
    event_cols: Dict[str, str] = {}
    epoch_cols: Dict[str, Tuple[str, str]] = {}

    starts = {c[:-6]: c for c in cols if c.endswith("_start")}
    ends = {c[:-4]: c for c in cols if c.endswith("_end")}
    for base, sc in starts.items():
        if base in ends:
            epoch_cols[base] = (sc, ends[base])

    for c in cols:
        if c.endswith("_time"):
            name = c[:-5]
            if name in epoch_cols:
                continue
            event_cols[name] = c

    return event_cols, epoch_cols


def _col_to_float_numpy(df, col):
    """Return column as float numpy array with coercion."""
    if col not in df.columns:
        raise KeyError(f"{col} not in DataFrame")

    s = df[col]

    # ---- pandas ----
    try:
        import pandas as pd

        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    except ImportError:
        pass

    # ---- polars ----
    try:
        import polars as pl

        if isinstance(s, pl.Series):
            return (
                s.cast(
                    pl.Float64, strict=False
                )  # coercion like errors="coerce"
                .to_numpy()
                .astype(float)
            )
    except ImportError:
        pass

    # ---- fallback (numpy / list-like) ----
    return np.asarray(s, dtype=float)


def _infer_window_from_time(
    time: np.ndarray, *, bin_size: float
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Validate time centers and derive the corresponding binning window."""
    centers = np.asarray(time, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise FromDataFrameError(
            "time must be a 1D array with at least 2 points."
        )
    dt = float(centers[1] - centers[0])
    if not np.allclose(np.diff(centers), dt):
        raise FromDataFrameError(
            "time vector must be uniformly spaced for binning."
        )
    if not np.isclose(dt, bin_size):
        raise FromDataFrameError(
            "time spacing must match bin_size when time is provided."
        )
    window = (centers[0] - dt / 2, centers[-1] + dt / 2)
    return centers, window


def _get_cell(df, row_idx: int, col: str):
    """Returns a cell from a Pandas or Polars DataFrame"""
    # pandas
    if hasattr(df, "iloc"):
        return df.iloc[row_idx][col]
    # polars
    return df.row(row_idx, named=True)[col]


def _infer_valid_interval(da: xr.DataArray) -> Tuple[float, float]:
    """Infer a (tmin, tmax) interval from ragged spike entries."""
    tmin = np.inf
    tmax = -np.inf
    for x in da.data.ravel():
        if x is None:
            continue
        arr = np.asarray(x, dtype=float)
        if arr.size == 0:
            continue
        tmin = min(tmin, float(arr.min()))
        tmax = max(tmax, float(arr.max()))
    if not np.isfinite(tmin) or not np.isfinite(tmax):
        raise FromDataFrameError(
            "Cannot infer valid interval from empty spikes."
        )
    if tmin == tmax:
        tmax = tmin + 1e-6
    return tmin, tmax


def _looks_like_units_df(df: pd.DataFrame, spike_times_col: str) -> bool:
    """Heuristic check for whether a DataFrame looks like a units/spikes table."""
    return spike_times_col in df.columns


def _build_events(  # noqa: C901
    trials_df: pd.DataFrame,
    *,
    trial_id_col: Optional[str],
    event_cols: Optional[Dict[str, str]],
    epoch_cols: Optional[Dict[str, Tuple[str, str]]],
    long_event_col: Optional[str],
    long_time_col: Optional[str],
    long_end_time_col: Optional[str],
    trial_coords: Optional[List[str]],
    time_unit: str,
    timebase: str,
) -> xr.DataArray:
    """
    Canonical events/epochs representation:
      events: DataArray dims (trial, event, bound)
      bound ∈ {"start","end"} ; instantaneous events have start==end
    """
    use_long = long_event_col is not None and long_time_col is not None
    if not use_long and event_cols is None and epoch_cols is None:
        event_cols, epoch_cols = _infer_wide_event_cols(trials_df)

    event_cols = event_cols or {}
    epoch_cols = epoch_cols or {}

    if use_long:
        if (
            long_event_col not in trials_df.columns
            or long_time_col not in trials_df.columns
        ):
            raise FromDataFrameError(
                "Long-format events requires columns "
                f"{long_event_col!r} and {long_time_col!r}."
            )

        # Get unique trial IDs in order of first appearance
        trial_ids = _get_ids(trials_df, trial_id_col, "trial")
        unique_trial_ids = []
        seen_trials = set()
        for tid in trial_ids:
            # Handle potential non-hashable types
            tid_key = tid if np.isscalar(tid) else tuple(tid) if hasattr(tid, '__iter__') else tid
            if tid_key not in seen_trials:
                unique_trial_ids.append(tid)
                seen_trials.add(tid_key)
        unique_trial_ids = np.asarray(unique_trial_ids)
        n_trials = len(unique_trial_ids)
        
        # Create mapping from trial_id to index
        trial_to_idx = {}
        for i, tid in enumerate(unique_trial_ids):
            tid_key = tid if np.isscalar(tid) else tuple(tid) if hasattr(tid, '__iter__') else tid
            trial_to_idx[tid_key] = i

        # Preserve event label order
        seen_events = set()
        event_names: List[str] = []
        for v in trials_df[long_event_col].astype(str).to_numpy():
            if v not in seen_events:
                event_names.append(v)
                seen_events.add(v)

        data = np.full(
            (n_trials, len(event_names), 2), np.nan, dtype=float
        )
        ev_to_i = {ev: i for i, ev in enumerate(event_names)}

        t_start = _col_to_float_numpy(trials_df, long_time_col)
        if long_end_time_col and long_end_time_col in trials_df.columns:
            t_end = _col_to_float_numpy(trials_df, long_end_time_col)
        else:
            t_end = t_start

        # Build mapping from trial_id to first row index for metadata extraction
        trial_id_to_first_row = {}
        all_trial_ids = _get_ids(trials_df, trial_id_col, "trial")
        for r in range(len(trials_df)):
            tid = all_trial_ids[r]
            tid_key = tid if np.isscalar(tid) else tuple(tid) if hasattr(tid, '__iter__') else tid
            if tid_key not in trial_id_to_first_row:
                trial_id_to_first_row[tid_key] = r
            ti = trial_to_idx[tid_key]
            ev = str(_get_cell(trials_df, r, long_event_col))
            ei = ev_to_i[ev]
            data[ti, ei, 0] = t_start[r]
            data[ti, ei, 1] = t_end[r]
        
        # Override trial_ids with unique ones for the DataArray
        trial_ids = unique_trial_ids

    else:
        if not event_cols and not epoch_cols:
            raise FromDataFrameError(
                "Could not infer any events/epochs. Provide event_cols/epoch_cols, "
                "or include columns like *_time or *_start/_end."
            )

        event_names = list(event_cols.keys()) + list(epoch_cols.keys())
        data = np.full(
            (len(trials_df), len(event_names), 2), np.nan, dtype=float
        )

        # instantaneous events
        for k, name in enumerate(event_cols.keys()):
            col = event_cols[name]
            ts = _col_to_float_numpy(trials_df, col)
            data[:, k, 0] = ts
            data[:, k, 1] = ts

        # epochs
        off = len(event_cols)
        for j, name in enumerate(epoch_cols.keys()):
            sc, ec = epoch_cols[name]
            ts = _col_to_float_numpy(trials_df, sc)
            te = _col_to_float_numpy(trials_df, ec)
            data[:, off + j, 0] = ts
            data[:, off + j, 1] = te
        
        # Get trial_ids for wide format
        trial_ids = _get_ids(trials_df, trial_id_col, "trial")

    events = xr.DataArray(
        data,
        dims=(C.trial, C.event, "bound"),
        coords={
            C.trial: trial_ids,
            C.event: np.asarray(event_names, dtype=object),
            "bound": np.asarray(["start", "end"], dtype=object),
        },
        name="events",
    )

    events.attrs[C.attr_time_unit] = time_unit
    events.attrs[C.attr_timebase] = timebase
    events.attrs[C.attr_kind] = "events"
    events.attrs[C.attr_history] = ["from_dataframe(events)"]

    # attach trial metadata as coords
    if trial_coords is None:
        exclude = set()
        if trial_id_col is not None:
            exclude.add(trial_id_col)
        exclude |= set(event_cols.values())
        for sc, ec in epoch_cols.values():
            exclude.add(sc)
            exclude.add(ec)
        if long_event_col:
            exclude.add(long_event_col)
        if long_time_col:
            exclude.add(long_time_col)
        if long_end_time_col:
            exclude.add(long_end_time_col)
        trial_coords = _default_coords(trials_df, exclude)

    # Extract metadata for each unique trial
    if use_long:
        # For long format, extract metadata from first row of each trial
        for c in trial_coords:
            coord_values = np.empty(len(trial_ids), dtype=object)
            for i, tid in enumerate(trial_ids):
                tid_key = tid if np.isscalar(tid) else tuple(tid) if hasattr(tid, '__iter__') else tid
                row_idx = trial_id_to_first_row[tid_key]
                coord_values[i] = _get_cell(trials_df, row_idx, c)
            events = events.assign_coords({c: (C.trial, coord_values)})
    else:
        # For wide format, use all rows directly
        for c in trial_coords:
            events = events.assign_coords({c: (C.trial, trials_df[c].to_numpy())})

    return events


def _build_spikes_units_only(
    units_df: pd.DataFrame,
    *,
    spike_times_col: str,
    unit_id_col: Optional[str],
    unit_coords: Optional[List[str]],
    time_unit: str,
) -> xr.DataArray:
    """
    Return ragged (unit, trial) with trial=[0].
    Spike times are *session time*.
    """
    if spike_times_col not in units_df.columns:
        raise FromDataFrameError(
            f"units_df must contain {spike_times_col!r} column."
        )

    unit_ids = _get_ids(units_df, unit_id_col, "unit")
    n_units = len(units_df)

    data = np.empty((n_units, 1), dtype=object)
    for ui in range(n_units):
        spk = _ensure_1d_float_array(
            _get_cell(units_df, ui, spike_times_col),
            ctx=f"unit {ui} spike_times",
        )
        data[ui, 0] = spk

    da = xr.DataArray(
        data,
        dims=(C.unit, C.trial),
        coords={C.unit: unit_ids, C.trial: np.asarray([0])},
        name="spikes",
    )
    da.attrs[C.attr_time_unit] = time_unit
    da.attrs[C.attr_timebase] = "session"
    da.attrs[C.attr_kind] = "spikes_ragged"
    da.attrs[C.attr_history] = ["from_dataframe(units_only)"]

    if unit_coords is None:
        exclude = set()
        if unit_id_col is not None:
            exclude.add(unit_id_col)
        exclude.add(spike_times_col)
        unit_coords = _default_coords(units_df, exclude)

    for c in unit_coords:
        da = da.assign_coords({c: (C.unit, units_df[c].to_numpy())})

    da.attrs[C.attr_valid_intervals] = [_infer_valid_interval(da)]
    return da


def _build_spikes_with_trials(  # noqa: C901
    units_df: pd.DataFrame,
    trials_df: pd.DataFrame,
    *,
    spike_times_col: str,
    unit_id_col: Optional[str],
    trial_id_col: Optional[str],
    trial_start_col: str,
    trial_end_col: str,
    align_to: Optional[str],  # anchor column in trials_df; default start_time
    unit_coords: Optional[List[str]],
    trial_coords: Optional[List[str]],
    bin_size: Optional[float],
    window: Optional[Tuple[float, float]],
    time: Optional[np.ndarray],
    time_unit: str,
) -> xr.DataArray:
    """
    units_df + trials_df as input
    - no bin_size -> ragged (unit, trial) relative to anchor (default trial_start)
    - bin_size    -> dense  (unit, trial, time) binned relative to anchor
      within window/time
    Spike times are assumed to be *session time* in units_df, segmented by
    [trial_start, trial_end].
    """
    if spike_times_col not in units_df.columns:
        raise FromDataFrameError(
            f"units_df must contain {spike_times_col!r} column."
        )
    if trial_start_col not in trials_df.columns:
        raise FromDataFrameError(
            f"trials_df must contain trial_start_col={trial_start_col!r}."
        )
    if trial_end_col not in trials_df.columns:
        if (
            trial_end_col == C.trial_end_col
            and "stop_time" in trials_df.columns
        ):
            trial_end_col = "stop_time"
        else:
            raise FromDataFrameError(
                f"trials_df must contain trial_end_col={trial_end_col!r}."
            )

    da_session = _build_spikes_units_only(
        units_df,
        spike_times_col=spike_times_col,
        unit_id_col=unit_id_col,
        unit_coords=unit_coords,
        time_unit=time_unit,
    )

    trial_ids = _get_ids(trials_df, trial_id_col, "trial")

    t_start = _col_to_float_numpy(trials_df, trial_start_col)
    t_end = _col_to_float_numpy(trials_df, trial_end_col)

    if np.any(~np.isfinite(t_start)) or np.any(~np.isfinite(t_end)):
        raise FromDataFrameError(
            "trial start/end contains NaNs; fill/drop before ingestion."
        )
    if np.any(t_end < t_start):
        raise FromDataFrameError("Found trials with trial_end < trial_start.")

    if align_to is None:
        anchor = t_start
        anchor_name = trial_start_col
    else:
        if align_to not in trials_df.columns:
            raise FromDataFrameError(
                f"align_to={align_to!r} not found in trials_df."
            )
        anchor = _col_to_float_numpy(trials_df, align_to)
        if np.any(~np.isfinite(anchor)):
            raise FromDataFrameError(
                f"align_to={align_to!r} contains NaNs; fill/drop before ingestion."
            )
        anchor_name = align_to

    events = xr.DataArray(
        anchor,
        dims=(C.trial,),
        coords={C.trial: trial_ids},
    ).expand_dims({C.event: np.asarray([anchor_name], dtype=object)})

    rel_start = t_start - anchor
    rel_end = t_end - anchor
    window_all = (float(rel_start.min()), float(rel_end.max()))
    window_use = window if window is not None else window_all

    if bin_size is None:
        da = ops_align(
            da_session, events=events, to=anchor_name, window=window_use
        )
        da.attrs[C.attr_kind] = "spikes_ragged"
    else:
        if time is not None:
            centers, window_use = _infer_window_from_time(
                time, bin_size=bin_size
            )
        else:
            if window is None:
                raise FromDataFrameError(
                    "bin_size specified but window is None."
                )
            window_use = window
            centers = None

        aligned = ops_align(
            da_session, events=events, to=anchor_name, window=window_use
        )
        da = ops_bin(
            aligned,
            dt=bin_size,
            window=window_use,
            output="rate",
            time_unit=time_unit,
        )
        if centers is not None:
            da = da.assign_coords({C.time: centers})

    # Attach metadata as coords
    if unit_coords is None:
        exclude = {spike_times_col}
        if unit_id_col is not None:
            exclude.add(unit_id_col)
        unit_coords = _default_coords(units_df, exclude)

    for c in unit_coords:
        da = da.assign_coords({c: (C.unit, units_df[c].to_numpy())})

    if trial_coords is None:
        exclude = {trial_start_col, trial_end_col}
        if trial_id_col is not None:
            exclude.add(trial_id_col)
        if align_to is not None:
            exclude.add(align_to)
        trial_coords = _default_coords(trials_df, exclude)

    for c in trial_coords:
        da = da.assign_coords({c: (C.trial, trials_df[c].to_numpy())})

    da.attrs[C.attr_time_unit] = time_unit
    da.attrs[C.attr_timebase] = "trial"
    da.attrs[C.attr_history] = [
        f"from_dataframe(units+trials, anchor={anchor_name})"
    ]

    return da


# -----------------------------
# Public API
# -----------------------------
def from_dataframe(
    *dfs: pd.DataFrame,
    # IDs: default to index unless explicitly provided
    unit_id_col: Optional[str] = None,
    trial_id_col: Optional[str] = None,
    # Spikes schema
    spike_times_col: str = C.spike_times_col,
    # Trials schema for segmentation
    trial_start_col: str = C.trial_start_col,
    trial_end_col: str = C.trial_end_col,
    align_to: Optional[str] = None,  # anchor per trial; default trial_start
    # Events schema (only used when trials_df is the ONLY df passed)
    event_cols: Optional[Dict[str, str]] = None,
    epoch_cols: Optional[Dict[str, Tuple[str, str]]] = None,
    long_event_col: Optional[str] = None,
    long_time_col: Optional[str] = None,
    long_end_time_col: Optional[str] = None,
    # Coords copied from input tables
    unit_coords: Optional[List[str]] = None,
    trial_coords: Optional[List[str]] = None,
    # Binning control (only relevant when both units_df + trials_df passed)
    bin_size: Optional[float] = None,
    window: Optional[Tuple[float, float]] = None,
    time: Optional[np.ndarray] = None,
    # Metadata
    time_unit: str = "s",
    validate: bool = True,
) -> xr.DataArray:
    """
    Convert pandas or polars DataFrames to xarray DataArrays for ephys analysis.

    Four use cases (by number of DataFrames and bin_size):

    1) from_dataframe(units_df):
         -> ragged spikes DataArray (unit, trial) with trial=[0]
            spike times are session-time (no segmentation)

    2) from_dataframe(trials_df) where it does NOT look like a units table:
         -> events/epochs DataArray (trial, event, bound)

    3) from_dataframe(units_df, trials_df, bin_size=None):
         -> ragged spikes DataArray (unit, trial)
            segmented by [trial_start, trial_end] and expressed relative to anchor
            (anchor = align_to column if provided else trial_start)

    4) from_dataframe(units_df, trials_df, bin_size=...):
         -> dense binned spikes DataArray (unit, trial, time) (rate Hz)
            segmented by trial bounds and binned relative to anchor within window/time

    The trials_df can be organized in "wide" or "long" format.
    
    Example of "wide" format:

        trial_id | start_time | end_time | go_cue_time | delay_start | delay_end
        ---------------------------------------------------------------------------
            1    |     0.0    |   2.0    |     0.5     |     0.7     |   1.5
            2    |     3.0    |   5.0    |     3.5     |     3.7     |   4.5

    Example of "long" format:

        trial_id | event    | start_time | end_time
        ---------------------------------------------
            1    | go_cue   |    0.5     |   0.5
            1    | delay    |    0.7     |   1.5
            2    | go_cue   |    3.5     |   3.5
            2    | delay    |    3.7     |   4.5

    Parameters
    ----------
    *dfs : pd.DataFrame
        One or two DataFrames. If one DataFrame is provided, it is interpreted
        as either a units table (containing spike times) or a trials/events
        table. If two DataFrames are provided, one must be a units table and
        the other a trials table.
    unit_id_col : str, optional
        Column name in units_df to use as unit identifiers. If None, uses the
        DataFrame index (for pandas DataFrames) or row index (for polars DataFrames).
    trial_id_col : str, optional
        Column name in trials_df to use as trial identifiers. If None, uses
        the DataFrame index (for pandas DataFrames) or row index (for polars DataFrames).
    spike_times_col : str, default="spike_times"
        Column name in units_df containing spike times (in session time).
        Each entry should be a 1D array-like of spike times in seconds.
    trial_start_col : str, default="start_time"
        Column name in trials_df indicating trial start times (session time).
    trial_end_col : str, default="end_time"
        Column name in trials_df indicating trial end times (session time).
    align_to : str, optional
        Column name in trials_df to use as the alignment anchor for each
        trial. If None, defaults to trial_start_col. Spike times will be
        expressed relative to this anchor.
    event_cols : dict[str, str], optional
        Dictionary mapping event names to column names in trials_df for
        instantaneous events. Only used when a single trials_df is provided
        (case 2). E.g., {"go_cue": "go_cue_time"}.
    epoch_cols : dict[str, tuple[str, str]], optional
        Dictionary mapping epoch names to (start_col, end_col) pairs in
        trials_df for extended epochs. Only used when a single trials_df is
        provided (case 2). E.g., {"delay": ("delay_start", "delay_end")}.
    long_event_col : str, optional
        Column name in trials_df containing event names in long format. Only
        used when a single trials_df is provided (case 2).
    long_time_col : str, optional
        Column name in trials_df containing event times corresponding to
        long_event_col. Only used when a single trials_df is provided (case 2).
    long_end_time_col : str, optional
        Column name in trials_df containing epoch end times for long format
        epochs. Only used when a single trials_df is provided (case 2).
    unit_coords : list[str], optional
        List of column names from units_df to attach as unit coordinates. If
        None, all columns except unit_id_col and spike_times_col are included.
    trial_coords : list[str], optional
        List of column names from trials_df to attach as trial coordinates. If
        None, all columns except trial_id_col, trial_start_col, trial_end_col,
        and align_to are included.
    bin_size : float, optional
        Time bin size (in time_unit) for binning spikes. Only relevant when
        both units_df and trials_df are provided. If None, returns ragged
        spikes (case 3). If specified, returns dense binned spikes (case 4).
    window : tuple[float, float], optional
        Time window (start, end) relative to anchor for extracting spikes,
        specified in time_unit. Required when bin_size is specified. If None
        with bin_size=None, uses the full trial duration.
    time : np.ndarray, optional
        Custom 1D array of uniformly-spaced time bin centers for binned output.
        Must have spacing equal to bin_size. If provided, overrides automatic
        bin center calculation. Only used when bin_size is specified.
    time_unit : str, default="s"
        Unit of time for all time-related parameters and output ("s" for
        seconds, "ms" for milliseconds, etc.).
    validate : bool, default=True
        If True, validates the output DataArray against ephys standards using
        aind_ephys_utils.standards.validate.

    Returns
    -------
    xr.DataArray
        Ephys data as an xarray DataArray with appropriate dimensions, 
        coordinates, and metadata attributes.

    Raises
    ------
    FromDataFrameError
        If inputs cannot be interpreted or validated, including:
        - Wrong number of DataFrames (must be 1 or 2)
        - Missing required columns
        - Invalid spike times format
        - Trial boundaries contain NaNs or invalid values
        - bin_size specified without window (when time is not provided)
        - time array is not uniformly spaced or doesn't match bin_size

    Examples
    --------
    Case 1: Units only (session time)
    >>> spikes = from_dataframe(units_df)

    Case 2: Events/epochs from trials
    >>> events = from_dataframe(trials_df)

    Case 3: Units + trials (ragged, trial time)
    >>> spikes = from_dataframe(units_df, trials_df, align_to="stim_onset")

    Case 4: Units + trials (binned, trial time)
    >>> spikes = from_dataframe(
    ...     units_df, trials_df,
    ...     bin_size=0.01, window=(-0.5, 1.0), align_to="stim_onset"
    ... )
    """
    if len(dfs) == 0 or len(dfs) > 2:
        raise FromDataFrameError("from_dataframe expects 1 or 2 DataFrames.")

    if len(dfs) == 1:
        df0 = dfs[0]
        if _looks_like_units_df(df0, spike_times_col):
            spikes = _build_spikes_units_only(
                df0,
                spike_times_col=spike_times_col,
                unit_id_col=unit_id_col,
                unit_coords=unit_coords,
                time_unit=time_unit,
            )
            if validate:
                validate_xarray(spikes)
            return spikes

        # Otherwise: treat as trials/events table
        events = _build_events(
            df0,
            trial_id_col=trial_id_col,
            event_cols=event_cols,
            epoch_cols=epoch_cols,
            long_event_col=long_event_col,
            long_time_col=long_time_col,
            long_end_time_col=long_end_time_col,
            trial_coords=trial_coords,
            time_unit=time_unit,
            timebase="session",
        )
        if validate:
            validate_xarray(events)
        return events

    # len(dfs) == 2: determine which is units vs trials by presence of spike_times_col
    df_a, df_b = dfs
    a_is_units = _looks_like_units_df(df_a, spike_times_col)
    b_is_units = _looks_like_units_df(df_b, spike_times_col)

    if a_is_units and not b_is_units:
        units_df, trials_df = df_a, df_b
    elif b_is_units and not a_is_units:
        units_df, trials_df = df_b, df_a
    elif a_is_units and b_is_units:
        raise FromDataFrameError(
            f"Both DataFrames contain {spike_times_col!r}; "
            "cannot infer units_df vs trials_df."
        )
    else:
        raise FromDataFrameError(
            f"Neither DataFrame contains {spike_times_col!r}; "
            "cannot infer units_df."
        )

    spikes = _build_spikes_with_trials(
        units_df,
        trials_df,
        spike_times_col=spike_times_col,
        unit_id_col=unit_id_col,
        trial_id_col=trial_id_col,
        trial_start_col=trial_start_col,
        trial_end_col=trial_end_col,
        align_to=align_to,
        unit_coords=unit_coords,
        trial_coords=trial_coords,
        bin_size=bin_size,
        window=window,
        time=time,
        time_unit=time_unit,
    )

    if validate:
        validate_xarray(spikes)
    return spikes


__all__ = ["from_dataframe", "FromDataFrameError"]
