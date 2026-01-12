"""Raster plot for ragged spike times.

The main entry point is `raster`, which visualizes per-trial spike times using
matplotlib.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


class RasterPlotError(ValueError):
    """Raised when raster plot inputs are invalid or inconsistent."""

    pass


def _as_list_of_1d_arrays(values: np.ndarray) -> list[np.ndarray]:
    """
    Convert an object array (trial,) of spike arrays into a list of 1D float arrays.
    """
    out: list[np.ndarray] = []
    for x in values:
        if x is None:
            out.append(np.asarray([], dtype=float))
        else:
            a = np.asarray(x, dtype=float)
            if a.ndim != 1:
                raise RasterPlotError(
                    f"Expected 1D spike arrays, got shape {a.shape}."
                )
            out.append(a)
    return out


def raster(
    spikes: xr.DataArray,
    *,
    unit: Optional[Union[int, str]] = None,
    units: Optional[Sequence[Union[int, str]]] = None,
    trials: Optional[Sequence[Union[int, str]]] = None,
    sort_by: Optional[str] = None,
    sort_ascending: bool = True,
    tlim: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    color: str = "k",
    linewidth: float = 0.8,
    alpha: float = 1.0,
    rasterized: bool = True,
    show_ylabel: bool = True,
    unit_gap: int = 6,
) -> plt.Axes:
    """
    Raster plot for ragged spike times.

    Parameters
    ----------
    spikes:
        Ragged spike DataArray. Expected dtype=object.
        Typical dims: ("unit","trial") with each entry a 1D array of spike times.
        Also supports a pre-selected ("trial",) object DataArray for a single unit.
    unit:
        Single unit id/label to select (if spikes has a 'unit' dimension).
    units:
        Multiple unit ids/labels to plot stacked (each unit's trials stacked).
        If provided, overrides `unit`.
    trials:
        Optional subset/order of trial ids/labels (applied after sorting).
    sort_by:
        Name of a trial coordinate to sort trials by (e.g. "rt", "choice", "reward").
        Sorting happens within each unit.
    sort_ascending:
        Sort direction for sort_by.
    tlim:
        (tmin, tmax) x-limits; also filters spikes to this range for speed.
    ax:
        Matplotlib Axes to draw on; created if None.
    color, linewidth, alpha:
        Styling for tick marks.
    rasterized:
        If True, rasterize the tick artists (useful for large figures / PDFs).
    show_ylabel:
        If True, label y-axis as "trial".
    unit_gap:
        Extra blank rows between units when plotting multiple units.

    Returns
    -------
    ax:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    if spikes.dtype != object:
        raise RasterPlotError(
            f"raster expects a ragged spike DataArray with dtype=object; got {spikes.dtype!r}."
        )

    # Normalize to a list of per-unit DataArrays with dims ("trial",)
    per_unit: list[Tuple[str, xr.DataArray]] = []

    if "unit" in spikes.dims:
        if units is not None:
            for u in units:
                da_u = spikes.sel(unit=u)
                if "trial" not in da_u.dims:
                    raise RasterPlotError(
                        "Selected unit did not yield a ('trial',) DataArray."
                    )
                per_unit.append((str(u), da_u))
        else:
            if unit is None:
                if spikes.sizes.get("unit", 0) == 1:
                    u0 = spikes["unit"].values[0]
                    da_u = spikes.isel(unit=0)
                    per_unit.append((str(u0), da_u))
                else:
                    raise RasterPlotError(
                        "spikes has a 'unit' dimension with multiple units; "
                        "please specify unit=... or units=[...]."
                    )
            else:
                da_u = spikes.sel(unit=unit)
                per_unit.append((str(unit), da_u))
    else:
        # assume already ("trial",)
        if "trial" not in spikes.dims:
            raise RasterPlotError(
                "Expected spikes dims to include ('unit','trial') or be pre-selected to ('trial',)."
            )
        per_unit.append(("unit", spikes))

    y0 = 0  # running y offset for stacked units

    for unit_label, da_u in per_unit:
        da = da_u

        # subset trials
        if trials is not None:
            da = da.sel(trial=list(trials))

        # sort trials by a trial coord
        if sort_by is not None:
            if sort_by not in da.coords:
                raise RasterPlotError(
                    f"sort_by={sort_by!r} not found in DataArray coords. "
                    f"Available: {list(da.coords)}"
                )
            order = np.argsort(np.asarray(da[sort_by].values))
            if not sort_ascending:
                order = order[::-1]
            da = da.isel(trial=order)

        # Convert to list-of-arrays for matplotlib.eventplot
        seq = _as_list_of_1d_arrays(np.asarray(da.values, dtype=object))

        # Optionally filter by tlim (reduces draw time for very dense spikes)
        if tlim is not None:
            tmin, tmax = tlim
            seq = [s[(s >= tmin) & (s <= tmax)] if s.size else s for s in seq]

        # y positions: one row per trial
        n_trials = len(seq)
        lineoffsets = np.arange(y0, y0 + n_trials)

        # eventplot draws each trial as a separate row of tick marks
        artists = ax.eventplot(
            seq,
            lineoffsets=lineoffsets,
            linelengths=0.8,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            orientation="horizontal",
        )
        # rasterize for big plots (PDF friendliness)
        if rasterized:
            for a in artists:
                try:
                    a.set_rasterized(True)
                except Exception:
                    pass

        # optional unit label on the left (only when plotting multiple units)
        if len(per_unit) > 1:
            ax.text(
                x=ax.get_xlim()[0] if tlim is None else tlim[0],
                y=y0 + n_trials / 2,
                s=str(unit_label),
                va="center",
                ha="right",
                fontsize=9,
            )

        y0 += n_trials + unit_gap

    # cosmetics
    ax.set_xlabel("time")
    if show_ylabel:
        ax.set_ylabel("trial")
    ax.invert_yaxis()  # common raster convention: trial 0 at top
    if tlim is not None:
        ax.set_xlim(tlim)
    ax.set_ylim(y0 - 1, -1)

    return ax
