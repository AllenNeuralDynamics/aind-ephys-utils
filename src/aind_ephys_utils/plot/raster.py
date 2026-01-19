"""Raster plot helpers for ragged spike times."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..core.conventions import C


def raster(  # noqa: C901
    spikes: xr.DataArray,
    *,
    group_by: Optional[Union[str, Sequence[str]]] = None,
    tlim: Optional[tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    color: str = "k",
    linewidth: float = 0.8,
    alpha: float = 1.0,
    rasterized: bool = True,
    show_ylabel: bool = True,
    unit_gap: int = 3,
) -> plt.Axes:
    """
    Raster plot for ragged spike DataArrays.

    Parameters
    ----------
    spikes:
        Ragged spike DataArray. Expected dtype=object with a trial dimension.
        Selection of units/trials should be done upstream.
    group_by:
        Optional trial coord(s) to split and color trials by condition.
    tlim:
        Optional (tmin, tmax) for filtering and x-limits.
    ax:
        Matplotlib Axes to draw on; created if None.
    color, linewidth, alpha:
        Styling for tick marks.
    rasterized:
        If True, rasterize artists for large figures.
    show_ylabel:
        If True, label y-axis as "trial".
    unit_gap:
        Extra blank rows between units when stacking multiple units.

    Returns
    -------
    plt.Axes
        The matplotlib Axes.
    """
    if spikes.dtype != object:
        raise ValueError(
            "raster expects a ragged spike DataArray with dtype=object."
        )
    if C.trial not in spikes.dims:
        if C.unit in spikes.dims:
            spikes = spikes.expand_dims({C.trial: [0]}).transpose(
                C.trial, C.unit
            )
            ylabel = C.unit
        else:
            spikes = spikes.expand_dims({C.trial: [0]})
            ylabel = C.trial
    else:
        ylabel = C.trial
    extra_dims = set(spikes.dims) - {C.trial, C.unit}
    if extra_dims:
        raise ValueError(
            f"raster expects only a '{C.trial}' dimension; found {extra_dims}."
        )

    if ax is None:
        _, ax = plt.subplots()

    if spikes.sizes.get(C.trial, 0) <= 1:
        unit_gap = 0
        show_unit_labels = False
    else:
        show_unit_labels = True
    y0 = 0
    per_unit = []
    if C.unit in spikes.dims:
        for u in spikes[C.unit].values:
            per_unit.append(spikes.sel({C.unit: u}))
    else:
        per_unit.append(spikes)

    unit_centers = []
    unit_labels = []
    for ui, da_u in enumerate(per_unit):
        groups = _group_trials(da_u, group_by=group_by)
        colors = _resolve_colors(color, len(groups))
        unit_start = y0

        for (label, da), c in zip(groups, colors):
            seq = _as_list_of_1d_arrays(np.asarray(da.values, dtype=object))
            if tlim is not None:
                tmin, tmax = tlim
                seq = [
                    s[(s >= tmin) & (s <= tmax)] if s.size else s for s in seq
                ]

            lineoffsets = np.arange(y0, y0 + len(seq))
            artists = ax.eventplot(
                seq,
                lineoffsets=lineoffsets,
                linelengths=0.8,
                colors=c,
                linewidths=linewidth,
                alpha=alpha,
                orientation="horizontal",
            )
            if rasterized:
                for a in artists:
                    try:
                        a.set_rasterized(True)
                    except Exception:
                        pass

            if label is not None and len(groups) > 1:
                ax.text(
                    x=ax.get_xlim()[0] if tlim is None else tlim[0],
                    y=y0 + len(seq) / 2,
                    s=str(label),
                    va="center",
                    ha="right",
                    fontsize=9,
                )

            y0 += len(seq)
        unit_end = y0
        if show_unit_labels:
            unit_centers.append((unit_start + unit_end) / 2)
            if C.unit in spikes.dims:
                unit_labels.append(str(spikes[C.unit].values[ui]))
            else:
                unit_labels.append("unit")
        y0 += unit_gap if ui < len(per_unit) - 1 else 0

    if tlim is not None:
        ax.set_xlim(tlim)
    if show_ylabel:
        if ylabel == C.unit:
            ax.set_ylabel("Unit")
        else:
            ax.set_ylabel("Trial")
    if show_unit_labels and unit_centers:
        ax.set_yticks(unit_centers)
        ax.set_yticklabels(unit_labels)
    time_unit = spikes.attrs.get(C.attr_time_unit, C.default_time_unit)
    ax.set_xlabel(f"Time ({time_unit})")
    return ax


def _as_list_of_1d_arrays(values: np.ndarray) -> list[np.ndarray]:
    """Convert ragged entries to a list of 1D float arrays."""
    out: list[np.ndarray] = []
    for x in values:
        if x is None:
            out.append(np.asarray([], dtype=float))
        else:
            arr = np.asarray(x, dtype=float)
            if arr.ndim != 1:
                raise ValueError(
                    f"Expected 1D spike arrays, got shape {arr.shape}."
                )
            out.append(arr)
    return out


def _group_trials(
    da: xr.DataArray, *, group_by: Optional[Union[str, Sequence[str]]]
) -> list[Tuple[Optional[str], xr.DataArray]]:
    """Split a DataArray into trial groups by one or more coords."""
    if group_by is None:
        return [(None, da)]
    if isinstance(group_by, str):
        group_by = [group_by]
    for g in group_by:
        if g not in da.coords:
            raise ValueError(f"group_by coord {g!r} not found in DataArray.")
    if C.trial not in da.dims:
        raise ValueError("group_by requires a trial dimension.")
    if len(group_by) == 1:
        key = group_by[0]
        return [(f"{key}={k}", v) for k, v in da.groupby(key)]

    labels = list(zip(*(da[g].values for g in group_by)))
    group_coord = xr.DataArray(
        labels, dims=(C.trial,), coords={C.trial: da[C.trial]}
    )
    da2 = da.assign_coords(_group=group_coord)
    out: list[Tuple[Optional[str], xr.DataArray]] = []
    for key, sub in da2.groupby("_group"):
        label = ",".join(f"{g}={v}" for g, v in zip(group_by, key))
        out.append((label, sub))
    return out


def _resolve_colors(color: str, n: int) -> Iterable[str]:
    """Resolve a list of colors for grouped plots."""
    if n <= 1:
        return [color]
    cmap = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cmap:
        return [color] * n
    return [cmap[i % len(cmap)] for i in range(n)]
