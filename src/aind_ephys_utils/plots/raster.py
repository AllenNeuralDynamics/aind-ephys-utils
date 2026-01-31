"""Raster plot helpers for ragged spike times."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..standards.conventions import C
from .annotations import add_scale_bars, annotate_events
from .colors import get_color_for_region


def raster(  # noqa: C901
    spikes: xr.DataArray,
    *,
    group_by: Optional[Union[str, Sequence[str]]] = None,
    color_by: Optional[Dict[Any, str]] = None,
    sort_by: Optional[str] = None,
    events: Optional[Union[Sequence[float], np.ndarray, xr.DataArray]] = None,
    intervals: Optional[
        Union[Sequence[Tuple[float, float]], np.ndarray, xr.DataArray]
    ] = None,
    event_labels: Optional[Union[Sequence[str], Dict[float, str]]] = None,
    interval_labels: Optional[
        Union[Sequence[str], Dict[Tuple[float, float], str]]
    ] = None,
    event_style: Optional[dict] = None,
    interval_style: Optional[dict] = None,
    tlim: Optional[tuple[float, float]] = None,
    ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None,
    color: str = "k",
    linewidth: float = 0.8,
    alpha: float = 1.0,
    rasterized: bool = True,
    show_ylabel: bool = True,
    unit_gap: int = 3,
    plot_all_units: bool = False,
    hide_axes: bool = False,
    x_scale: Optional[float] = None,
    y_scale: Optional[float] = None,
    x_scale_text: Optional[str] = None,
    y_scale_text: Optional[str] = None,
) -> Union[plt.Axes, np.ndarray]:
    """
    Raster plot for ragged spike DataArrays.

    Parameters
    ----------
    spikes:
        Ragged spike DataArray. Expected dtype=object with a trial dimension.
        Selection of units/trials should be done upstream.
    group_by:
        Optional trial coord(s) to split and color trials by condition.
    color_by:
        Optional mapping from group value or label to colors. Overrides
        region-based coloring when provided.
    sort_by:
        Optional trial coord to sort trials by (ascending).
    events:
        Optional event times to overlay as vertical lines.
    intervals:
        Optional intervals to overlay as shaded regions.
    event_labels:
        Optional labels for events (sequence aligned with events or mapping).
    interval_labels:
        Optional labels for intervals (sequence aligned with intervals or mapping).
    event_style:
        Optional styling dict passed to ``ax.axvline``.
    interval_style:
        Optional styling dict passed to ``ax.axvspan``.
    hide_axes:
        If True, hide axes and draw scale bars.
    x_scale:
        Optional length of x-axis scale bar in data units.
    y_scale:
        Optional length of y-axis scale bar in data units.
    x_scale_text:
        Optional label text for the x-axis scale bar.
    y_scale_text:
        Optional label text for the y-axis scale bar.
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
    plot_all_units:
        If False (default), cap multi-axis plots at 10 units.

    Returns
    -------
    plt.Axes or np.ndarray
        The matplotlib Axes, or an array of Axes when multiple units are shown.
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
        else:
            spikes = spikes.expand_dims({C.trial: [0]})
    extra_dims = set(spikes.dims) - {C.trial, C.unit}
    if extra_dims:
        raise ValueError(
            f"raster expects only a '{C.trial}' dimension; found {extra_dims}."
        )

    n_trials = spikes.sizes.get(C.trial, 0)
    n_units = spikes.sizes.get(C.unit, 1) if C.unit in spikes.dims else 1
    multi_axes = n_trials > 1 and C.unit in spikes.dims and n_units > 1

    if multi_axes:
        unit_vals = list(spikes[C.unit].values)
        if not plot_all_units and len(unit_vals) > 10:
            unit_vals = unit_vals[:10]
            spikes = spikes.sel({C.unit: unit_vals})

        if ax is None:
            _, axes = plt.subplots(len(unit_vals), 1, sharex=True)
        elif isinstance(ax, plt.Axes):
            if len(unit_vals) != 1:
                raise ValueError(
                    "Multiple units require a matching sequence of axes."
                )
            axes = np.asarray([ax])
        else:
            axes = np.asarray(ax)
            if len(axes) != len(unit_vals):
                raise ValueError(
                    "Number of axes must match number of plotted units."
                )

        for i, (u, ax_i) in enumerate(zip(unit_vals, axes)):
            _plot_raster_on_axis(
                spikes.sel({C.unit: u}),
                ax=ax_i,
                group_by=group_by,
                color_by=color_by,
                sort_by=sort_by,
                tlim=tlim,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                rasterized=rasterized,
                show_ylabel=show_ylabel,
                unit_gap=unit_gap,
                show_unit_labels_override=False,
                set_xlabel=i == len(unit_vals) - 1,
            )
            ax_i.set_title(f"Unit {u}")
            annotate_events(
                ax=ax_i,
                events=events,
                intervals=intervals,
                event_labels=event_labels,
                interval_labels=interval_labels,
                event_style=event_style,
                interval_style=interval_style,
            )
            if hide_axes:
                time_unit = spikes.attrs.get(
                    C.attr_time_unit, C.default_time_unit
                )
                add_scale_bars(
                    ax=ax_i,
                    x_unit=time_unit,
                    y_unit="trials",
                    x_length=x_scale,
                    y_length=y_scale,
                    x_text=x_scale_text,
                    y_text=y_scale_text,
                )
        return axes

    if ax is None:
        _, ax = plt.subplots()

    _plot_raster_on_axis(
        spikes,
        ax=ax if isinstance(ax, plt.Axes) else ax[0],
        group_by=group_by,
        color_by=color_by,
        sort_by=sort_by,
        tlim=tlim,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        rasterized=rasterized,
        show_ylabel=show_ylabel,
        unit_gap=unit_gap,
        show_unit_labels_override=None,
        set_xlabel=True,
    )
    annotate_events(
        ax=ax if isinstance(ax, plt.Axes) else ax[0],
        events=events,
        intervals=intervals,
        event_labels=event_labels,
        interval_labels=interval_labels,
        event_style=event_style,
        interval_style=interval_style,
    )
    if hide_axes:
        time_unit = spikes.attrs.get(C.attr_time_unit, C.default_time_unit)
        n_trials = spikes.sizes.get(C.trial, 0)
        y_label = "trials" if n_trials > 1 else "units"
        add_scale_bars(
            ax=ax if isinstance(ax, plt.Axes) else ax[0],
            x_unit=time_unit,
            y_unit=y_label,
            x_length=x_scale,
            y_length=y_scale,
            x_text=x_scale_text,
            y_text=y_scale_text,
        )
    return ax


def _plot_raster_on_axis(  # noqa: C901
    spikes: xr.DataArray,
    *,
    ax: plt.Axes,
    group_by: Optional[Union[str, Sequence[str]]],
    color_by: Optional[Dict[Any, str]],
    sort_by: Optional[str],
    tlim: Optional[tuple[float, float]],
    color: str,
    linewidth: float,
    alpha: float,
    rasterized: bool,
    show_ylabel: bool,
    unit_gap: int,
    show_unit_labels_override: Optional[bool],
    set_xlabel: bool,
) -> None:
    """Plot a raster for one axis, optionally stacking units."""
    n_trials = spikes.sizes.get(C.trial, 0)
    group_dim = None
    if C.trial in spikes.dims and n_trials > 1:
        group_dim = C.trial
    elif C.unit in spikes.dims:
        group_dim = C.unit

    if n_trials <= 1:
        unit_gap = 0
        show_unit_labels = False
    else:
        show_unit_labels = True
    if show_unit_labels_override is not None:
        show_unit_labels = show_unit_labels_override

    y0 = 0
    per_unit = []
    if C.unit in spikes.dims:
        for u in spikes[C.unit].values:
            per_unit.append(spikes.sel({C.unit: u}))
    else:
        per_unit.append(spikes)

    unit_centers = []
    unit_labels = []
    if group_dim == C.unit:
        da_sorted = _sort_by_dim(spikes, sort_by=sort_by, dim=group_dim)
        groups = _group_by_dim(da_sorted, group_by=group_by, dim=group_dim)
        allow_region = _allow_region_colors(group_by)
        colors = _resolve_group_colors(
            groups, color=color, color_by=color_by, allow_region=allow_region
        )
        for (label, da, _), c in zip(groups, colors):
            if C.trial in da.dims:
                da_vals = da.isel({C.trial: 0}).values
            else:
                da_vals = da.values
            seq = _as_list_of_1d_arrays(np.asarray(da_vals, dtype=object))
            if tlim is not None:
                tmin, tmax = tlim
                seq = [
                    s[(s >= tmin) & (s <= tmax)] if s.size else s for s in seq
                ]

            lineoffsets = np.arange(y0, y0 + len(seq))
            artists = ax.eventplot(
                seq,
                lineoffsets=lineoffsets,
                linelengths=1.0,
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
    else:
        for ui, da_u in enumerate(per_unit):
            da_u = _sort_by_dim(da_u, sort_by=sort_by, dim=group_dim)
            groups = _group_by_dim(da_u, group_by=group_by, dim=group_dim)
            allow_region = _allow_region_colors(group_by)
            colors = _resolve_group_colors(
                groups,
                color=color,
                color_by=color_by,
                allow_region=allow_region,
            )
            unit_start = y0

            for (label, da, _), c in zip(groups, colors):
                seq = _as_list_of_1d_arrays(
                    np.asarray(da.values, dtype=object)
                )
                if tlim is not None:
                    tmin, tmax = tlim
                    seq = [
                        s[(s >= tmin) & (s <= tmax)] if s.size else s
                        for s in seq
                    ]

                lineoffsets = np.arange(y0, y0 + len(seq))
                artists = ax.eventplot(
                    seq,
                    lineoffsets=lineoffsets,
                    linelengths=1.0,
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
        if C.trial in spikes.dims and n_trials > 1:
            ax.set_ylabel("Trial")
        else:
            ax.set_ylabel("Unit")
    if show_unit_labels and unit_centers:
        ax.set_yticks(unit_centers)
        ax.set_yticklabels(unit_labels)
    if set_xlabel:
        time_unit = spikes.attrs.get(C.attr_time_unit, C.default_time_unit)
        ax.set_xlabel(f"Time ({time_unit})")


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


def _group_by_dim(
    da: xr.DataArray,
    *,
    group_by: Optional[Union[str, Sequence[str]]],
    dim: Optional[str],
) -> list[Tuple[Optional[str], xr.DataArray]]:
    """Split a DataArray into groups by one or more coords along dim."""
    if group_by is None:
        return [(None, da, None)]
    if dim is None:
        raise ValueError("group_by requires a trial or unit dimension.")
    if isinstance(group_by, str):
        group_by = [group_by]
    for g in group_by:
        if g not in da.coords:
            raise ValueError(f"group_by coord {g!r} not found in DataArray.")
    for g in group_by:
        if dim not in da[g].dims:
            raise ValueError(
                f"group_by coord {g!r} must have '{dim}' dimension."
            )
    if len(group_by) == 1:
        key = group_by[0]
        return [(f"{key}={k}", v, k) for k, v in da.groupby(key)]

    labels = list(zip(*(da[g].values for g in group_by)))
    group_coord = xr.DataArray(labels, dims=(dim,), coords={dim: da[dim]})
    da2 = da.assign_coords(_group=group_coord)
    out: list[Tuple[Optional[str], xr.DataArray, Any]] = []
    for key, sub in da2.groupby("_group"):
        label = ",".join(f"{g}={v}" for g, v in zip(group_by, key))
        out.append((label, sub, key))
    return out


def _sort_by_dim(
    da: xr.DataArray, *, sort_by: Optional[str], dim: Optional[str]
) -> xr.DataArray:
    """Sort by a scalar coordinate along dim."""
    if sort_by is None:
        return da
    if sort_by not in da.coords:
        raise ValueError(f"sort_by coord {sort_by!r} not found in DataArray.")
    if dim is None:
        raise ValueError("sort_by requires a trial or unit dimension.")
    coord = da.coords[sort_by]
    if dim not in coord.dims:
        raise ValueError(
            f"sort_by coord {sort_by!r} must have '{dim}' dimension."
        )
    if coord.ndim != 1:
        raise ValueError(f"sort_by coord {sort_by!r} must be 1D over '{dim}'.")
    return da.sortby(sort_by)


def _allow_region_colors(
    group_by: Optional[Union[str, Sequence[str]]],
) -> bool:
    """Return True if region-based colors are allowed."""
    if group_by is None:
        return False
    if isinstance(group_by, str):
        return True
    return len(group_by) == 1


def _resolve_group_colors(
    groups: Sequence[Tuple[Optional[str], xr.DataArray, Any]],
    *,
    color: str,
    color_by: Optional[Dict[Any, str]],
    allow_region: bool,
) -> list[str]:
    """Resolve colors for groups with overrides and region lookups."""
    fallback = list(_resolve_colors(color, len(groups)))
    out: list[str] = []
    for (label, _, key), default in zip(groups, fallback):
        override = None
        if color_by:
            override = color_by.get(key)
            if override is None and label is not None:
                override = color_by.get(label)
        if override is None and allow_region and key is not None:
            try:
                override = get_color_for_region(key)
            except ValueError:
                override = None
        out.append(override or default)
    return out


def _resolve_colors(color: str, n: int) -> Iterable[str]:
    """Resolve a list of colors for grouped plots."""
    if n <= 1:
        return [color]
    cmap = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cmap:
        return [color] * n
    return [cmap[i % len(cmap)] for i in range(n)]
