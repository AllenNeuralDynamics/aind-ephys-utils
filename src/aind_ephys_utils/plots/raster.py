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
    markersize: Optional[float] = None,
    alpha: Optional[float] = 1.0,
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
    color, linewidth, markersize, alpha:
        Styling for raster markers.
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
                markersize=markersize,
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
        markersize=markersize,
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
    markersize: Optional[float],
    alpha: Optional[float],
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
    if len(per_unit) <= 1:
        # For single-unit rasters, keep trial-oriented y-axis ticks.
        show_unit_labels = False

    unit_centers = []
    unit_labels = []
    if group_dim == C.unit:
        sort_idx = _sort_indices_by_dim(spikes, sort_by=sort_by, dim=group_dim)
        groups = _group_indices_by_dim(
            spikes, group_by=group_by, dim=group_dim, sort_idx=sort_idx
        )
        allow_region = _allow_region_colors(group_by)
        colors = _resolve_group_colors(
            groups, color=color, color_by=color_by, allow_region=allow_region
        )
        if C.trial in spikes.dims:
            base_vals = np.asarray(
                spikes.isel({C.trial: 0}).values, dtype=object
            ).ravel()
        else:
            base_vals = np.asarray(spikes.values, dtype=object).ravel()

        for (label, idx, _), c in zip(groups, colors):
            seq_vals = base_vals[idx]
            seq = _as_list_of_1d_arrays(np.asarray(seq_vals, dtype=object))
            if tlim is not None:
                tmin, tmax = tlim
                seq = [
                    (
                        s[
                            np.searchsorted(s, tmin): np.searchsorted(
                                s, tmax, "right"
                            )
                        ]
                        if s.size
                        else s
                    )
                    for s in seq
                ]
            lineoffsets = np.arange(y0, y0 + len(seq))
            x, y = _flatten_for_scatter(seq, lineoffsets)
            _scatter_spikes(
                ax,
                x,
                y,
                color=c,
                markersize=markersize,
                linewidth=linewidth,
                alpha=alpha,
                rasterized=rasterized,
            )
            _draw_group_label(ax, label, groups, y0, len(seq), tlim)
            y0 += len(seq)
    else:
        for ui, da_u in enumerate(per_unit):
            sort_idx = _sort_indices_by_dim(
                da_u, sort_by=sort_by, dim=group_dim
            )
            groups = _group_indices_by_dim(
                da_u, group_by=group_by, dim=group_dim, sort_idx=sort_idx
            )
            allow_region = _allow_region_colors(group_by)
            colors = _resolve_group_colors(
                groups,
                color=color,
                color_by=color_by,
                allow_region=allow_region,
            )
            unit_start = y0
            da_u_vals = np.asarray(da_u.values, dtype=object).ravel()

            for (label, idx, _), c in zip(groups, colors):
                seq = _as_list_of_1d_arrays(da_u_vals[idx])
                if tlim is not None:
                    tmin, tmax = tlim
                    seq = [
                        s[(s >= tmin) & (s <= tmax)] if s.size else s
                        for s in seq
                    ]
                lineoffsets = np.arange(y0, y0 + len(seq))
                x, y = _flatten_for_scatter(seq, lineoffsets)
                _scatter_spikes(
                    ax,
                    x,
                    y,
                    color=c,
                    markersize=markersize,
                    linewidth=linewidth,
                    alpha=alpha,
                    rasterized=rasterized,
                )
                _draw_group_label(ax, label, groups, y0, len(seq), tlim)
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
    if y0 > 0:
        # Explicit limits avoid backend-dependent autoscale clipping.
        ax.set_ylim(-0.5, y0 - 0.5)
    if set_xlabel:
        time_unit = spikes.attrs.get(C.attr_time_unit, C.default_time_unit)
        ax.set_xlabel(f"Time ({time_unit})")


def _scatter_spikes(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    markersize: Optional[float],
    linewidth: float,
    alpha: Optional[float],
    rasterized: bool,
) -> None:
    """Plot spike times as a scatter on ax (no-op if no spikes)."""
    if not x.size:
        return
    artist = ax.scatter(
        x,
        y,
        s=(
            float(markersize)
            if markersize is not None
            else max(1.0, linewidth * 8.0)
        ),
        c=color,
        alpha=alpha,
        marker=".",
        linewidths=0,
    )
    if rasterized:
        try:
            artist.set_rasterized(True)
        except Exception:
            pass


def _draw_group_label(
    ax: plt.Axes,
    label: Optional[str],
    groups: list,
    y0: int,
    seq_len: int,
    tlim: Optional[tuple],
) -> None:
    """Draw a group label to the left of the plot if multiple groups exist."""
    if label is not None and len(groups) > 1:
        ax.text(
            x=ax.get_xlim()[0] if tlim is None else tlim[0],
            y=y0 + seq_len / 2,
            s=str(label),
            va="center",
            ha="right",
            fontsize=9,
        )


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


def _flatten_for_scatter(
    seq: Sequence[np.ndarray], lineoffsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ragged spike vectors and matching y offsets for scatter."""
    if not seq:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    lengths = np.asarray([len(s) for s in seq], dtype=int)
    total = int(lengths.sum())
    if total == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = np.concatenate([s for s in seq if len(s) > 0]).astype(
        float, copy=False
    )
    y = np.repeat(lineoffsets.astype(float, copy=False), lengths)
    return x, y


def _to_hashable(value: Any) -> Any:
    """Convert values used as dict keys into hashable forms."""
    if np.isscalar(value) or isinstance(value, (str, bytes)):
        return value
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _sort_indices_by_dim(
    da: xr.DataArray, *, sort_by: Optional[str], dim: Optional[str]
) -> np.ndarray:
    """Return stable sorted indices along dim using NumPy."""
    n = da.sizes.get(dim, 0) if dim is not None else int(np.asarray(da).size)
    base = np.arange(n, dtype=int)
    if sort_by is None:
        return base
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
    vals = np.asarray(coord.values)
    return np.argsort(vals, kind="stable")


def _group_indices_by_dim(  # noqa: C901
    da: xr.DataArray,
    *,
    group_by: Optional[Union[str, Sequence[str]]],
    dim: Optional[str],
    sort_idx: np.ndarray,
) -> list[Tuple[Optional[str], np.ndarray, Any]]:
    """Group index positions by one or more coords along dim using NumPy."""
    if group_by is None:
        return [(None, sort_idx, None)]
    if dim is None:
        raise ValueError("group_by requires a trial or unit dimension.")
    if isinstance(group_by, str):
        group_by = [group_by]

    for g in group_by:
        if g not in da.coords:
            raise ValueError(f"group_by coord {g!r} not found in DataArray.")
        if dim not in da[g].dims:
            raise ValueError(
                f"group_by coord {g!r} must have '{dim}' dimension."
            )

    if len(group_by) == 1:
        g = group_by[0]
        key_vals = np.asarray(da[g].values)[sort_idx]
        # Label each group as "coord=value" for single grouping dimensions.
        name_fn = lambda k: f"{g}={k}"  # noqa: E731
    else:
        arrays = [np.asarray(da[g].values)[sort_idx] for g in group_by]
        key_vals = list(zip(*arrays))
        # Label each group as "coord1=v1,coord2=v2,..." for joint groupings.
        name_fn = lambda k: ",".join(  # noqa: E731
            f"{g}={v}" for g, v in zip(group_by, k)
        )

    order: list[Any] = []
    buckets: Dict[Any, list[int]] = {}
    labels: Dict[Any, Any] = {}
    for pos, key_raw in zip(sort_idx, key_vals):
        key = _to_hashable(key_raw)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
            labels[key] = key_raw
        buckets[key].append(int(pos))

    out: list[Tuple[Optional[str], np.ndarray, Any]] = []
    for key in order:
        label_val = labels[key]
        out.append(
            (
                name_fn(label_val),
                np.asarray(buckets[key], dtype=int),
                label_val,
            )
        )
    return out


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
    groups: Sequence[Tuple[Optional[str], np.ndarray, Any]],
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
