"""PSTH plotting helpers."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..standards.conventions import C
from ..ops.psth import psth as ops_psth


def psth(  # noqa: C901
    data: xr.DataArray,
    *,
    group_by: Optional[Union[str, Sequence[str]]] = None,
    dim: str = C.trial,
    reduce: str = "mean",
    compute: bool = True,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    alpha: float = 1.0,
    linewidth: float = 1.5,
    label: Optional[str] = None,
) -> tuple[xr.DataArray, Optional[plt.Axes]]:
    """
    Plot a PSTH from a binned/continuous DataArray.

    Parameters
    ----------
    data:
        Input DataArray. Selection of units/trials should be done upstream.
        If compute=True and `dim` is present, ops.psth is used to reduce.
    group_by:
        Optional trial coord(s) to split and plot conditions.
    dim:
        Trial dimension to reduce across when compute=True.
    reduce:
        Reduction method passed to ops.psth.
    compute:
        If True, compute the PSTH from data using ops.psth.
    ax:
        Matplotlib Axes to draw on; created if None.
    color, alpha, linewidth, label:
        Plot styling options.

    Returns
    -------
    (psth_da, ax)
        psth_da is the reduced DataArray; ax is the matplotlib Axes (or None).
    """
    if ax is None:
        _, ax = plt.subplots()

    groups = _group_trials(data, group_by=group_by)
    colors = _resolve_colors(color, len(groups))

    psth_list: list[xr.DataArray] = []
    labels: list[Optional[str]] = []

    for (label_g, da_g), c in zip(groups, colors):
        if compute and dim in da_g.dims:
            if da_g.dtype == object:
                raise ValueError(
                    "plot.psth expects numeric data when compute=True; "
                    "bin ragged spikes first (e.g. da.ephys.bin(...))."
                )
            da_plot = ops_psth(da_g, dim=dim, reduce=reduce)
        else:
            da_plot = da_g
        psth_list.append(da_plot)
        labels.append(label_g if label is None else label)

        x = da_plot[C.time].values if C.time in da_plot.coords else None
        if C.unit in da_plot.dims:
            unit_vals = list(da_plot[C.unit].values)
            unit_colors = (
                [c] * len(unit_vals)
                if color is not None
                else _resolve_colors(None, len(unit_vals))
            )
            for u, cu in zip(unit_vals, unit_colors):
                y = da_plot.sel({C.unit: u}).values
                line_label = labels[-1]
                if line_label is None:
                    line_label = f"unit={u}"
                else:
                    line_label = f"{line_label}, unit={u}"
                if x is None:
                    ax.plot(
                        y,
                        color=cu,
                        alpha=alpha,
                        linewidth=linewidth,
                        label=line_label,
                    )
                else:
                    ax.plot(
                        x,
                        y,
                        color=cu,
                        alpha=alpha,
                        linewidth=linewidth,
                        label=line_label,
                    )
        else:
            y = da_plot.values
            if x is None:
                ax.plot(
                    y, color=c, alpha=alpha, linewidth=linewidth, label=labels[-1]
                )
            else:
                ax.plot(
                    x,
                    y,
                    color=c,
                    alpha=alpha,
                    linewidth=linewidth,
                    label=labels[-1],
                )

    if len(groups) > 1 or any(C.unit in p.dims for p in psth_list):
        ax.legend(frameon=False)
    time_unit = data.attrs.get(C.attr_time_unit, C.default_time_unit)
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel("Rate (Hz)")

    if len(psth_list) == 1:
        return psth_list[0], ax

    out = xr.concat(psth_list, dim="group")
    out = out.assign_coords(group=np.asarray(labels, dtype=object))
    return out, ax


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


def _resolve_colors(color: Optional[str], n: int) -> Sequence[Optional[str]]:
    """Resolve a list of colors for grouped plots."""
    if n <= 1:
        return [color]
    cmap = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cmap:
        return [color] * n
    return [cmap[i % len(cmap)] for i in range(n)]
