"""Low-dimensional trajectory plotting helpers."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from ..core.conventions import C


def trajectory(
    data: xr.DataArray,
    *,
    group_by: Optional[Union[str, Sequence[str]]] = None,
    component_dim: str = "component",
    time_dim: str = C.time,
    components: Optional[Sequence[int]] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    alpha: float = 1.0,
    linewidth: float = 1.5,
    label: Optional[str] = None,
) -> plt.Axes:
    """
    Plot trajectories in low-dimensional space.

    Parameters
    ----------
    data:
        Reduced DataArray (e.g., components x time). Selection should be done
        upstream; this function only plots.
    group_by:
        Optional trial coord(s) to split and plot conditions.
    component_dim:
        Name of the component dimension.
    time_dim:
        Name of the time dimension.
    components:
        Optional component indices to plot (e.g., [0,1] or [0,1,2]).
    ax:
        Matplotlib Axes to draw on; created if None.
    color, alpha, linewidth, label:
        Plot styling options.

    Returns
    -------
    plt.Axes
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    if component_dim not in data.dims:
        raise ValueError(
            f"component_dim {component_dim!r} not found in DataArray dims."
        )
    if time_dim not in data.dims:
        raise ValueError(f"time_dim {time_dim!r} not found in DataArray dims.")

    if components is None:
        components = [0, 1]
    if len(components) not in (2, 3):
        raise ValueError("trajectory requires 2 or 3 components.")

    groups = _group_trials(data, group_by=group_by)
    colors = _resolve_colors(color, len(groups))

    if len(components) == 3 and not hasattr(ax, "plot3D"):
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for (label_g, da_g), c in zip(groups, colors):
        if C.trial in da_g.dims:
            da_plot = da_g.mean(dim=C.trial, keep_attrs=True)
        else:
            da_plot = da_g

        sel = da_plot.sel({component_dim: list(components)})
        if len(components) == 2:
            x = sel.sel({component_dim: components[0]}).values
            y = sel.sel({component_dim: components[1]}).values
            ax.plot(
                x,
                y,
                color=c,
                alpha=alpha,
                linewidth=linewidth,
                label=label_g if label is None else label,
            )
        else:
            x = sel.sel({component_dim: components[0]}).values
            y = sel.sel({component_dim: components[1]}).values
            z = sel.sel({component_dim: components[2]}).values
            ax.plot3D(
                x,
                y,
                z,
                color=c,
                alpha=alpha,
                linewidth=linewidth,
                label=label_g if label is None else label,
            )

    time_unit = data.attrs.get(C.attr_time_unit, C.default_time_unit)
    if len(components) == 2:
        ax.set_xlabel(f"Component {components[0] + 1}")
        ax.set_ylabel(f"Component {components[1] + 1}")
    else:
        ax.set_xlabel(f"Component {components[0] + 1}")
        ax.set_ylabel(f"Component {components[1] + 1}")
        ax.set_zlabel(f"Component {components[2] + 1}")
    if time_dim in data.coords:
        ax.set_title(f"Time ({time_unit})")

    if len(groups) > 1:
        ax.legend(frameon=False)
    return ax


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


def _resolve_colors(
    color: Optional[str], n: int
) -> Sequence[Optional[str]]:
    """Resolve a list of colors for grouped plots."""
    if n <= 1:
        return [color]
    cmap = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cmap:
        return [color] * n
    return [cmap[i % len(cmap)] for i in range(n)]
