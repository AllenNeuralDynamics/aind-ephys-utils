"""Annotation helpers for plot overlays."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def annotate_events(  # noqa: C901
    *,
    ax: Optional[plt.Axes] = None,
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
) -> plt.Axes:
    """
    Overlay events (vertical lines) and intervals (shaded spans).

    Parameters
    ----------
    ax:
        Matplotlib Axes to draw on; defaults to ``plt.gca()``.
    events:
        1D sequence or DataArray of event times.
    intervals:
        Sequence of (start, end) tuples or DataArray with shape (n, 2).
    event_labels:
        Optional labels for events (sequence aligned with events or mapping).
    interval_labels:
        Optional labels for intervals (sequence aligned with intervals or mapping).
    event_style:
        Optional styling dict passed to ``ax.axvline``.
    interval_style:
        Optional styling dict passed to ``ax.axvspan``.
    """
    if ax is None:
        ax = plt.gca()
    label_items: list[tuple[float, str, bool]] = []
    if events is not None:
        style = {"color": "0.3", "linewidth": 1.0, "alpha": 0.8}
        if event_style:
            style.update(event_style)
        event_values = _as_1d_values(events)
        event_texts = _resolve_event_labels(event_values, event_labels)
        for t, text in zip(event_values, event_texts):
            ax.axvline(float(t), **style)
            if text:
                label_items.append((float(t), text, True))
    if intervals is not None:
        style = {"color": "0.7", "alpha": 0.3}
        if interval_style:
            style.update(interval_style)
        interval_values = list(_as_intervals(intervals))
        interval_texts = _resolve_interval_labels(
            interval_values, interval_labels
        )
        for (start, end), text in zip(interval_values, interval_texts):
            ax.axvspan(float(start), float(end), **style)
            if text:
                label_items.append(((start + end) / 2.0, text, False))
    if label_items:
        _draw_labels(ax, label_items)
    return ax


def add_scale_bars(
    ax: plt.Axes,
    *,
    x_unit: str,
    y_unit: str,
    x_length: Optional[float] = None,
    y_length: Optional[float] = None,
    x_text: Optional[str] = None,
    y_text: Optional[str] = None,
) -> None:
    """Draw simple scale bars and hide axes."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    if x_length is None:
        x_length = 0.1 * x_range
    if y_length is None:
        y_length = 0.1 * y_range

    x0 = x_min + 0.05 * x_range
    y0 = y_min + 0.05 * y_range
    ax.plot([x0, x0 + x_length], [y0, y0], color="k", linewidth=2)
    ax.plot([x0, x0], [y0, y0 + y_length], color="k", linewidth=2)

    x_label = (
        x_text
        if x_text is not None
        else f"{_format_length(x_length)} {x_unit}"
    )
    ax.text(
        x0 + x_length / 2.0,
        y0 - 0.03 * y_range,
        x_label,
        ha="center",
        va="top",
        fontsize=9,
        color="k",
    )
    y_label = (
        y_text
        if y_text is not None
        else f"{_format_length(y_length)} {y_unit}"
    )
    ax.text(
        x0 - 0.03 * x_range,
        y0 + y_length / 2.0,
        y_label,
        ha="right",
        va="center",
        fontsize=9,
        color="k",
        rotation=90,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _as_1d_values(
    values: Union[Sequence[float], np.ndarray, xr.DataArray]
) -> np.ndarray:
    """Coerce input into a 1D float array."""
    if isinstance(values, xr.DataArray):
        arr = np.asarray(values.values, dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("events must be a 1D sequence of times.")
    return arr


def _as_intervals(
    values: Union[Sequence[Tuple[float, float]], np.ndarray, xr.DataArray]
) -> Iterable[Tuple[float, float]]:
    """Coerce input into an iterable of (start, end) pairs."""
    if isinstance(values, xr.DataArray):
        arr = np.asarray(values.values, dtype=float)
    else:
        arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("intervals must be a (n, 2) array or list of tuples.")
    for start, end in arr:
        yield float(start), float(end)


def _resolve_event_labels(
    events: np.ndarray, labels: Optional[Union[Sequence[str], Dict[float, str]]]
) -> list[Optional[str]]:
    """Resolve event labels from a sequence or mapping."""
    if labels is None:
        return [None] * len(events)
    if isinstance(labels, dict):
        return [labels.get(float(t)) for t in events]
    if len(labels) != len(events):
        raise ValueError("event_labels must match the length of events.")
    return list(labels)


def _resolve_interval_labels(
    intervals: Sequence[Tuple[float, float]],
    labels: Optional[
        Union[Sequence[str], Dict[Tuple[float, float], str]]
    ],
) -> list[Optional[str]]:
    """Resolve interval labels from a sequence or mapping."""
    if labels is None:
        return [None] * len(intervals)
    if isinstance(labels, dict):
        return [labels.get((float(s), float(e))) for s, e in intervals]
    if len(labels) != len(intervals):
        raise ValueError("interval_labels must match the length of intervals.")
    return list(labels)


def _draw_labels(
    ax: plt.Axes, items: Sequence[Tuple[float, str, bool]]
) -> None:
    """Draw labels above the plot, with optional event triangles."""
    label_y = 1.06
    tri_y = 1.02
    for x, text, is_event in items:
        ax.text(
            x,
            label_y,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            color="k",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
        if is_event:
            ax.plot(
                [x],
                [tri_y],
                marker="v",
                color="k",
                markersize=4,
                linestyle="None",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )


def _format_length(value: float) -> str:
    """Format a scale length with a compact precision."""
    if value == 0:
        return "0"
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


__all__ = ["add_scale_bars", "annotate_events"]
