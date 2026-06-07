"""Epoch-based spike time manipulation utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, cast

import numpy as np


def concat_spikes_in_epochs(  # noqa: C901
    spike_times: Sequence[float] | np.ndarray,
    epochs: Iterable[tuple[float, float]] | np.ndarray,
    *,
    include_right: bool = False,
    assume_sorted_spikes: bool = True,
    assume_sorted_epochs: bool = False,
    return_extras: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Clip spikes to epochs, make times relative, and concatenate continuously.

    Parameters
    ----------
    spike_times
        1D array-like of spike times (seconds). Duplicates allowed.
    epochs
        Iterable of (start, end) pairs. Overlapping epochs cause spikes in the
        overlap to appear multiple times (once per epoch).
    include_right
        If True, epochs are treated as [start, end]; otherwise [start, end).
    assume_sorted_spikes
        If True, skips sorting ``spike_times``.
    assume_sorted_epochs
        If True, skips sorting ``epochs`` by start time.
    return_extras
        If True, also returns a dict with ``orig_indices``, ``epoch_ids``,
        ``epoch_offsets``, and ``epoch_durations``.

    Returns
    -------
    out_times : np.ndarray
        1D array of concatenated spike times, shifted so time is continuous
        across epochs.
    extras : dict
        Only if ``return_extras=True``.

    Examples
    --------
    >>> t = np.array([0.1, 0.3, 0.35, 0.9, 1.05, 1.2, 1.8, 2.1])
    >>> epochs = [(0.25, 0.5), (1.0, 1.5), (1.9, 2.2)]
    >>> out = concat_spikes_in_epochs(t, epochs)
    >>> out
    array([0.05, 0.1 , 0.3 , 0.45, 0.95])

    Epoch durations are [0.25, 0.50, 0.30].  Cumulative offsets are
    [0.0, 0.25, 0.75].  Per-epoch relative spikes are shifted by the
    cumulative offset so the output is continuous.
    """
    t = np.asarray(spike_times, dtype=float)
    ep = np.asarray(epochs, dtype=float)
    if ep.ndim != 2 or ep.shape[1] != 2:
        raise ValueError("epochs must be an array-like of (start, end) pairs")

    if np.any(~np.isfinite(ep)):
        raise ValueError("epochs contain non-finite values")
    if np.any(ep[:, 1] < ep[:, 0]):
        raise ValueError("each epoch must satisfy end >= start")

    if not assume_sorted_epochs:
        order_ep = np.argsort(ep[:, 0], kind="stable")
        ep = ep[order_ep]

    if assume_sorted_spikes:
        t_sorted = t
        sorter = None
    else:
        sorter = np.argsort(t, kind="stable")
        t_sorted = t[sorter]

    durations = np.maximum(0.0, ep[:, 1] - ep[:, 0])
    offsets = np.concatenate(([0.0], np.cumsum(durations)[:-1]))

    side_hi: Literal["left", "right"] = "right" if include_right else "left"
    kept_slices: list[tuple[int, int, int]] = []
    kept_counts: list[int] = []

    for ei, (start, end) in enumerate(ep):
        lo = np.searchsorted(t_sorted, start, side="left")
        hi = np.searchsorted(t_sorted, end, side=side_hi)
        if hi > lo:
            kept_slices.append((ei, lo, hi))
            kept_counts.append(hi - lo)

    if not kept_slices:
        out = np.empty(0, dtype=float)
        if return_extras:
            extras = {
                "orig_indices": np.empty(0, dtype=int),
                "epoch_ids": np.empty(0, dtype=int),
                "epoch_offsets": offsets,
                "epoch_durations": durations,
            }
            return out, extras
        return out

    total = int(np.sum(kept_counts))
    out = np.empty(total, dtype=float)
    out_epoch_ids = np.empty(total, dtype=int)
    orig_idx = np.empty(total, dtype=int)

    pos = 0
    for ei, lo, hi in kept_slices:
        seg = t_sorted[lo:hi]
        rel = (seg - ep[ei, 0]) + offsets[ei]
        n = hi - lo
        out[pos : pos + n] = rel
        out_epoch_ids[pos : pos + n] = ei
        if sorter is None:
            orig_idx[pos : pos + n] = np.arange(lo, hi)
        else:
            orig_idx[pos : pos + n] = sorter[lo:hi]
        pos += n

    if return_extras:
        extras = {
            "orig_indices": orig_idx,
            "epoch_ids": out_epoch_ids,
            "epoch_offsets": offsets,
            "epoch_durations": durations,
        }
        return out, extras
    return out


def concat_event_windows(
    spike_times_by_unit: list[np.ndarray],
    events: np.ndarray,
    window: tuple[float, float],
    *,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Concatenate spike trains clipped to windows around events.

    For each unit, clips spikes to ``[event + window[0], event + window[1])``
    for every event, converts to relative time within each window, and
    stitches them into one continuous train.

    Parameters
    ----------
    spike_times_by_unit
        List of sorted spike time arrays, one per unit.
    events
        1-D array of event times (seconds).
    window
        ``(start, end)`` relative to each event (seconds).
    shuffle
        If ``True``, randomly permute the event order before alignment
        (useful for generating surrogate data).
    rng
        NumPy random generator for shuffling.  Ignored when ``shuffle=False``.

    Returns
    -------
    list[np.ndarray]
        One continuous spike train per unit with times in
        ``[0, n_events * window_duration)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    ev = np.asarray(events)
    if shuffle:
        # Each unit gets an independent permutation so inter-unit
        # temporal relationships are destroyed (surrogate data).
        out: list[np.ndarray] = []
        for st in spike_times_by_unit:
            ev_perm = rng.permutation(ev)
            epochs = np.column_stack(
                [ev_perm + window[0], ev_perm + window[1]]
            )
            concatenated = cast(
                np.ndarray,
                concat_spikes_in_epochs(st, epochs, assume_sorted_epochs=True),
            )
            out.append(np.sort(concatenated))
        return out
    epochs = np.column_stack([ev + window[0], ev + window[1]])
    return [
        cast(np.ndarray, concat_spikes_in_epochs(st, epochs))
        for st in spike_times_by_unit
    ]


def add_random_offset_with_wrap(
    arrays: list[np.ndarray],
    total_duration: float,
    offset_min: float = 0,
    offset_max: float | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Add a random offset to each array of times and wrap around ``[0, total_duration]``.

    Parameters
    ----------
    arrays
        List of arrays containing times in ``[0, total_duration]``.
    total_duration
        Maximum time value; times wrap modulo this value.
    offset_min
        Minimum random offset.
    offset_max
        Maximum random offset.  Defaults to *total_duration*.
    rng
        NumPy random generator.  If ``None``, a new default generator is used.
    """
    if offset_max is None:
        offset_max = total_duration
    if rng is None:
        rng = np.random.default_rng()

    shifted_arrays: list[np.ndarray] = []
    for arr in arrays:
        arr = np.asarray(arr)
        offset = rng.uniform(offset_min, offset_max)
        shifted = np.sort((arr + offset) % total_duration)
        shifted_arrays.append(shifted)
    return shifted_arrays
