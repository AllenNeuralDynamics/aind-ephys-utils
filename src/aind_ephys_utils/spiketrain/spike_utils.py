"""Spike train manipulation and event-alignment utilities.

Most functions in this module are pure-numpy.  The synchrony-detection
helpers (``build_synchrony_index`` / ``count_coincident_units``) require
the ``numba`` optional dependency for the JIT-compiled inner kernel —
they raise ``ImportError`` at call time when ``aind-ephys-utils`` is
installed without the ``[numba]`` extra.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from aind_ephys_utils._numba import njit, prange

# ---------------------------------------------------------------------------
# Spike censoring / proximity queries
# ---------------------------------------------------------------------------


def exclude_near(
    spikes: npt.NDArray, events: npt.NDArray, tol: float
) -> npt.NDArray:
    """Remove spikes within *tol* seconds of any event.

    Parameters
    ----------
    spikes
        Sorted spike times.
    events
        Sorted event times.
    tol
        Exclusion tolerance (seconds).  Spikes within ``[event - tol,
        event + tol]`` are removed.

    Returns
    -------
    npt.NDArray
        Spike times with near-event spikes removed.
    """
    if len(events) == 0:
        return spikes.copy()
    idx = np.searchsorted(events, spikes)
    near_right = (idx < len(events)) & (
        np.abs(events[np.clip(idx, 0, len(events) - 1)] - spikes) <= tol
    )
    near_left = (idx > 0) & (
        np.abs(events[np.clip(idx - 1, 0, len(events) - 1)] - spikes) <= tol
    )
    return spikes[~(near_right | near_left)]


def near_events(
    spike_times: npt.NDArray, event_times: npt.NDArray, tol: float
) -> npt.NDArray:
    """Boolean mask: True for spikes within *tol* of any event.

    Parameters
    ----------
    spike_times
        Sorted spike times.
    event_times
        Sorted event times.
    tol
        Proximity tolerance (seconds).

    Returns
    -------
    npt.NDArray
        Boolean array of same length as *spike_times*.
    """
    idx = np.searchsorted(event_times, spike_times)
    near_right = (idx < len(event_times)) & (
        np.abs(
            event_times[np.clip(idx, 0, len(event_times) - 1)] - spike_times
        )
        <= tol
    )
    near_left = (idx > 0) & (
        np.abs(
            event_times[np.clip(idx - 1, 0, len(event_times) - 1)]
            - spike_times
        )
        <= tol
    )
    return near_right | near_left


def count_coincident_per_train(
    target_spikes: npt.NDArray,
    other_spike_trains: list[npt.NDArray],
    tol: float,
) -> npt.NDArray:
    """For each spike, count how many other trains have a spike within *tol*.

    Parameters
    ----------
    target_spikes
        Sorted spike times for the target unit.
    other_spike_trains
        List of sorted spike time arrays for other units.
    tol
        Coincidence tolerance (seconds).

    Returns
    -------
    npt.NDArray
        Integer array of length ``len(target_spikes)``.  Each element
        is the number of other trains with at least one spike within
        *tol* of the corresponding target spike.
    """
    counts = np.zeros(len(target_spikes), dtype=np.int64)
    for other_st in other_spike_trains:
        if len(other_st) == 0:
            continue
        idx = np.searchsorted(other_st, target_spikes)
        near_right = (idx < len(other_st)) & (
            np.abs(
                other_st[np.clip(idx, 0, len(other_st) - 1)] - target_spikes
            )
            <= tol
        )
        near_left = (idx > 0) & (
            np.abs(
                other_st[np.clip(idx - 1, 0, len(other_st) - 1)]
                - target_spikes
            )
            <= tol
        )
        counts += (near_right | near_left).astype(np.int64)
    return counts


# ---------------------------------------------------------------------------
# Indexed synchrony detection
# ---------------------------------------------------------------------------
#
# ``count_coincident_per_train`` above is O(N_target × N_others ×
# log(avg_train_len)) per call: a Python loop dispatches one numpy
# searchsorted per "other" train, so the cost scales with the number of
# other units even though each individual numpy call is vectorized.
#
# When the same pool of "other" trains is queried for many target trains
# (the common case in artifact-synchrony screening), there's a much
# cheaper algorithmic shape:
#
#   1. Concatenate all other trains into one sorted-time array, plus a
#      parallel array of unit indices that says which train each spike
#      came from.  Done once.
#   2. For each target spike, two ``np.searchsorted`` calls find the
#      slice of the merged array within ``[t-tol, t+tol]``.  A small
#      pass over that slice counts the *distinct* unit indices it
#      contains.
#
# Per target spike that's O(log N_total) for the search + O(slice
# length).  For N_target × N_other spike trains of average length L the
# old approach is roughly O(N_target × N_other × log L); the indexed
# approach is roughly O((N_target + N_other × L) × log(N_other × L))
# amortized over all target queries against the same index.  In
# practice the indexed approach is 10-100× faster on the population-
# synchrony screening loop in the CCG capsule.
#
# The inner kernel is numba-jitted with ``parallel=True`` so the per-
# target-spike work fans out across cores.


def build_synchrony_index(
    spike_trains: list[npt.NDArray],
) -> tuple[npt.NDArray, npt.NDArray]:
    """Concatenate spike trains into one sorted-time array + unit-index array.

    Parameters
    ----------
    spike_trains
        List of sorted spike-time arrays.  Each train's spikes are
        labelled with its positional index ``0..len(spike_trains)-1`` in
        the returned ``unit_ids`` array.

    Returns
    -------
    sorted_times : npt.NDArray
        All spikes from all trains, sorted ascending in time.  Shape
        ``(N_total,)``, dtype ``float64``.
    unit_ids : npt.NDArray
        Parallel array giving the originating train index of each spike
        in ``sorted_times``.  Shape ``(N_total,)``, dtype ``int32``.
    """
    if len(spike_trains) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32)
    total = sum(len(s) for s in spike_trains)
    times = np.empty(total, dtype=np.float64)
    uids = np.empty(total, dtype=np.int32)
    pos = 0
    for i, s in enumerate(spike_trains):
        n = len(s)
        if n == 0:
            continue
        times[pos : pos + n] = s
        uids[pos : pos + n] = i
        pos += n
    order = np.argsort(times, kind="stable")
    return times[order], uids[order]


@njit(parallel=True, cache=True)
def _count_coincident_units_kernel(
    target_spikes: npt.NDArray,
    sorted_times: npt.NDArray,
    sorted_uids: npt.NDArray,
    n_units: int,
    tol: float,
    exclude_uid: int,
) -> npt.NDArray:
    """Numba kernel: per-target-spike count of distinct unit ids in window.

    For each target spike ``t`` find the slice ``sorted_times[lo:hi]``
    within ``[t-tol, t+tol]`` (inclusive) and count how many distinct
    values of ``sorted_uids[lo:hi]`` it contains, optionally skipping
    ``exclude_uid``.  Parallelizes the outer loop over target spikes.
    Annotations are informational only — numba does its own type
    inference at JIT time.
    """
    n_target = target_spikes.shape[0]
    counts = np.zeros(n_target, dtype=np.int32)
    for i in prange(n_target):
        t = target_spikes[i]
        lo = np.searchsorted(sorted_times, t - tol, side="left")
        hi = np.searchsorted(sorted_times, t + tol, side="right")
        seen = np.zeros(n_units, dtype=np.bool_)
        n = 0
        for j in range(lo, hi):
            uid = sorted_uids[j]
            if uid == exclude_uid:
                continue
            if not seen[uid]:
                seen[uid] = True
                n += 1
        counts[i] = n
    return counts


def count_coincident_units(
    target_spikes: npt.NDArray,
    sorted_times: npt.NDArray,
    sorted_uids: npt.NDArray,
    n_units: int,
    tol: float,
    exclude_uid: int = -1,
) -> npt.NDArray:
    """Per-spike count of distinct other units within ``tol`` of each target spike.

    Parameters
    ----------
    target_spikes
        Sorted spike times for the target unit.
    sorted_times
        Sorted spike times of the candidate-other-unit pool, as
        produced by :func:`build_synchrony_index`.
    sorted_uids
        Parallel unit-index array, as produced by
        :func:`build_synchrony_index`.
    n_units
        Number of distinct unit indices in ``sorted_uids`` (the size of
        the unit-id space, used to size the per-spike "seen" bitmap).
    tol
        Coincidence tolerance in seconds.  A spike from another unit
        within ``[t-tol, t+tol]`` of a target spike at time ``t``
        contributes one (and only one) to that target spike's count.
    exclude_uid
        Optional unit index to skip (defaults to ``-1`` = no skip).
        Use this to omit a target unit's own spikes when querying
        against an index that includes them.

    Returns
    -------
    npt.NDArray
        ``int32`` array of length ``len(target_spikes)`` giving the
        number of *distinct* other unit ids whose spike trains have at
        least one spike within ``tol`` of each target spike.
    """
    return _count_coincident_units_kernel(
        np.ascontiguousarray(target_spikes, dtype=np.float64),
        sorted_times,
        sorted_uids,
        int(n_units),
        float(tol),
        int(exclude_uid),
    )


# ---------------------------------------------------------------------------
# Event alignment
# ---------------------------------------------------------------------------


def align_to_events(
    times: npt.NDArray,
    events: npt.NDArray,
    window: tuple[float, float],
) -> list[npt.NDArray]:
    """Align *times* to *events*, return per-event relative times.

    Parameters
    ----------
    times
        Sorted absolute times (e.g. spike times or lick times).
    events
        Sorted event times to align to.
    window
        ``(left, right)`` in seconds relative to each event.

    Returns
    -------
    list[npt.NDArray]
        One array per event, containing times relative to the event
        that fall within the window.
    """
    trials: list[npt.NDArray] = []
    for ev in events:
        lo = np.searchsorted(times, ev + window[0], side="left")
        hi = np.searchsorted(times, ev + window[1], side="right")
        trials.append(times[lo:hi] - ev)
    return trials


def align_to_events_exclude_trigger(
    times: npt.NDArray,
    events: npt.NDArray,
    window: tuple[float, float],
    tol: float = 1e-6,
) -> list[npt.NDArray]:
    """Align *times* to *events*, excluding the triggering event itself.

    Useful when *times* and *events* overlap (e.g. aligning lick times
    to lick events).

    Parameters
    ----------
    times
        Sorted absolute times.
    events
        Sorted event times.
    window
        ``(left, right)`` relative to each event.
    tol
        Times within *tol* of zero (the trigger) are excluded.

    Returns
    -------
    list[npt.NDArray]
        Per-event relative times with the trigger removed.
    """
    trials: list[npt.NDArray] = []
    for ev in events:
        lo = np.searchsorted(times, ev + window[0], side="left")
        hi = np.searchsorted(times, ev + window[1], side="right")
        rel = times[lo:hi] - ev
        trials.append(rel[np.abs(rel) > tol])
    return trials
