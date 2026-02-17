"""Typed containers for electrophysiology data.

Two primary types:

- :class:`RaggedSpikes` -- variable-length spike trains per trial/unit
- :class:`BinnedSpikes` -- dense ``(trial, unit, time)`` arrays

Both carry metadata dicts and support constructors from raw arrays and
interop with xarray via ``.to_xarray()`` / ``.from_xarray()``.

For operations on these types, use :mod:`aind_ephys_utils.core` (pure
numpy) or :mod:`aind_ephys_utils.xarray` (DataArray in/out with
``.pipe()`` chaining).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from . import core


# ------------------------------------------------------------------
# RaggedSpikes
# ------------------------------------------------------------------


@dataclass
class RaggedSpikes:
    """Variable-length spike times organized by trial and unit.

    Attributes
    ----------
    spike_times : list of list of ndarray
        ``spike_times[trial][unit]`` is a sorted 1-D float64 array of
        spike times (in seconds, relative to trial anchor or session).
    n_trials : int
        Number of trials.
    n_units : int
        Number of units.
    time_window : tuple of float or None
        ``(tmin, tmax)`` valid interval, if known.
    trial_meta : dict of {str: ndarray}
        Per-trial metadata. Each value has shape ``(n_trials,)``.
    unit_meta : dict of {str: ndarray}
        Per-unit metadata. Each value has shape ``(n_units,)``.
    """

    spike_times: list[list[NDArray[np.float64]]]
    n_trials: int
    n_units: int
    time_window: tuple[float, float] | None = None
    trial_meta: dict[str, NDArray] = field(default_factory=dict)
    unit_meta: dict[str, NDArray] = field(default_factory=dict)

    # ---- constructors ----

    @classmethod
    def from_arrays(
        cls,
        spike_times: (list[NDArray[np.float64]] | dict[object, NDArray[np.float64]]),
        anchor_times: NDArray[np.float64] | None = None,
        window: tuple[float, float] | None = None,
        *,
        unit_ids: Sequence | None = None,
        trial_meta: dict[str, NDArray] | None = None,
        unit_meta: dict[str, NDArray] | None = None,
    ) -> RaggedSpikes:
        """Construct from per-unit spike time arrays.

        Parameters
        ----------
        spike_times : list of ndarray, or dict
            Per-unit spike times. If list, ``spike_times[u]`` is a 1-D
            sorted array of spike times for unit *u* (session time).
            If dict, keys are unit IDs and values are 1-D arrays.
        anchor_times : ndarray, shape (n_trials,), optional
            If provided, spikes are aligned to these anchors with the
            given *window*, producing trial-segmented ragged data. If
            ``None``, all spikes are placed in a single "trial".
        window : (float, float), optional
            Required when *anchor_times* is provided.
        unit_ids : sequence, optional
            Unit identifiers. Defaults to ``range(n_units)``.
        trial_meta : dict, optional
            Per-trial metadata arrays.
        unit_meta : dict, optional
            Per-unit metadata arrays.
        """
        # Normalize spike_times to list form.
        if isinstance(spike_times, dict):
            if unit_ids is None:
                unit_ids = list(spike_times.keys())
            spike_list = [spike_times[k] for k in unit_ids]
        else:
            spike_list = list(spike_times)

        n_units = len(spike_list)
        if unit_ids is None:
            unit_ids = list(range(n_units))

        umeta = {"unit_id": np.asarray(unit_ids)}
        if unit_meta:
            umeta.update(unit_meta)

        if anchor_times is not None:
            if window is None:
                raise ValueError("window is required when anchor_times is provided.")
            aligned = core.align(spike_list, np.asarray(anchor_times), window)
            return cls(
                spike_times=aligned,
                n_trials=len(anchor_times),
                n_units=n_units,
                time_window=window,
                trial_meta=trial_meta or {},
                unit_meta=umeta,
            )

        # Single "trial" containing all spikes.
        single_trial = [np.asarray(s, dtype=np.float64) for s in spike_list]
        return cls(
            spike_times=[single_trial],
            n_trials=1,
            n_units=n_units,
            time_window=None,
            trial_meta=trial_meta or {},
            unit_meta=umeta,
        )

    # ---- interop ----

    def to_xarray(self):
        """Convert to ``xr.DataArray`` with object dtype."""
        from .adapters.xarray import ragged_to_xarray

        return ragged_to_xarray(self)

    @classmethod
    def from_xarray(cls, da) -> RaggedSpikes:
        """Construct from an ``xr.DataArray`` with object dtype."""
        from .adapters.xarray import xarray_to_ragged

        return xarray_to_ragged(da)


# ------------------------------------------------------------------
# BinnedSpikes
# ------------------------------------------------------------------


@dataclass
class BinnedSpikes:
    """Dense binned spike data with fixed ``(trial, unit, time)`` layout.

    Attributes
    ----------
    data : ndarray, shape (n_trials, n_units, n_time)
        Spike counts or rates.
    time : ndarray, shape (n_time,)
        Bin center times.
    trial_meta : dict of {str: ndarray}
        Per-trial metadata. Each value has shape ``(n_trials,)``.
    unit_meta : dict of {str: ndarray}
        Per-unit metadata. Each value has shape ``(n_units,)``.
    """

    data: NDArray[np.float64]
    time: NDArray[np.float64]
    trial_meta: dict[str, NDArray] = field(default_factory=dict)
    unit_meta: dict[str, NDArray] = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return self.data.shape[0]

    @property
    def n_units(self) -> int:
        return self.data.shape[1]

    @property
    def n_time(self) -> int:
        return self.data.shape[2]

    # ---- constructors ----

    @classmethod
    def from_arrays(
        cls,
        data: NDArray[np.float64],
        time: NDArray[np.float64],
        *,
        axes: tuple[str, ...] | None = None,
        trial_meta: dict[str, NDArray] | None = None,
        unit_meta: dict[str, NDArray] | None = None,
    ) -> BinnedSpikes:
        """Construct from arrays with optional axis transposition.

        Parameters
        ----------
        data : ndarray
            Dense spike data.
        time : ndarray, shape (n_time,)
            Time bin centers.
        axes : tuple of str, optional
            Current axis names, e.g. ``("unit", "trial", "time")``.
            Data will be transposed to canonical
            ``(trial, unit, time)``. If ``None``, assumes already in
            canonical order.
        trial_meta, unit_meta : dict, optional
            Metadata arrays.
        """
        if axes is not None:
            canonical = ("trial", "unit", "time")
            if sorted(axes) != sorted(canonical):
                raise ValueError(f"axes must contain exactly {canonical}, got {axes}.")
            perm = tuple(axes.index(c) for c in canonical)
            data = np.transpose(data, perm)
        return cls(
            data=np.asarray(data, dtype=np.float64),
            time=np.asarray(time, dtype=np.float64),
            trial_meta=trial_meta or {},
            unit_meta=unit_meta or {},
        )

    # ---- interop ----

    def to_xarray(self):
        """Convert to ``xr.DataArray`` with named dimensions."""
        from .adapters.xarray import binned_to_xarray

        return binned_to_xarray(self)

    @classmethod
    def from_xarray(cls, da) -> BinnedSpikes:
        """Construct from an ``xr.DataArray``."""
        from .adapters.xarray import xarray_to_binned

        return xarray_to_binned(da)
