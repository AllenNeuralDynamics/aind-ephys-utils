"""xarray accessor for exploratory electrophysiology analysis.

This module registers the `.ephys` accessor for `xarray.DataArray`.

The accessor is intended to provide a small, composable public surface area
that interoperates with standard xarray operations (e.g., `.sel`, `.mean`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import xarray as xr

from ..ops.align import align as _align
from ..ops.baseline import baseline as _baseline
from ..ops.bin import bin as _bin
from ..ops.normalize import normalize as _normalize
from ..ops.psth import psth as _psth
from ..ops.reduce import reduce as _reduce
from ..ops.restrict import restrict as _restrict
from ..ops.smooth import smooth as _smooth
from ..plots.psth import psth as _plot_psth
from ..plots.raster import raster as _plot_raster
from ..plots.trajectory import trajectory as _plot_trajectory
from ..standards.conventions import C
from ..standards.validate import validate


@dataclass(frozen=True)
class _Help:
    """Lightweight helper object used to display `.ephys` usage hints."""

    text: str

    def __repr__(self) -> str:
        """Return the help text (so printing the object shows the guide)."""
        return self.text


@dataclass(frozen=True)
class _PlotAccessor:
    """Plotting helper for the `.ephys` accessor."""

    _obj: xr.DataArray

    def raster(self, *args: Any, **kwargs: Any):
        """Raster plot helper."""
        return _plot_raster(self._obj, *args, **kwargs)

    def psth(self, *args: Any, **kwargs: Any):
        """PSTH plot helper."""
        return _plot_psth(self._obj, *args, **kwargs)

    def trajectory(self, *args: Any, **kwargs: Any):
        """Trajectory plot helper."""
        return _plot_trajectory(self._obj, *args, **kwargs)


@xr.register_dataarray_accessor("ephys")
class EphysDataArrayAccessor:
    """
    Accessor for exploratory electrophysiology analysis.

    Public surface area (intentionally small):
      - da.ephys.validate()
      - da.ephys.align(events, to, window)
      - da.ephys.bin(...)
      - da.ephys.smooth(...)
      - da.ephys.baseline(...)
      - da.ephys.normalize(...)
      - da.ephys.psth(...)
      - da.ephys.reduce(...)
      - da.ephys.restrict(...)
      - da.ephys.plot.<...>
    """

    def __init__(self, xarray_obj: xr.DataArray):
        """Create the accessor for a specific `xarray.DataArray` instance."""
        self._obj = xarray_obj

    @property
    def guide(self) -> _Help:
        """Return a short cheatsheet describing the accessor API."""
        return _Help(
            "da.ephys cheatsheet\n"
            "  - da.ephys.validate()\n"
            "  - da.ephys.align(events, to='go_cue', window=(-0.5, 1.0))\n"
            "  - da.ephys.bin(dt=0.01, output='rate')\n"
            "  - da.ephys.smooth(method='gaussian', sigma=0.02)\n"
            "  - da.ephys.baseline(window=(-0.2, 0.0), mode='subtract')\n"
            "  - da.ephys.normalize(dim='trial', method='zscore')\n"
            "  - da.ephys.psth(dim='trial', method='mean')\n"
            "  - da.ephys.reduce(method='pca', dim='unit', n_components=10)\n"
            "  - da.ephys.restrict(window=(-0.5, 1.0))\n"
            "  - da.ephys.plot.raster(...), da.ephys.plot.psth(...)\n"
        )

    def validate(self, *, kind: Optional[str] = None) -> xr.DataArray:
        """Validate and return the underlying DataArray.

        Parameters
        ----------
        kind:
            If provided, require that the DataArray matches the inferred kind.

        Returns
        -------
        xr.DataArray
            The same object, returned for call chaining.
        """
        validate(self._obj, kind=kind)
        return self._obj

    def align(
        self,
        events: xr.DataArray,
        to: str,
        window: Tuple[float, float],
    ) -> xr.DataArray:
        """Align to an event and extract a time window.

        Parameters
        ----------
        events:
            Events object describing event times.
        to:
            Event label to align to.
        window:
            (tmin, tmax) window around the event time.

        Returns
        -------
        xr.DataArray
            Aligned DataArray.
        """
        validate(self._obj)  # friendly errors early
        return _align(self._obj, events=events, to=to, window=window)

    def bin(
        self,
        dt: float,
        window: Optional[Tuple[float, float]] = None,
        output: str = "rate",
        time_unit: str = "s",
    ) -> xr.DataArray:
        """Bin ragged spikes into a dense (trial, unit, time) representation.

        Parameters
        ----------
        dt:
            Bin width in seconds.
        output:
            Output type (e.g. "rate" or "count").
        time_unit:
            Unit for time values.

        Returns
        -------
        xr.DataArray
            Binned spikes.

        Notes
        -----
        Binning preserves compatible coordinates and updates ephys attrs.
        """
        return _bin(
            self._obj, dt=dt, window=window, output=output, time_unit=time_unit
        )

    def smooth(
        self,
        *,
        dim: str = C.time,
        method: str = "boxcar",
        sigma: Optional[float] = None,
        window: Optional[float] = None,
        boundary: str = "reflect",
    ) -> xr.DataArray:
        """Smooth a signal along a dimension.

        Notes
        -----
        Supports gaussian or boxcar smoothing over a single dimension.
        """
        return _smooth(
            self._obj,
            dim=dim,
            method=method,
            sigma=sigma,
            window=window,
            boundary=boundary,
        )

    def baseline(
        self,
        *,
        window: Tuple[float, float],
        dim: str = C.time,
        mode: str = "subtract",
    ) -> xr.DataArray:
        """Apply baseline correction over a window.

        Notes
        -----
        Applies subtract/divide/zscore baselining over a time window.
        """
        return _baseline(self._obj, window=window, dim=dim, mode=mode)

    def normalize(
        self,
        *,
        dim: Union[str, Tuple[str, ...]],
        method: str = "zscore",
    ) -> xr.DataArray:
        """Normalize data across one or more dimensions.

        Notes
        -----
        Zero-variance slices are set to 0 to avoid NaNs.
        """
        return _normalize(self._obj, dim=dim, method=method)

    def psth(
        self,
        *,
        dim: str = C.trial,
        method: str = "mean",
        group_by: Optional[Union[str, Sequence[str]]] = None,
        keep_trials: bool = False,
    ) -> xr.DataArray:
        """Reduce across trials to compute a PSTH-style summary.

        Notes
        -----
        If keep_trials=True, includes the trial-wise data alongside summary.
        """
        return _psth(
            self._obj,
            dim=dim,
            method=method,
            group_by=group_by,
            keep_trials=keep_trials,
        )

    def reduce(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Reduce data dimensionality (e.g. PCA) in an xarray-friendly way.

        Notes
        -----
        Methods include PCA variants, dPCA, and supervised linear reductions.
        """
        return _reduce(self._obj, *args, **kwargs)

    def restrict(
        self,
        *,
        window: Tuple[float, float],
        dim: str = C.time,
    ) -> xr.DataArray:
        """Restrict to a time window for dense or ragged data."""
        return _restrict(self._obj, window=window, dim=dim)

    @property
    def plot(self) -> _PlotAccessor:  # type: ignore[override]
        """Plot helper sub-accessor."""
        return _PlotAccessor(self._obj)
