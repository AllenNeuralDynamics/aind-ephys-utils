"""xarray accessor for exploratory electrophysiology analysis.

This module registers the `.ephys` accessor for `xarray.DataArray`.

The accessor is intended to provide a small, composable public surface area
that interoperates with standard xarray operations (e.g., `.sel`, `.mean`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import xarray as xr

from ..core.conventions import C
from ..core.validate import validate
from ..ops.align import align as _align
from ..ops.baseline import baseline as _baseline
from ..ops.bin import bin as _bin
from ..ops.normalize import normalize as _normalize
from ..ops.psth import psth as _psth
from ..ops.reduce import reduce as _reduce
from ..ops.smooth import smooth as _smooth


@dataclass(frozen=True)
class _Help:
    """Lightweight helper object used to display `.ephys` usage hints."""

    text: str

    def __repr__(self) -> str:
        """Return the help text (so printing the object shows the guide)."""
        return self.text


@xr.register_dataarray_accessor("ephys")
class EphysDataArrayAccessor:
    """
    Accessor for exploratory electrophysiology analysis.

    Public surface area (intentionally small):
      - da.ephys.validate()
      - da.ephys.align(events, to, window)
      - da.ephys.bin(...)      [stub for now]
      - da.ephys.smooth(...)   [stub for now]
      - da.ephys.baseline(...) [stub for now]
      - da.ephys.normalize(...) [stub for now]
      - da.ephys.psth(...)     [stub for now]
      - da.ephys.reduce(...)   [stub for now]
      - da.ephys.plot.<...>    [optional accessor, stub for now]
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
            "  - da.ephys.psth(dim='trial', reduce='mean')\n"
            "  - da.ephys.reduce(method='pca', dim='unit', n=10, stack=('trial','time'))\n"
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

    # --- stubs below: you’ll fill these in next ---
    def bin(
        self,
        dt: float,
        tlim: Optional[Tuple[float, float]] = None,
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
        This is a stub in the current refactor.
        """
        return _bin(
            self._obj, dt=dt, tlim=tlim, output=output, time_unit=time_unit
        )

    def smooth(
        self,
        *,
        dim: str = C.time,
        method: str = "gaussian",
        sigma: Optional[float] = None,
        window: Optional[float] = None,
        boundary: str = "reflect",
    ) -> xr.DataArray:
        """Smooth a signal along a dimension.

        Notes
        -----
        This is a stub in the current refactor.
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
        This is a stub in the current refactor.
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
        This is a stub in the current refactor.
        """
        return _normalize(self._obj, dim=dim, method=method)

    def psth(
        self,
        *,
        dim: str = C.trial,
        reduce: str = "mean",
        keep_trials: bool = False,
    ) -> xr.DataArray:
        """Reduce across trials to compute a PSTH-style summary.

        Notes
        -----
        This is a stub in the current refactor.
        """
        return _psth(
            self._obj, dim=dim, reduce=reduce, keep_trials=keep_trials
        )

    def reduce(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Reduce data dimensionality (e.g. PCA) in an xarray-friendly way.

        Notes
        -----
        This is a stub in the current refactor.
        """
        return _reduce(self._obj, *args, **kwargs)

    def plot(self, *args: Any, **kwargs: Any) -> xr.DataArray:
        """Plot helper entry point.

        Notes
        -----
        This is a stub in the current refactor. Plotting is available as functions
        under `aind_ephys_utils.plot`.
        """
        raise NotImplementedError("da.ephys.plot is not implemented yet.")
