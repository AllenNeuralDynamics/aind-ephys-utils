"""Tests for the xarray sliding-PSTH skin over the numpy core."""

import unittest

import numpy as np
import xarray as xr

import aind_ephys_utils  # noqa: F401  (registers the .ephys accessor)
from aind_ephys_utils.ops import sliding_psth
from aind_ephys_utils.spiketrain.psth import sliding_window_psth


def _ragged_spikes(rng, n_trials, n_units, rate, t=1.0):
    """Build a ragged (trial, unit) spikes DataArray."""
    data = np.empty((n_trials, n_units), dtype=object)
    for i in range(n_trials):
        for j in range(n_units):
            n = rng.poisson(rate * t)
            data[i, j] = np.sort(rng.uniform(0.0, t, size=n))
    return xr.DataArray(
        data,
        dims=("trial", "unit"),
        coords={"trial": np.arange(n_trials), "unit": np.arange(n_units)},
    )


class SlidingPSTHSkinTest(unittest.TestCase):
    """The xarray skin must reuse, and agree with, the numpy core."""

    def test_shape_and_rate_recovery(self) -> None:
        """Skin returns (unit, time) and recovers a flat Poisson rate."""
        rng = np.random.default_rng(0)
        da = _ragged_spikes(rng, n_trials=300, n_units=3, rate=20.0)
        out = sliding_psth(
            da, window_size=0.1, bin_size=0.01, window=(0.0, 1.0)
        )
        self.assertEqual(out.dims, ("unit", "time"))
        self.assertEqual(out.sizes["unit"], 3)
        central = (out["time"] > 0.2) & (out["time"] < 0.8)
        self.assertAlmostEqual(
            float(out.isel(unit=0).where(central, drop=True).mean()),
            20.0,
            delta=2.5,
        )

    def test_skin_matches_core(self) -> None:
        """Per-unit skin output equals the raw-array core exactly."""
        rng = np.random.default_rng(1)
        da = _ragged_spikes(rng, n_trials=40, n_units=2, rate=15.0)
        out = sliding_psth(
            da, window_size=0.08, bin_size=0.005, window=(0.0, 1.0)
        )
        trials_u1 = [da.values[i, 1] for i in range(da.sizes["trial"])]
        time_bins, rate, _ = sliding_window_psth(
            trials_u1,
            window_size=0.08,
            bin_size=0.005,
            time_range=(0.0, 1.0),
        )
        np.testing.assert_array_almost_equal(out["time"].values, time_bins)
        np.testing.assert_array_almost_equal(out.isel(unit=1).values, rate)

    def test_accessor_delegates_to_op(self) -> None:
        """da.ephys.sliding_psth matches the ops function."""
        rng = np.random.default_rng(2)
        da = _ragged_spikes(rng, n_trials=20, n_units=2, rate=10.0)
        via_op = sliding_psth(da, window_size=0.1, window=(0.0, 1.0))
        via_acc = da.ephys.sliding_psth(window_size=0.1, window=(0.0, 1.0))
        np.testing.assert_array_almost_equal(via_acc.values, via_op.values)

    def test_binned_attr_is_set(self) -> None:
        """Output advertises itself as binned for downstream ops."""
        rng = np.random.default_rng(3)
        da = _ragged_spikes(rng, n_trials=10, n_units=1, rate=10.0)
        out = sliding_psth(da, window_size=0.1, window=(0.0, 1.0))
        self.assertEqual(out.attrs.get("ephys.kind"), "binned")


if __name__ == "__main__":
    unittest.main()
