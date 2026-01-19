"""Tests for ops module implementations."""

import unittest

import numpy as np
import xarray as xr

from aind_ephys_utils.ops import baseline, bin, normalize, psth, reduce, smooth


class OpsTest(unittest.TestCase):
    """Basic tests for ops functions."""

    def test_bin_ragged_trial_unit(self) -> None:
        spikes = xr.DataArray(
            data=np.array(
                [[np.array([0.05, 0.15]), np.array([0.02])]],
                dtype=object,
            ),
            dims=("trial", "unit"),
            coords={"trial": [0], "unit": [0, 1]},
        )
        binned = bin(spikes, dt=0.1, tlim=(0.0, 0.2), output="count")
        self.assertEqual(binned.dims, ("trial", "unit", "time"))
        self.assertEqual(binned.shape, (1, 2, 2))
        np.testing.assert_array_equal(binned.values[0, 0], [1, 1])
        np.testing.assert_array_equal(binned.values[0, 1], [1, 0])

    def test_smooth_gaussian(self) -> None:
        da = xr.DataArray(
            [0.0, 0.0, 1.0, 0.0, 0.0],
            dims=("time",),
            coords={"time": np.arange(5) * 0.1},
        )
        out = smooth(da, sigma=0.1)
        self.assertEqual(out.shape, da.shape)
        self.assertAlmostEqual(float(out.sum()), 1.0, places=6)

    def test_baseline_subtract(self) -> None:
        da = xr.DataArray(
            [[1.0, 2.0, 3.0, 4.0]],
            dims=("trial", "time"),
            coords={"time": [0.0, 1.0, 2.0, 3.0]},
        )
        out = baseline(da, window=(0.0, 1.0), mode="subtract")
        np.testing.assert_allclose(out.values[0, 0:2], [-0.5, 0.5])

    def test_normalize_zscore(self) -> None:
        da = xr.DataArray(
            [[0.0, 2.0], [2.0, 4.0]],
            dims=("trial", "time"),
            coords={"trial": [0, 1], "time": [0, 1]},
        )
        out = normalize(da, dim="trial", method="zscore")
        np.testing.assert_allclose(out.mean(dim="trial"), 0.0, atol=1e-7)

    def test_psth_mean(self) -> None:
        da = xr.DataArray(
            [[1.0, 3.0], [3.0, 5.0]],
            dims=("trial", "time"),
            coords={"trial": [0, 1], "time": [0, 1]},
        )
        out = psth(da, dim="trial", reduce="mean")
        np.testing.assert_allclose(out.values, [2.0, 4.0])

    def test_reduce_pca(self) -> None:
        da = xr.DataArray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dims=("trial", "unit"),
            coords={"trial": [0, 1, 2], "unit": [0, 1]},
        )
        out = reduce(da, method="pca", dim="unit", n_components=1, stack=("trial",))
        self.assertIn("component", out.dims)
        self.assertIn("trial", out.dims)
        self.assertEqual(out.sizes["component"], 1)
