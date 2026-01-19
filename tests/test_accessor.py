"""Tests for DataArray accessor behaviors."""

import unittest

import numpy as np
import xarray as xr


class AccessorTest(unittest.TestCase):
    """Exercise the .ephys accessor methods."""

    def test_bin_and_plot_accessor(self) -> None:
        """Bin ragged spikes and access plot helpers."""
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([0.1, 0.2])
        spikes = xr.DataArray(
            data=data,
            dims=("trial", "unit"),
            coords={"trial": [0], "unit": [0]},
        )
        binned = spikes.ephys.bin(0.1, tlim=(0.0, 0.3))
        self.assertIn("time", binned.dims)
        plotter = spikes.ephys.plot
        self.assertTrue(hasattr(plotter, "raster"))

    def test_reduce_returns_dataset(self) -> None:
        """Return dataset with projections and weights."""
        da = xr.DataArray(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            dims=("trial", "unit"),
            coords={"trial": [0, 1], "unit": [0, 1]},
        )
        ds = da.ephys.reduce(
            method="pca", dim="unit", n_components=1, return_dataset=True
        )
        self.assertIn("projections", ds)
        self.assertIn("weights", ds)
