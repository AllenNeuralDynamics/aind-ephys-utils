"""Tests for xarray <-> dataclass type round-trip conversions."""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from aind_ephys_utils.adapters.xarray import (
    binned_to_xarray,
    ragged_to_xarray,
    xarray_to_binned,
    xarray_to_ragged,
)
from aind_ephys_utils.types import BinnedSpikes, RaggedSpikes


class BinnedRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.binned = BinnedSpikes(
            data=np.arange(24, dtype=float).reshape(2, 3, 4),
            time=np.array([0.0, 0.1, 0.2, 0.3]),
            trial_meta={"condition": np.array(["A", "B"])},
            unit_meta={
                "unit_id": np.array([10, 20, 30]),
                "region": np.array(["V1", "M1", "V1"]),
            },
        )

    def test_round_trip(self):
        da = binned_to_xarray(self.binned)
        recovered = xarray_to_binned(da)
        assert_array_equal(recovered.data, self.binned.data)
        assert_array_equal(recovered.time, self.binned.time)

    def test_metadata_preserved(self):
        da = binned_to_xarray(self.binned)
        recovered = xarray_to_binned(da)
        assert_array_equal(
            recovered.trial_meta["condition"], ["A", "B"]
        )
        assert_array_equal(
            recovered.unit_meta["region"], ["V1", "M1", "V1"]
        )

    def test_xarray_has_correct_dims(self):
        da = binned_to_xarray(self.binned)
        self.assertEqual(da.dims, ("trial", "unit", "time"))
        self.assertEqual(da.shape, (2, 3, 4))

    def test_unit_id_as_coord(self):
        da = binned_to_xarray(self.binned)
        assert_array_equal(da.coords["unit"].values, [10, 20, 30])

    def test_metadata_as_coords(self):
        da = binned_to_xarray(self.binned)
        self.assertIn("condition", da.coords)
        assert_array_equal(da.coords["condition"].values, ["A", "B"])
        self.assertIn("region", da.coords)
        assert_array_equal(da.coords["region"].values, ["V1", "M1", "V1"])

    def test_from_xarray_transposes(self):
        """Non-canonical dim order should be handled."""
        da = xr.DataArray(
            data=np.zeros((3, 4, 2)),
            dims=("unit", "time", "trial"),
            coords={
                "unit": [0, 1, 2],
                "time": [0.0, 0.1, 0.2, 0.3],
                "trial": [0, 1],
            },
        )
        bs = xarray_to_binned(da)
        self.assertEqual(bs.data.shape, (2, 3, 4))


class RaggedRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.ragged = RaggedSpikes(
            spike_times=[
                [np.array([0.1, 0.2]), np.array([0.3])],
                [np.array([0.5]), np.array([0.6, 0.7, 0.8])],
            ],
            n_trials=2,
            n_units=2,
            time_window=(-0.5, 1.0),
            trial_meta={"condition": np.array(["A", "B"])},
            unit_meta={
                "unit_id": np.array([10, 20]),
                "region": np.array(["V1", "M1"]),
            },
        )

    def test_round_trip(self):
        da = ragged_to_xarray(self.ragged)
        recovered = xarray_to_ragged(da)
        self.assertEqual(recovered.n_trials, 2)
        self.assertEqual(recovered.n_units, 2)
        assert_allclose(
            recovered.spike_times[0][0], self.ragged.spike_times[0][0]
        )
        assert_allclose(
            recovered.spike_times[1][1], self.ragged.spike_times[1][1]
        )

    def test_time_window_preserved(self):
        da = ragged_to_xarray(self.ragged)
        recovered = xarray_to_ragged(da)
        self.assertEqual(recovered.time_window, (-0.5, 1.0))

    def test_metadata_preserved(self):
        da = ragged_to_xarray(self.ragged)
        recovered = xarray_to_ragged(da)
        assert_array_equal(
            recovered.trial_meta["condition"], ["A", "B"]
        )
        assert_array_equal(
            recovered.unit_meta["region"], ["V1", "M1"]
        )

    def test_xarray_has_correct_structure(self):
        da = ragged_to_xarray(self.ragged)
        self.assertEqual(da.dims, ("trial", "unit"))
        self.assertEqual(da.dtype, object)
        self.assertEqual(da.shape, (2, 2))

    def test_none_entries_handled(self):
        """None entries in ragged xarray should become empty arrays."""
        data = np.empty((2, 1), dtype=object)
        data[0, 0] = np.array([1.0, 2.0])
        data[1, 0] = None
        da = xr.DataArray(
            data, dims=("trial", "unit"),
            coords={"trial": [0, 1], "unit": [0]},
        )
        rs = xarray_to_ragged(da)
        assert_allclose(rs.spike_times[0][0], [1.0, 2.0])
        self.assertEqual(rs.spike_times[1][0].size, 0)


class InteropMethodsTest(unittest.TestCase):
    """Test .to_xarray() and .from_xarray() on the dataclass types."""

    def test_binned_to_xarray(self):
        bs = BinnedSpikes(
            data=np.zeros((2, 3, 4)),
            time=np.arange(4) * 0.1,
        )
        da = bs.to_xarray()
        self.assertIsInstance(da, xr.DataArray)
        self.assertEqual(da.dims, ("trial", "unit", "time"))

    def test_binned_from_xarray(self):
        da = xr.DataArray(
            data=np.zeros((2, 3, 4)),
            dims=("trial", "unit", "time"),
            coords={
                "trial": [0, 1],
                "unit": [0, 1, 2],
                "time": [0.0, 0.1, 0.2, 0.3],
            },
        )
        bs = BinnedSpikes.from_xarray(da)
        self.assertEqual(bs.data.shape, (2, 3, 4))

    def test_ragged_to_xarray(self):
        rs = RaggedSpikes(
            spike_times=[[np.array([0.1, 0.2])]],
            n_trials=1,
            n_units=1,
        )
        da = rs.to_xarray()
        self.assertIsInstance(da, xr.DataArray)
        self.assertEqual(da.dims, ("trial", "unit"))

    def test_ragged_from_xarray(self):
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([0.1, 0.2])
        da = xr.DataArray(
            data, dims=("trial", "unit"),
            coords={"trial": [0], "unit": [0]},
        )
        rs = RaggedSpikes.from_xarray(da)
        self.assertEqual(rs.n_trials, 1)
        assert_allclose(rs.spike_times[0][0], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
