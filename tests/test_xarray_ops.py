"""Tests for the xarray ops layer (xarray.py).

Each wrapper is tested for correctness, coord preservation, and
consistency with the corresponding core function.
"""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from aind_ephys_utils.standards.conventions import C
from aind_ephys_utils.xarray import (
    align,
    baseline,
    bin,
    normalize,
    psth,
    restrict,
    smooth,
)


# ------------------------------------------------------------------
# Fixture helpers
# ------------------------------------------------------------------


def _make_dense_da(
    n_trials: int = 3,
    n_units: int = 2,
    n_time: int = 50,
    dt: float = 0.01,
    seed: int = 42,
) -> xr.DataArray:
    """Standard (trial, unit, time) DataArray with non-index coords."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_trials, n_units, n_time))
    time = np.arange(n_time) * dt
    return xr.DataArray(
        data,
        dims=(C.trial, C.unit, C.time),
        coords={
            C.trial: np.arange(n_trials),
            C.unit: np.arange(n_units),
            C.time: time,
            "condition": (
                C.trial,
                np.array([["A", "B"][i % 2] for i in range(n_trials)]),
            ),
            "region": (C.unit, np.array([["V1", "M1"][i % 2] for i in range(n_units)])),
        },
        attrs={"source": "test"},
    )


def _make_ragged_da(
    n_trials: int = 2,
    n_units: int = 2,
    seed: int = 42,
) -> xr.DataArray:
    """Ragged (trial, unit) DataArray with object dtype."""
    rng = np.random.default_rng(seed)
    data = np.empty((n_trials, n_units), dtype=object)
    for t in range(n_trials):
        for u in range(n_units):
            n_spikes = rng.integers(5, 20)
            data[t, u] = np.sort(rng.uniform(-0.5, 1.0, n_spikes))
    return xr.DataArray(
        data,
        dims=(C.trial, C.unit),
        coords={
            C.trial: np.arange(n_trials),
            C.unit: np.arange(n_units),
        },
        attrs={C.attr_valid_intervals: [(-0.5, 1.0)]},
    )


def _make_events(
    anchor_times: list[float],
    event_name: str = "stim",
) -> xr.Dataset:
    """Events dataset for align tests."""
    n_trials = len(anchor_times)
    return xr.Dataset(
        {"t": (("trial", "event"), np.array(anchor_times).reshape(-1, 1))},
        coords={
            "trial": np.arange(n_trials),
            "event": [event_name],
        },
    )


# ------------------------------------------------------------------
# smooth
# ------------------------------------------------------------------


class SmoothTest(unittest.TestCase):
    def test_shape_preserved(self):
        da = _make_dense_da()
        out = smooth(da, sigma=0.03)
        self.assertEqual(out.dims, da.dims)
        self.assertEqual(out.shape, da.shape)

    def test_coords_preserved(self):
        da = _make_dense_da()
        out = smooth(da, sigma=0.03)
        assert_array_equal(
            out.coords["condition"].values, da.coords["condition"].values
        )
        assert_array_equal(out.coords["region"].values, da.coords["region"].values)

    def test_attrs_preserved(self):
        da = _make_dense_da()
        out = smooth(da, sigma=0.03)
        self.assertEqual(out.attrs["source"], "test")

    def test_gaussian_changes_data(self):
        da = _make_dense_da()
        out = smooth(da, sigma=0.03)
        self.assertFalse(np.array_equal(out.values, da.values))

    def test_boxcar(self):
        da = _make_dense_da()
        out = smooth(da, method="boxcar", window=0.03)
        self.assertEqual(out.shape, da.shape)

    def test_non_canonical_dim_order(self):
        da = _make_dense_da().transpose(C.unit, C.time, C.trial)
        out = smooth(da, sigma=0.03)
        self.assertEqual(out.dims, (C.unit, C.time, C.trial))
        self.assertEqual(out.shape, da.shape)

    def test_missing_dim_raises(self):
        da = _make_dense_da()
        with self.assertRaises(ValueError):
            smooth(da, sigma=0.03, dim="channel")


# ------------------------------------------------------------------
# baseline
# ------------------------------------------------------------------


class BaselineTest(unittest.TestCase):
    def test_subtract(self):
        data = np.array([[[1.0, 2.0, 3.0, 4.0]]])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit, C.time),
            coords={C.trial: [0], C.unit: [0], C.time: time},
        )
        out = baseline(da, window=(0.0, 1.0), mode="subtract")
        assert_allclose(out.values[0, 0], [-0.5, 0.5, 1.5, 2.5])

    def test_divide(self):
        data = np.array([[[2.0, 4.0, 6.0, 8.0]]])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit, C.time),
            coords={C.trial: [0], C.unit: [0], C.time: time},
        )
        out = baseline(da, window=(0.0, 1.0), mode="divide")
        assert_allclose(out.values[0, 0], [2.0 / 3.0, 4.0 / 3.0, 2.0, 8.0 / 3.0])

    def test_zscore_zero_std(self):
        data = np.array([[[5.0, 5.0, 10.0, 15.0]]])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit, C.time),
            coords={C.trial: [0], C.unit: [0], C.time: time},
        )
        out = baseline(da, window=(0.0, 1.0), mode="zscore")
        self.assertFalse(np.any(np.isnan(out.values)))
        assert_allclose(out.values[0, 0, :2], [0.0, 0.0])

    def test_coords_preserved(self):
        da = _make_dense_da()
        out = baseline(da, window=(0.0, 0.2))
        assert_array_equal(
            out.coords["condition"].values, da.coords["condition"].values
        )
        assert_array_equal(out.coords["region"].values, da.coords["region"].values)

    def test_empty_window_raises(self):
        da = _make_dense_da()
        with self.assertRaises(ValueError):
            baseline(da, window=(5.0, 6.0))


# ------------------------------------------------------------------
# normalize
# ------------------------------------------------------------------


class NormalizeTest(unittest.TestCase):
    def test_zscore_across_trials(self):
        da = _make_dense_da(n_trials=10)
        out = normalize(da, dim=C.trial, method="zscore")
        assert_allclose(out.values.mean(axis=0), 0.0, atol=1e-10)

    def test_minmax_range(self):
        da = _make_dense_da(n_trials=10)
        out = normalize(da, dim=C.trial, method="minmax")
        self.assertGreaterEqual(out.values.min(), -1e-10)
        self.assertLessEqual(out.values.max(), 1.0 + 1e-10)

    def test_zero_variance(self):
        data = np.ones((5, 2, 10))
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit, C.time),
            coords={C.trial: range(5), C.unit: range(2), C.time: range(10)},
        )
        out = normalize(da, dim=C.trial, method="zscore")
        self.assertFalse(np.any(np.isnan(out.values)))
        assert_allclose(out.values, 0.0)

    def test_multi_dim(self):
        data = np.ones((5, 3, 10)) * 3.0
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit, C.time),
            coords={C.trial: range(5), C.unit: range(3), C.time: range(10)},
        )
        out = normalize(da, dim=(C.trial, C.time), method="zscore")
        assert_allclose(out.values, 0.0)

    def test_unknown_dim_raises(self):
        da = _make_dense_da()
        with self.assertRaises(ValueError):
            normalize(da, dim="channel")

    def test_preserves_original_dim_order(self):
        da = _make_dense_da().transpose(C.unit, C.time, C.trial)
        out = normalize(da, dim=C.trial, method="zscore")
        self.assertEqual(out.dims, (C.unit, C.time, C.trial))


# ------------------------------------------------------------------
# psth
# ------------------------------------------------------------------


class PsthTest(unittest.TestCase):
    def test_mean_removes_trial_dim(self):
        da = _make_dense_da()
        out = psth(da, method="mean")
        self.assertNotIn(C.trial, out.dims)
        self.assertEqual(out.shape, (da.sizes[C.unit], da.sizes[C.time]))

    def test_median(self):
        data = np.array(
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
                [[5.0, 6.0]],
            ]
        )
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit, C.time),
            coords={C.trial: range(3), C.unit: [0], C.time: [0.0, 1.0]},
        )
        out = psth(da, method="median")
        assert_allclose(out.values, [[3.0, 4.0]])

    def test_with_labels(self):
        data = np.array(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                [[5.0], [6.0]],
                [[7.0], [8.0]],
            ]
        )
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit, C.time),
            coords={C.trial: range(4), C.unit: [0, 1], C.time: [0.0]},
        )
        labels = np.array(["A", "B", "A", "B"])
        out = psth(da, labels=labels)
        # Group A: trials 0, 2
        assert_allclose(out.sel({C.trial: "A"}).values, [[(1 + 5) / 2], [(2 + 6) / 2]])
        # Group B: trials 1, 3
        assert_allclose(out.sel({C.trial: "B"}).values, [[(3 + 7) / 2], [(4 + 8) / 2]])

    def test_unit_coords_survive(self):
        da = _make_dense_da()
        out = psth(da)
        self.assertIn(C.unit, out.coords)
        assert_array_equal(out.coords[C.unit].values, da.coords[C.unit].values)
        # Non-index coord on unit dim should survive too
        self.assertIn("region", out.coords)


# ------------------------------------------------------------------
# restrict
# ------------------------------------------------------------------


class RestrictTest(unittest.TestCase):
    def test_dense_restrict(self):
        da = _make_dense_da(n_time=10, dt=1.0)
        out = restrict(da, (2.0, 5.0))
        assert_array_equal(out.coords[C.time].values, [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(out.sizes[C.time], 4)

    def test_dense_coords_preserved(self):
        da = _make_dense_da(n_time=10, dt=1.0)
        out = restrict(da, (2.0, 5.0))
        assert_array_equal(
            out.coords["condition"].values, da.coords["condition"].values
        )

    def test_ragged_restrict(self):
        da = _make_ragged_da()
        out = restrict(da, (0.0, 0.5))
        self.assertEqual(out.dtype, object)
        for t in range(out.sizes[C.trial]):
            for u in range(out.sizes[C.unit]):
                arr = out.values[t, u]
                if arr.size > 0:
                    self.assertGreaterEqual(arr.min(), 0.0)
                    self.assertLessEqual(arr.max(), 0.5)


# ------------------------------------------------------------------
# bin
# ------------------------------------------------------------------


class BinTest(unittest.TestCase):
    def test_basic_binning(self):
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([0.05, 0.15])
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit),
            coords={C.trial: [0], C.unit: [0]},
        )
        out = bin(da, dt=0.1, window=(0.0, 0.3), output="count")
        self.assertEqual(out.dims, (C.trial, C.unit, C.time))
        self.assertEqual(out.shape, (1, 1, 3))
        assert_allclose(out.values[0, 0], [1.0, 1.0, 0.0])

    def test_rate_vs_count(self):
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([0.05, 0.15])
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit),
            coords={C.trial: [0], C.unit: [0]},
        )
        rate = bin(da, dt=0.1, window=(0.0, 0.3), output="rate")
        count = bin(da, dt=0.1, window=(0.0, 0.3), output="count")
        assert_allclose(rate.values, count.values / 0.1)

    def test_window_from_attrs(self):
        da = _make_ragged_da()
        out = bin(da, dt=0.1)
        self.assertIn(C.time, out.dims)

    def test_non_object_raises(self):
        da = _make_dense_da()
        with self.assertRaises(ValueError):
            bin(da, dt=0.1)

    def test_unit_only(self):
        data = np.empty((2,), dtype=object)
        data[0] = np.array([0.05, 0.15])
        data[1] = np.array([0.25])
        da = xr.DataArray(
            data,
            dims=(C.unit,),
            coords={C.unit: [0, 1]},
        )
        out = bin(da, dt=0.1, window=(0.0, 0.3), output="count")
        self.assertEqual(out.dims, (C.unit, C.time))
        self.assertEqual(out.shape, (2, 3))

    def test_trial_only(self):
        data = np.empty((2,), dtype=object)
        data[0] = np.array([0.05, 0.15])
        data[1] = np.array([0.25])
        da = xr.DataArray(
            data,
            dims=(C.trial,),
            coords={C.trial: [0, 1]},
        )
        out = bin(da, dt=0.1, window=(0.0, 0.3), output="count")
        self.assertEqual(out.dims, (C.trial, C.time))
        self.assertEqual(out.shape, (2, 3))


# ------------------------------------------------------------------
# align
# ------------------------------------------------------------------


class AlignTest(unittest.TestCase):
    def test_basic_alignment(self):
        # Single "trial" with all session spikes, 2 units
        data = np.empty((1, 2), dtype=object)
        data[0, 0] = np.arange(10, dtype=float)  # 0..9
        data[0, 1] = np.arange(10, dtype=float) + 0.3
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit),
            coords={C.trial: [0], C.unit: [0, 1]},
        )
        events = _make_events([2.0, 5.0])
        out = align(da, events=events, to="stim", window=(-0.5, 0.5))
        self.assertEqual(out.dims, (C.trial, C.unit))
        self.assertEqual(out.shape, (2, 2))
        # Trial 0 (anchor=2.0), unit 0: spike at 2.0 → 0.0
        assert_allclose(out.values[0, 0], [0.0])

    def test_events_dataarray(self):
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([0.5, 1.5, 2.5])
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit),
            coords={C.trial: [0], C.unit: [0]},
        )
        events_da = xr.DataArray(
            [[1.0]],
            dims=("trial", "event"),
            coords={"trial": [0], "event": ["stim"]},
        )
        out = align(da, events=events_da, to="stim", window=(-0.6, 0.6))
        self.assertEqual(out.sizes[C.trial], 1)
        assert_allclose(out.values[0, 0], [-0.5, 0.5])

    def test_unit_coords_preserved(self):
        data = np.empty((1, 2), dtype=object)
        data[0, 0] = np.array([1.0, 2.0])
        data[0, 1] = np.array([1.5, 2.5])
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit),
            coords={
                C.trial: [0],
                C.unit: ["neuron_a", "neuron_b"],
                "region": (C.unit, ["V1", "M1"]),
            },
        )
        events = _make_events([1.5])
        out = align(da, events=events, to="stim", window=(-1.0, 1.0))
        assert_array_equal(out.coords[C.unit].values, ["neuron_a", "neuron_b"])
        self.assertIn("region", out.coords)

    def test_invalid_window_raises(self):
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([1.0])
        da = xr.DataArray(
            data,
            dims=(C.trial, C.unit),
            coords={C.trial: [0], C.unit: [0]},
        )
        events = _make_events([1.0])
        with self.assertRaises(ValueError):
            align(da, events=events, to="stim", window=(1.0, -1.0))


# ------------------------------------------------------------------
# Consistency with core
# ------------------------------------------------------------------


class ConsistencyTest(unittest.TestCase):
    def test_smooth_matches_core(self):
        from aind_ephys_utils import core

        da = _make_dense_da()
        dt = core.infer_dt(da.coords[C.time].values.astype(float))
        xr_result = smooth(da, sigma=0.03)
        np_result = core.smooth(da.values.astype(float), dt, sigma=0.03)
        assert_allclose(xr_result.values, np_result)

    def test_baseline_matches_core(self):
        from aind_ephys_utils import core

        da = _make_dense_da()
        time = da.coords[C.time].values.astype(float)
        xr_result = baseline(da, window=(0.0, 0.2), mode="subtract")
        np_result = core.baseline(
            da.values.astype(float), time, (0.0, 0.2), mode="subtract"
        )
        assert_allclose(xr_result.values, np_result)


# ------------------------------------------------------------------
# .pipe() chaining
# ------------------------------------------------------------------


class PipeChainTest(unittest.TestCase):
    def test_full_pipeline(self):
        da = _make_dense_da(n_trials=4, n_time=100)
        labels = np.array(["A", "B", "A", "B"])
        result = (
            da.pipe(smooth, sigma=0.03)
            .pipe(baseline, window=(0.0, 0.2))
            .pipe(normalize, dim=C.trial)
            .pipe(psth, labels=labels)
        )
        # Should have trial dim replaced by group labels
        self.assertIn(C.trial, result.dims)
        self.assertEqual(result.sizes[C.trial], 2)
        assert_array_equal(result.coords[C.trial].values, ["A", "B"])
        # unit and time dims should survive
        self.assertIn(C.unit, result.dims)
        self.assertIn(C.time, result.dims)


if __name__ == "__main__":
    unittest.main()
