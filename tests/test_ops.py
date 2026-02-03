"""Tests for ops module implementations."""

import unittest

import numpy as np
import xarray as xr

from aind_ephys_utils.ops import (
    align,
    baseline,
    bin,
    normalize,
    psth,
    reduce,
    restrict,
    smooth,
)
from aind_ephys_utils.ops.align import (
    EphysAlignError,
    _get_event_times,
    _normalize_events,
)
from aind_ephys_utils.standards.validate import EphysValidationError


class OpsTest(unittest.TestCase):
    """Basic tests for ops functions."""

    def test_bin_ragged_trial_unit(self) -> None:
        """Bin ragged spikes over trial/unit."""
        spikes = xr.DataArray(
            data=np.array(
                [[np.array([0.05, 0.15]), np.array([0.02])]],
                dtype=object,
            ),
            dims=("trial", "unit"),
            coords={"trial": [0], "unit": [0, 1]},
        )
        binned = bin(spikes, dt=0.1, window=(0.0, 0.2), output="count")
        self.assertEqual(binned.dims, ("trial", "unit", "time"))
        self.assertEqual(binned.shape, (1, 2, 2))
        np.testing.assert_array_equal(binned.values[0, 0], [1, 1])
        np.testing.assert_array_equal(binned.values[0, 1], [1, 0])

    def test_smooth_gaussian(self) -> None:
        """Smooth a 1D signal with a Gaussian kernel."""
        da = xr.DataArray(
            [0.0, 0.0, 1.0, 0.0, 0.0],
            dims=("time",),
            coords={"time": np.arange(5) * 0.1},
        )
        out = smooth(da, method="gaussian", sigma=0.1)
        self.assertEqual(out.shape, da.shape)
        total = float(out.sum())
        self.assertTrue(0.9 < total < 1.2)

    def test_baseline_subtract(self) -> None:
        """Subtract baseline mean from a time window."""
        da = xr.DataArray(
            [[1.0, 2.0, 3.0, 4.0]],
            dims=("trial", "time"),
            coords={"time": [0.0, 1.0, 2.0, 3.0]},
        )
        out = baseline(da, window=(0.0, 1.0), mode="subtract")
        np.testing.assert_allclose(out.values[0, 0:2], [-0.5, 0.5])

    def test_normalize_zscore(self) -> None:
        """Z-score normalize across trials."""
        da = xr.DataArray(
            [[0.0, 2.0], [2.0, 4.0]],
            dims=("trial", "time"),
            coords={"trial": [0, 1], "time": [0, 1]},
        )
        out = normalize(da, dim="trial", method="zscore")
        np.testing.assert_allclose(out.mean(dim="trial"), 0.0, atol=1e-7)

    def test_psth_mean(self) -> None:
        """Reduce across trials to a mean PSTH."""
        da = xr.DataArray(
            [[1.0, 3.0], [3.0, 5.0]],
            dims=("trial", "time"),
            coords={"trial": [0, 1], "time": [0, 1]},
        )
        out = psth(da, dim="trial", reduce="mean")
        np.testing.assert_allclose(out.values, [2.0, 4.0])

    def test_reduce_pca(self) -> None:
        """Run PCA reduction over trials."""
        da = xr.DataArray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dims=("trial", "unit"),
            coords={"trial": [0, 1, 2], "unit": [0, 1]},
        )
        out = reduce(
            da, method="pca", dim="unit", n_components=1, stack=("trial",)
        )
        self.assertIn("component", out.dims)
        self.assertIn("trial", out.dims)
        self.assertEqual(out.sizes["component"], 1)

    def test_restrict_dense(self) -> None:
        """Restrict a dense DataArray to a time window."""
        da = xr.DataArray(
            np.arange(6, dtype=float),
            dims=("time",),
            coords={"time": np.linspace(0.0, 0.5, 6)},
        )
        out = restrict(da, window=(0.2, 0.4))
        self.assertTrue(out.time.min() >= 0.2)
        self.assertTrue(out.time.max() <= 0.4)

    def test_restrict_ragged(self) -> None:
        """Restrict ragged spikes using searchsorted."""
        data = np.empty((1, 2), dtype=object)
        data[0, 0] = np.array([0.1, 0.3, 0.6])
        data[0, 1] = np.array([0.05, 0.2])
        spikes = xr.DataArray(
            data=data,
            dims=("trial", "unit"),
            coords={"trial": [0], "unit": [0, 1]},
        )
        out = restrict(spikes, window=(0.1, 0.4))
        np.testing.assert_allclose(out.values[0, 0], [0.1, 0.3])
        np.testing.assert_allclose(out.values[0, 1], [0.2])


class NormalizeEventsTest(unittest.TestCase):
    """Tests for _normalize_events helper."""

    def test_dataset_with_t_var(self) -> None:
        """Dataset with 't' variable passes through."""
        events = xr.Dataset(
            {"t": (("trial", "event"), [[1.0, 2.0]])},
            coords={"trial": [0], "event": ["stim", "reward"]},
        )
        result = _normalize_events(events)
        self.assertIn("t", result)
        np.testing.assert_array_equal(result["t"].values, [[1.0, 2.0]])

    def test_dataarray_simple(self) -> None:
        """DataArray of event times is wrapped in Dataset."""
        da = xr.DataArray(
            [[1.0, 2.0]],
            dims=("trial", "event"),
            coords={"trial": [0], "event": ["stim", "reward"]},
        )
        result = _normalize_events(da)
        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("t", result)

    def test_dataarray_with_bound_dim(self) -> None:
        """DataArray with (trial, event, bound) selects 'start'."""
        da = xr.DataArray(
            [[[1.0, 1.5], [2.0, 2.5]]],
            dims=("trial", "event", "bound"),
            coords={
                "trial": [0],
                "event": ["stim", "reward"],
                "bound": ["start", "end"],
            },
        )
        result = _normalize_events(da)
        self.assertIn("t", result)
        # Should select bound="start"
        np.testing.assert_array_equal(result["t"].values, [[1.0, 2.0]])

    def test_dataarray_with_bound_missing_start(self) -> None:
        """DataArray with bound dim but no 'start' raises error."""
        da = xr.DataArray(
            [[[1.0, 1.5]]],
            dims=("trial", "event", "bound"),
            coords={
                "trial": [0],
                "event": ["stim"],
                "bound": ["begin", "end"],
            },
        )
        with self.assertRaises(EphysAlignError):
            _normalize_events(da)

    def test_dataset_missing_t_var(self) -> None:
        """Dataset without 't' variable raises error."""
        events = xr.Dataset(
            {"times": (("trial", "event"), [[1.0, 2.0]])},
            coords={"trial": [0], "event": ["stim", "reward"]},
        )
        with self.assertRaises(EphysAlignError):
            _normalize_events(events)

    def test_invalid_type(self) -> None:
        """Non-xarray input raises TypeError."""
        with self.assertRaises(TypeError):
            _normalize_events({"t": [1.0, 2.0]})


class GetEventTimesTest(unittest.TestCase):
    """Tests for _get_event_times helper."""

    def test_select_event(self) -> None:
        """Select specific event label."""
        events = xr.Dataset(
            {"t": (("trial", "event"), [[1.0, 2.0], [1.5, 2.5]])},
            coords={"trial": [0, 1], "event": ["stim", "reward"]},
        )
        result = _get_event_times(events, to="stim")
        np.testing.assert_array_equal(result.values, [1.0, 1.5])
        self.assertEqual(result.dims, ("trial",))

    def test_missing_event_dim(self) -> None:
        """Missing 'event' dimension raises error."""
        events = xr.Dataset(
            {"t": (("trial",), [1.0, 2.0])},
            coords={"trial": [0, 1]},
        )
        with self.assertRaises(EphysAlignError):
            _get_event_times(events, to="stim")

    def test_missing_event_label(self) -> None:
        """Missing event label raises error."""
        events = xr.Dataset(
            {"t": (("trial", "event"), [[1.0, 2.0]])},
            coords={"trial": [0], "event": ["stim", "reward"]},
        )
        with self.assertRaises(EphysAlignError):
            _get_event_times(events, to="nonexistent")


class AlignContinuousTest(unittest.TestCase):
    """Tests for align with continuous data."""

    def test_continuous_with_trial_dim(self) -> None:
        """Align continuous signal with trial dimension."""
        da = xr.DataArray(
            np.random.randn(2, 10),
            dims=("trial", "time"),
            coords={"trial": [0, 1], "time": np.linspace(-0.5, 0.5, 10)},
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.0], [0.0]])},
            coords={"trial": [0, 1], "event": ["stim"]},
        )
        result = align(da, events=events, to="stim", window=(-0.3, 0.3))
        self.assertIn("trial", result.dims)
        self.assertIn("time", result.dims)
        # Window should be respected
        self.assertTrue(result.time.min() >= -0.3)
        self.assertTrue(result.time.max() <= 0.3)

    def test_continuous_missing_time_dim(self) -> None:
        """Continuous data without time dim raises error."""
        da = xr.DataArray(
            np.arange(10, dtype=float),
            dims=("channel",),
            coords={"channel": range(10)},
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.5]])},
            coords={"trial": [0], "event": ["stim"]},
        )
        with self.assertRaises(EphysValidationError):
            align(da, events=events, to="stim", window=(-0.2, 0.2))


class AlignBinnedTest(unittest.TestCase):
    """Tests for align with binned spikes."""

    def test_binned_spikes(self) -> None:
        """Align binned spike data."""
        da = xr.DataArray(
            np.random.randint(0, 5, (2, 3, 10)),
            dims=("trial", "unit", "time"),
            coords={
                "trial": [0, 1],
                "unit": [0, 1, 2],
                "time": np.linspace(-0.5, 0.5, 10),
            },
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.0], [0.0]])},
            coords={"trial": [0, 1], "event": ["stim"]},
        )
        result = align(da, events=events, to="stim", window=(-0.3, 0.3))
        self.assertIn("trial", result.dims)
        self.assertIn("unit", result.dims)
        self.assertIn("time", result.dims)


class AlignRaggedTest(unittest.TestCase):
    """Tests for align with ragged spike data."""

    def test_ragged_spikes_basic(self) -> None:
        """Align ragged spikes to event."""
        data = np.empty((1, 2), dtype=object)
        data[0, 0] = np.array([0.1, 0.3, 0.5, 0.7])
        data[0, 1] = np.array([0.2, 0.4, 0.6])
        spikes = xr.DataArray(
            data=data,
            dims=("trial", "unit"),
            coords={"trial": [0], "unit": [0, 1]},
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.4]])},
            coords={"trial": [0], "event": ["stim"]},
        )
        result = align(spikes, events=events, to="stim", window=(-0.2, 0.2))
        # Spikes at 0.3, 0.5 are within window (0.2-0.6 absolute)
        # Aligned: 0.3-0.4=-0.1, 0.5-0.4=0.1
        aligned_unit0 = result.values[0, 0]
        self.assertTrue(len(aligned_unit0) > 0)
        self.assertTrue(all(aligned_unit0 >= -0.2))
        self.assertTrue(all(aligned_unit0 <= 0.2))
        self.assertIn("ephys.valid_intervals", result.attrs)

    def test_ragged_spikes_empty_result(self) -> None:
        """Ragged spikes with no spikes in window returns empty arrays."""
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([10.0, 20.0])  # Far from event
        spikes = xr.DataArray(
            data=data,
            dims=("trial", "unit"),
            coords={"trial": [0], "unit": [0]},
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.5]])},
            coords={"trial": [0], "event": ["stim"]},
        )
        result = align(spikes, events=events, to="stim", window=(-0.1, 0.1))
        self.assertEqual(len(result.values[0, 0]), 0)

    def test_ragged_spikes_missing_trial_dim(self) -> None:
        """Ragged spikes without trial dim raises error."""
        data = np.empty((2,), dtype=object)
        data[0] = np.array([0.1, 0.2])
        data[1] = np.array([0.3, 0.4])
        spikes = xr.DataArray(
            data=data,
            dims=("unit",),
            coords={"unit": [0, 1]},
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.5]])},
            coords={"trial": [0], "event": ["stim"]},
        )
        with self.assertRaises(EphysValidationError):
            align(spikes, events=events, to="stim", window=(-0.1, 0.1))

    def test_ragged_spikes_none_entry(self) -> None:
        """Ragged spikes with None entry returns empty array."""
        data = np.empty((1, 2), dtype=object)
        data[0, 0] = None
        data[0, 1] = np.array([0.5])
        spikes = xr.DataArray(
            data=data,
            dims=("trial", "unit"),
            coords={"trial": [0], "unit": [0, 1]},
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.5]])},
            coords={"trial": [0], "event": ["stim"]},
        )
        result = align(spikes, events=events, to="stim", window=(-0.1, 0.1))
        self.assertEqual(len(result.values[0, 0]), 0)
        self.assertEqual(len(result.values[0, 1]), 1)


class AlignWindowValidationTest(unittest.TestCase):
    """Tests for window validation in align."""

    def test_invalid_window_min_equals_max(self) -> None:
        """Window with min == max raises error."""
        da = xr.DataArray(
            np.arange(10, dtype=float),
            dims=("time",),
            coords={"time": np.linspace(0.0, 0.9, 10)},
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.5]])},
            coords={"trial": [0], "event": ["stim"]},
        )
        with self.assertRaises(EphysAlignError):
            align(da, events=events, to="stim", window=(0.2, 0.2))

    def test_invalid_window_min_greater_than_max(self) -> None:
        """Window with min > max raises error."""
        da = xr.DataArray(
            np.arange(10, dtype=float),
            dims=("time",),
            coords={"time": np.linspace(0.0, 0.9, 10)},
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.5]])},
            coords={"trial": [0], "event": ["stim"]},
        )
        with self.assertRaises(EphysAlignError):
            align(da, events=events, to="stim", window=(0.3, 0.1))
