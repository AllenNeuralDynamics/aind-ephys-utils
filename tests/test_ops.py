"""Tests for ops module implementations."""

import unittest
import warnings

import numpy as np
import xarray as xr

from aind_ephys_utils.ops import (
    align,
    baseline,
    bin,
    normalize,
    pseudopop,
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

    def test_bin_accepts_bin_size_alias(self) -> None:
        """bin_size should be interchangeable with dt."""
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([0.05, 0.15])
        spikes = xr.DataArray(data, dims=("trial", "unit"))
        via_dt = bin(spikes, dt=0.1, window=(0.0, 0.2), output="count")
        via_alias = bin(
            spikes, bin_size=0.1, window=(0.0, 0.2), output="count"
        )
        xr.testing.assert_equal(via_alias, via_dt)

    def test_bin_rejects_dt_and_bin_size(self) -> None:
        """Only one bin-size alias may be supplied."""
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([0.05])
        spikes = xr.DataArray(data, dims=("trial", "unit"))
        with self.assertRaisesRegex(ValueError, "only one"):
            bin(spikes, dt=0.1, bin_size=0.1, window=(0.0, 0.2))

    def test_bin_auto(self) -> None:
        """Automatic bin selection should produce a valid dense result."""
        rng = np.random.default_rng(0)
        data = np.empty((20, 1), dtype=object)
        for trial in range(data.shape[0]):
            data[trial, 0] = np.sort(rng.uniform(0.0, 1.0, size=20))
        spikes = xr.DataArray(data, dims=("trial", "unit"))
        out = bin(spikes, dt="auto", window=(0.0, 1.0))
        self.assertEqual(out.dims, ("trial", "unit", "time"))
        self.assertGreater(out.sizes["time"], 1)

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

    def test_smooth_boxcar_causal_edge_normalization(self) -> None:
        """Causal boxcar should not attenuate the leading sample."""
        da = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0],
            dims=("time",),
            coords={"time": np.arange(4) * 0.1},
        )
        out = smooth(da, method="boxcar", window=0.3)
        # With dt=0.1 and window=0.3, kernel length is 3. Leading samples
        # should be means of available history: [1, (1+2)/2, (1+2+3)/3, ...]
        np.testing.assert_allclose(
            out.values[:3], np.array([1.0, 1.5, 2.0]), atol=1e-7
        )

    def test_baseline_subtract(self) -> None:
        """Subtract baseline mean from a time window."""
        da = xr.DataArray(
            [[1.0, 2.0, 3.0, 4.0]],
            dims=("trial", "time"),
            coords={"time": [0.0, 1.0, 2.0, 3.0]},
        )
        out = baseline(da, window=(0.0, 1.0), mode="subtract")
        np.testing.assert_allclose(out.values[0, 0:2], [-0.5, 0.5])

    def test_baseline_zscore_zero_std_returns_zero(self) -> None:
        """Zero-variance baseline slices should produce 0, not NaN/inf."""
        da = xr.DataArray(
            [[1.0, 1.0, 1.0, 1.0]],
            dims=("trial", "time"),
            coords={"time": [0.0, 1.0, 2.0, 3.0]},
        )
        out = baseline(da, window=(0.0, 1.0), mode="zscore")
        np.testing.assert_allclose(out.values, np.zeros_like(out.values))

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
        out = psth(da, dim="trial", method="mean")
        np.testing.assert_allclose(out.values, [2.0, 4.0])

    def test_psth_bins_and_smooths_ragged_spikes(self) -> None:
        """Ragged PSTH should compose bin, reduction, and boxcar smoothing."""
        spikes = xr.DataArray(
            np.array(
                [
                    [np.array([0.05, 0.15])],
                    [np.array([0.05])],
                ],
                dtype=object,
            ),
            dims=("trial", "unit"),
        )
        out = psth(
            spikes,
            bin_size=0.1,
            smooth_window=0.2,
            window=(0.0, 0.3),
        )
        expected = smooth(
            bin(spikes, dt=0.1, window=(0.0, 0.3)).mean(dim="trial"),
            method="boxcar",
            window=0.2,
        )
        xr.testing.assert_allclose(out, expected)

    def test_psth_ragged_requires_bin_size(self) -> None:
        """Ragged spikes need an explicit or automatic bin size."""
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([0.05])
        spikes = xr.DataArray(data, dims=("trial", "unit"))
        with self.assertRaisesRegex(ValueError, "requires bin_size"):
            psth(spikes)

    def test_psth_accepts_auto_bin_size(self) -> None:
        """Ragged PSTH should support automatic bin-width selection."""
        rng = np.random.default_rng(1)
        data = np.empty((20, 1), dtype=object)
        for trial in range(data.shape[0]):
            data[trial, 0] = np.sort(rng.uniform(0.0, 1.0, size=20))
        spikes = xr.DataArray(data, dims=("trial", "unit"))
        out = psth(spikes, bin_size="auto", window=(0.0, 1.0))
        self.assertEqual(out.dims, ("unit", "time"))
        self.assertGreater(out.sizes["time"], 1)

    def test_psth_accessor_bins_and_smooths(self) -> None:
        """The accessor should expose integrated binning and smoothing."""
        data = np.empty((2, 1), dtype=object)
        data[0, 0] = np.array([0.05])
        data[1, 0] = np.array([0.15])
        spikes = xr.DataArray(data, dims=("trial", "unit"))
        via_op = psth(
            spikes,
            bin_size=0.1,
            smooth_window=0.2,
            window=(0.0, 0.3),
        )
        via_accessor = spikes.ephys.psth(
            bin_size=0.1,
            smooth_window=0.2,
            window=(0.0, 0.3),
        )
        xr.testing.assert_allclose(via_accessor, via_op)

    def test_psth_group_by_mean(self) -> None:
        """Group by trial coord before averaging."""
        da = xr.DataArray(
            [[1.0, 3.0], [3.0, 5.0], [10.0, 14.0], [14.0, 18.0]],
            dims=("trial", "time"),
            coords={
                "trial": [0, 1, 2, 3],
                "time": [0, 1],
                "condition": ("trial", ["a", "a", "b", "b"]),
            },
        )
        out = psth(da, dim="trial", method="mean", group_by="condition")
        self.assertEqual(out.dims, ("condition", "time"))
        np.testing.assert_array_equal(out["condition"].values, ["a", "b"])
        np.testing.assert_allclose(out.sel(condition="a").values, [2.0, 4.0])
        np.testing.assert_allclose(out.sel(condition="b").values, [12.0, 16.0])

    def test_psth_group_by_median(self) -> None:
        """Group by trial coord before median reduction."""
        da = xr.DataArray(
            [[1.0, 7.0], [3.0, 5.0], [10.0, 30.0], [14.0, 18.0]],
            dims=("trial", "time"),
            coords={
                "trial": [0, 1, 2, 3],
                "time": [0, 1],
                "condition": ("trial", ["a", "a", "b", "b"]),
            },
        )
        out = psth(da, dim="trial", method="median", group_by="condition")
        np.testing.assert_allclose(out.sel(condition="a").values, [2.0, 6.0])
        np.testing.assert_allclose(out.sel(condition="b").values, [12.0, 24.0])

    def test_psth_group_by_multiple_coords(self) -> None:
        """Group by multiple trial coords and return separate group dims."""
        da = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]],
            dims=("trial", "time"),
            coords={
                "trial": [0, 1, 2, 3],
                "time": [0, 1],
                "f1": ("trial", ["a", "a", "b", "b"]),
                "f2": ("trial", ["x", "y", "x", "y"]),
            },
        )
        out = psth(da, dim="trial", method="mean", group_by=["f1", "f2"])
        self.assertEqual(set(out.dims), {"f1", "f2", "time"})
        np.testing.assert_allclose(out.sel(f1="a", f2="x").values, [1.0, 2.0])
        np.testing.assert_allclose(
            out.sel(f1="b", f2="y").values, [30.0, 40.0]
        )

    def test_psth_group_by_disallows_keep_trials(self) -> None:
        """group_by and keep_trials cannot be used together."""
        da = xr.DataArray(
            [[1.0, 3.0], [3.0, 5.0]],
            dims=("trial", "time"),
            coords={
                "trial": [0, 1],
                "time": [0, 1],
                "condition": ("trial", ["a", "a"]),
            },
        )
        with self.assertRaises(ValueError):
            _ = psth(
                da,
                dim="trial",
                method="mean",
                group_by="condition",
                keep_trials=True,
            )

    def test_psth_invalid_method_none(self) -> None:
        """method=None should raise a clear error."""
        da = xr.DataArray(
            [[1.0, 3.0], [3.0, 5.0]],
            dims=("trial", "time"),
            coords={"trial": [0, 1], "time": [0, 1]},
        )
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            _ = psth(da, dim="trial", method=None)

    def test_psth_invalid_method_type(self) -> None:
        """Non-string method should raise a clear error."""
        da = xr.DataArray(
            [[1.0, 3.0], [3.0, 5.0]],
            dims=("trial", "time"),
            coords={"trial": [0, 1], "time": [0, 1]},
        )
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            _ = psth(da, dim="trial", method=123)  # type: ignore[arg-type]

    def test_pseudopop_grouped_concat(self) -> None:
        """Concatenate grouped PSTHs across sessions on unit axis."""
        da0 = xr.DataArray(
            np.array(
                [
                    [[1.0, 3.0], [2.0, 4.0]],
                    [[3.0, 5.0], [4.0, 6.0]],
                ]
            ),
            dims=("trial", "unit", "time"),
            coords={
                "trial": [0, 1],
                "unit": [0, 1],
                "time": [0, 1],
                "choice": ("trial", ["a", "a"]),
            },
        )
        da1 = xr.DataArray(
            np.array(
                [
                    [[10.0, 12.0]],
                    [[14.0, 16.0]],
                ]
            ),
            dims=("trial", "unit", "time"),
            coords={
                "trial": [0, 1],
                "unit": [0],
                "time": [0, 1],
                "choice": ("trial", ["a", "a"]),
            },
        )
        out = pseudopop(
            [da0, da1], group_by="choice", session_ids=["s0", "s1"]
        )
        self.assertEqual(out.dims, ("choice", "unit", "time"))
        self.assertEqual(out.sizes["unit"], 3)
        np.testing.assert_array_equal(
            out["session"].values, np.array(["s0", "s0", "s1"], dtype=object)
        )
        np.testing.assert_array_equal(out["unit"].values, np.array([0, 1, 2]))
        np.testing.assert_allclose(
            out.sel(choice="a").isel(unit=0).values, np.array([2.0, 4.0])
        )
        np.testing.assert_allclose(
            out.sel(choice="a").isel(unit=2).values, np.array([12.0, 14.0])
        )

    def test_pseudopop_default_session_ids(self) -> None:
        """Default session IDs should be s0, s1, ..."""
        da = xr.DataArray(
            np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
            dims=("trial", "unit", "time"),
            coords={
                "trial": [0, 1],
                "unit": [0],
                "time": [0, 1],
                "choice": ("trial", ["a", "a"]),
            },
        )
        out = pseudopop([da], group_by="choice")
        np.testing.assert_array_equal(
            out["session"].values, np.array(["s0"], dtype=object)
        )

    def test_pseudopop_session_ids_length_mismatch(self) -> None:
        """session_ids length must match number of sessions."""
        da = xr.DataArray(
            np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
            dims=("trial", "unit", "time"),
            coords={
                "trial": [0, 1],
                "unit": [0],
                "time": [0, 1],
                "choice": ("trial", ["a", "a"]),
            },
        )
        with self.assertRaises(ValueError):
            _ = pseudopop([da], group_by="choice", session_ids=["s0", "s1"])

    def test_pseudopop_concat_join_explicit(self) -> None:
        """Concatenation should set join explicitly to avoid FutureWarning."""
        da0 = xr.DataArray(
            np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
            dims=("trial", "unit", "time"),
            coords={
                "trial": [0, 1],
                "unit": [0],
                "time": [0.0, 0.1],
                "choice": ("trial", ["a", "a"]),
            },
        )
        da1 = xr.DataArray(
            np.array([[[10.0, 20.0]], [[30.0, 40.0]]]),
            dims=("trial", "unit", "time"),
            coords={
                "trial": [0, 1],
                "unit": [0],
                "time": [0.0, 0.2],
                "choice": ("trial", ["a", "a"]),
            },
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = pseudopop([da0, da1], group_by="choice")
        future_warnings = [
            w for w in caught if issubclass(w.category, FutureWarning)
        ]
        self.assertEqual(len(future_warnings), 0)
        np.testing.assert_array_equal(
            out["time"].values, np.array([0.0, 0.1, 0.2])
        )

    def test_pseudopop_group_by_multiple_coords(self) -> None:
        """Pseudopop should support multi-label trial grouping."""
        da0 = xr.DataArray(
            np.array(
                [
                    [[1.0, 2.0]],
                    [[3.0, 4.0]],
                    [[10.0, 20.0]],
                    [[30.0, 40.0]],
                ]
            ),
            dims=("trial", "unit", "time"),
            coords={
                "trial": [0, 1, 2, 3],
                "unit": [0],
                "time": [0.0, 0.1],
                "f1": ("trial", ["a", "a", "b", "b"]),
                "f2": ("trial", ["x", "y", "x", "y"]),
            },
        )
        out = pseudopop([da0], group_by=["f1", "f2"])
        self.assertEqual(set(out.dims), {"f1", "f2", "unit", "time"})
        np.testing.assert_allclose(
            out.sel(f1="a", f2="x").isel(unit=0).values, np.array([1.0, 2.0])
        )

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

    def test_normalize_numpy_input(self) -> None:
        """normalize should accept NumPy input and return NumPy by default."""
        arr = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=float)
        out = normalize(arr, dim="trial", dims=("trial", "time"))
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-7)

    def test_smooth_numpy_input(self) -> None:
        """smooth should accept dense NumPy input with explicit dims/coords."""
        arr = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
        out = smooth(
            arr,
            method="gaussian",
            sigma=1.0,
            dims=("time",),
            coords={"time": np.arange(5, dtype=float)},
        )
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, arr.shape)

    def test_baseline_numpy_input(self) -> None:
        """baseline should accept NumPy input and return NumPy by default."""
        arr = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
        out = baseline(
            arr,
            window=(0.0, 1.0),
            mode="subtract",
            dims=("trial", "time"),
            coords={"time": np.array([0.0, 1.0, 2.0, 3.0], dtype=float)},
        )
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_allclose(out[0, :2], [-0.5, 0.5])

    def test_restrict_numpy_input(self) -> None:
        """restrict should accept NumPy input with a time dimension."""
        arr = np.arange(10, dtype=float)
        out = restrict(
            arr,
            window=(2.0, 5.0),
            dims=("time",),
            coords={"time": np.arange(10, dtype=float)},
        )
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.array([2.0, 3.0, 4.0, 5.0]))

    def test_psth_numpy_input(self) -> None:
        """psth should accept NumPy input and reduce over a named dim."""
        arr = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=float)
        out = psth(arr, dim="trial", method="mean", dims=("trial", "time"))
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_allclose(out, np.array([2.0, 4.0]))

    def test_bin_ragged_session_list_input(self) -> None:
        """bin should accept session ragged list input."""
        spikes = [
            np.array([0.05, 0.15], dtype=float),
            np.array([0.02], dtype=float),
        ]
        out = bin(spikes, dt=0.1, window=(0.0, 0.2), output="count")
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1, 2, 2))
        np.testing.assert_array_equal(out[0, 0], [1, 1])
        np.testing.assert_array_equal(out[0, 1], [1, 0])

    def test_align_ragged_session_list_input(self) -> None:
        """align should accept session ragged list and return trial ragged list."""
        spikes = [np.array([0.1, 0.3, 0.5, 0.7], dtype=float)]
        out = align(
            spikes,
            events=np.array([0.4, 0.6]),
            to="stim",
            window=(-0.2, 0.2),
        )
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 1)
        np.testing.assert_allclose(out[0][0], np.array([-0.1, 0.1]))
        np.testing.assert_allclose(out[1][0], np.array([-0.1, 0.1]))

    def test_numpy_input_return_type_xarray(self) -> None:
        """NumPy input should support explicit xarray output."""
        arr = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=float)
        out = psth(
            arr,
            dim="trial",
            method="mean",
            dims=("trial", "time"),
            return_type="xarray",
        )
        self.assertIsInstance(out, xr.DataArray)


class ReduceMethodsTest(unittest.TestCase):
    """Coverage for all dimensionality-reduction methods."""

    @staticmethod
    def _make_reduce_data() -> xr.DataArray:
        """Construct simple separable trial x unit x time data."""
        n_trial, n_unit, n_time = 6, 3, 5
        choice = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        base = np.zeros((n_trial, n_unit, n_time), dtype=float)
        t = np.linspace(-0.2, 0.2, n_time)
        base[:, 0, :] = choice[:, None] * 2.0 + t[None, :]
        base[:, 1, :] = (1 - choice)[:, None] * 1.5 - t[None, :]
        base[:, 2, :] = 0.25 * t[None, :] + 0.1
        return xr.DataArray(
            base,
            dims=("trial", "unit", "time"),
            coords={
                "trial": np.arange(n_trial),
                "unit": np.arange(n_unit),
                "time": t,
                "choice": ("trial", choice),
            },
        )

    def test_reduce_pca(self) -> None:
        """PCA returns projection and weights with expected dimensions."""
        da = self._make_reduce_data()
        out = reduce(
            da,
            method="pca",
            dim="unit",
            n_components=2,
            stack=("trial", "time"),
        )
        self.assertIn("projections", out)
        self.assertIn("weights", out)
        self.assertEqual(
            out["projections"].dims, ("component", "trial", "time")
        )
        self.assertEqual(out["weights"].dims, ("component", "unit"))
        self.assertEqual(out["projections"].sizes["component"], 2)

    def test_reduce_dpca(self) -> None:
        """dPCA returns marginalized projections and per-marginal weights."""
        da = self._make_reduce_data()
        out = reduce(
            da,
            method="dpca",
            dim="unit",
            n_components=2,
            labels="choice",
        )
        self.assertIn("projections", out)
        self.assertIn("weights", out)
        self.assertIn("marginal", out["projections"].dims)
        self.assertIn("component", out["projections"].dims)
        self.assertIn("choice", out["projections"].dims)
        self.assertIn("time", out["projections"].dims)
        self.assertEqual(
            out["weights"].dims, ("marginal", "component", "unit")
        )

    def test_reduce_tdr_from_trial_data(self) -> None:
        """TDR on trial data should condition-average and return expected dims."""
        da = self._make_reduce_data()
        out = reduce(
            da,
            method="tdr",
            dim="unit",
            n_components=2,
            labels="choice",
        )
        self.assertEqual(
            out["projections"].dims, ("condition", "component", "time")
        )
        self.assertEqual(out["weights"].dims, ("condition", "unit"))
        self.assertEqual(out["projections"].sizes["condition"], 2)
        self.assertEqual(out["weights"].sizes["condition"], 2)

    def test_reduce_tdr_from_condition_data(self) -> None:
        """TDR should accept condition x unit x time input directly."""
        da = self._make_reduce_data().groupby("choice").mean("trial")
        da = da.rename({"choice": "condition"})
        out = reduce(
            da,
            method="tdr",
            dim="unit",
            n_components=2,
        )
        self.assertEqual(
            out["projections"].dims, ("condition", "component", "time")
        )
        self.assertEqual(out["weights"].dims, ("condition", "unit"))

    def test_reduce_tdr_trial_data_requires_labels(self) -> None:
        """TDR on trial data should require labels for conditioning."""
        da = self._make_reduce_data()
        with self.assertRaisesRegex(ValueError, "labels are required"):
            _ = reduce(da, method="tdr", dim="unit", n_components=2)

    def test_reduce_dpca_missing_condition_combinations_raises(self) -> None:
        """dPCA should fail when label combinations are incomplete."""
        da = self._make_reduce_data()
        da = da.assign_coords(
            block=("trial", np.array([0, 0, 1, 1, 1, 1], dtype=int))
        )
        with self.assertRaisesRegex(
            ValueError, "complete condition combinations"
        ):
            _ = reduce(
                da,
                method="dpca",
                dim="unit",
                n_components=2,
                labels=["choice", "block"],
            )

    def test_reduce_gpfa(self) -> None:
        """GPFA returns projections and weights with expected dimensions."""
        da = self._make_reduce_data()
        da = da - da.min() + 1e-6
        out = reduce(
            da,
            method="gpfa",
            dim="unit",
            n_components=2,
        )
        self.assertIn("projections", out)
        self.assertIn("weights", out)
        self.assertEqual(
            out["projections"].dims, ("component", "trial", "time")
        )
        self.assertEqual(out["weights"].dims, ("component", "unit"))
        self.assertEqual(out["projections"].sizes["component"], 2)

    def test_reduce_gpfa_options(self) -> None:
        """GPFA options should be configurable via gpfa_options dict."""
        da = self._make_reduce_data()
        da = da - da.min() + 1e-6
        out = reduce(
            da,
            method="gpfa",
            dim="unit",
            n_components=2,
            gpfa_options={"max_iters": 50, "freq_ll": 2},
        )
        self.assertEqual(
            out["projections"].dims, ("component", "trial", "time")
        )
        self.assertEqual(out["weights"].dims, ("component", "unit"))

    def test_reduce_gpfa_options_unknown_key(self) -> None:
        """Unknown GPFA options should raise a clear error."""
        da = self._make_reduce_data()
        da = da - da.min() + 1e-6
        with self.assertRaises(ValueError):
            _ = reduce(
                da,
                method="gpfa",
                dim="unit",
                n_components=2,
                gpfa_options={"not_a_real_option": 1},
            )

    def test_reduce_gpfa_fast_mode(self) -> None:
        """Fast mode should run with a larger GP update interval."""
        da = self._make_reduce_data()
        da = da - da.min() + 1e-6
        out = reduce(
            da,
            method="gpfa",
            dim="unit",
            n_components=2,
            gpfa_options={"fast_mode": True, "gp_param_update_every": 3},
        )
        self.assertEqual(
            out["projections"].dims, ("component", "trial", "time")
        )
        self.assertEqual(out["weights"].dims, ("component", "unit"))

    def test_reduce_gpfa_options_invalid_update_interval(self) -> None:
        """GP parameter update interval must be >= 1."""
        da = self._make_reduce_data()
        da = da - da.min() + 1e-6
        with self.assertRaises(ValueError):
            _ = reduce(
                da,
                method="gpfa",
                dim="unit",
                n_components=2,
                gpfa_options={"gp_param_update_every": 0},
            )

    def test_reduce_gpfa_random_state_reproducible(self) -> None:
        """Fixed random_state should make GPFA outputs deterministic."""
        da = self._make_reduce_data()
        da = da - da.min() + 1e-6
        opts = {"max_iters": 30, "freq_ll": 2, "random_state": 123}
        out1 = reduce(
            da,
            method="gpfa",
            dim="unit",
            n_components=2,
            gpfa_options=opts,
        )
        out2 = reduce(
            da,
            method="gpfa",
            dim="unit",
            n_components=2,
            gpfa_options=opts,
        )
        np.testing.assert_allclose(
            out1["projections"].values, out2["projections"].values
        )
        np.testing.assert_allclose(
            out1["weights"].values, out2["weights"].values
        )

    def test_reduce_gpfa_options_invalid_random_state(self) -> None:
        """random_state must be int or NumPy RNG object."""
        da = self._make_reduce_data()
        da = da - da.min() + 1e-6
        with self.assertRaises(ValueError):
            _ = reduce(
                da,
                method="gpfa",
                dim="unit",
                n_components=2,
                gpfa_options={"random_state": "bad-seed"},
            )

    def test_reduce_coding_direction(self) -> None:
        """Coding direction returns a single discriminant component."""
        da = self._make_reduce_data()
        out = reduce(
            da,
            method="coding_direction",
            dim="unit",
            labels="choice",
            stack=("trial", "time"),
        )
        self.assertEqual(
            out["projections"].dims, ("component", "trial", "time")
        )
        self.assertEqual(out["weights"].dims, ("component", "unit"))
        self.assertEqual(out["projections"].sizes["component"], 1)

    def test_reduce_logistic(self) -> None:
        """Logistic reduction returns one component for binary labels."""
        da = self._make_reduce_data()
        out = reduce(
            da,
            method="logistic",
            dim="unit",
            labels="choice",
            stack=("trial", "time"),
        )
        self.assertEqual(
            out["projections"].dims, ("component", "trial", "time")
        )
        self.assertEqual(out["weights"].dims, ("component", "unit"))
        self.assertEqual(out["projections"].sizes["component"], 1)

    def test_reduce_logistic_labels_dataarray(self) -> None:
        """Logistic reduction should accept labels passed as DataArray."""
        da = self._make_reduce_data()
        out = reduce(
            da,
            method="logistic",
            dim="unit",
            labels=da.coords["choice"],
            stack=("trial", "time"),
        )
        self.assertEqual(
            out["projections"].dims, ("component", "trial", "time")
        )
        self.assertEqual(out["weights"].dims, ("component", "unit"))

    def test_reduce_lda(self) -> None:
        """LDA reduction returns one component for binary labels."""
        da = self._make_reduce_data()
        out = reduce(
            da,
            method="lda",
            dim="unit",
            labels="choice",
            stack=("trial", "time"),
        )
        self.assertEqual(
            out["projections"].dims, ("component", "trial", "time")
        )
        self.assertEqual(out["weights"].dims, ("component", "unit"))
        self.assertEqual(out["projections"].sizes["component"], 1)

    def test_reduce_rrr(self) -> None:
        """RRR returns requested number of components for multivariate targets."""
        da = self._make_reduce_data()
        target_data = np.stack(
            [
                da.values[:, 0, :] + da.values[:, 1, :],
                da.values[:, 2, :] - da.values[:, 1, :],
            ],
            axis=-1,
        )
        targets = xr.DataArray(
            target_data,
            dims=("trial", "time", "target"),
            coords={
                "trial": da.coords["trial"],
                "time": da.coords["time"],
                "target": ["y0", "y1"],
            },
        )
        out = reduce(
            da,
            method="rrr",
            dim="unit",
            stack=("trial", "time"),
            targets=targets,
            rank=2,
        )
        self.assertEqual(
            out["projections"].dims, ("component", "trial", "time")
        )
        self.assertEqual(out["weights"].dims, ("component", "unit"))
        self.assertEqual(out["projections"].sizes["component"], 2)


class NormalizeEventsTest(unittest.TestCase):
    """Tests for _normalize_events helper."""

    def test_dataset_input_rejected(self) -> None:
        """Dataset input is no longer supported for _normalize_events."""
        events = xr.Dataset(
            {"t": (("trial", "event"), [[1.0, 2.0]])},
            coords={"trial": [0], "event": ["stim", "reward"]},
        )
        with self.assertRaises(TypeError):
            _normalize_events(events)  # type: ignore[arg-type]

    def test_dataarray_simple(self) -> None:
        """DataArray of event times passes through unchanged."""
        da = xr.DataArray(
            [[1.0, 2.0]],
            dims=("trial", "event"),
            coords={"trial": [0], "event": ["stim", "reward"]},
        )
        result = _normalize_events(da)
        self.assertIsInstance(result, xr.DataArray)
        np.testing.assert_array_equal(result.values, [[1.0, 2.0]])

    def test_dataarray_with_bound_dim(self) -> None:
        """DataArray with (trial, event, bound) selects start times."""
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
        np.testing.assert_array_equal(result.values, [[1.0, 2.0]])

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

    def test_invalid_type(self) -> None:
        """Non-xarray input raises TypeError."""
        with self.assertRaises(TypeError):
            _normalize_events({"t": [1.0, 2.0]})

    def test_array_like_event_times(self) -> None:
        """1D array-like event times are coerced to trial/event DataArray."""
        result = _normalize_events([1.0, 2.0], to="stim")
        np.testing.assert_array_equal(result.values, [[1.0], [2.0]])
        np.testing.assert_array_equal(result["event"].values, ["stim"])

    def test_array_like_requires_1d(self) -> None:
        """2D array-like input is ambiguous and rejected."""
        with self.assertRaises(EphysAlignError):
            _normalize_events(np.array([[1.0, 2.0]]), to="stim")


class GetEventTimesTest(unittest.TestCase):
    """Tests for _get_event_times helper."""

    def test_select_event(self) -> None:
        """Select specific event label."""
        events = xr.DataArray(
            [[1.0, 2.0], [1.5, 2.5]],
            dims=("trial", "event"),
            coords={"trial": [0, 1], "event": ["stim", "reward"]},
        )
        result = _get_event_times(events, to="stim")
        np.testing.assert_array_equal(result.values, [1.0, 1.5])
        self.assertEqual(result.dims, ("trial",))

    def test_missing_event_dim(self) -> None:
        """Missing 'event' dimension raises error."""
        events = xr.DataArray(
            [1.0, 2.0],
            dims=("trial",),
            coords={"trial": [0, 1]},
        )
        with self.assertRaises(EphysAlignError):
            _get_event_times(events, to="stim")

    def test_missing_event_label(self) -> None:
        """Missing event label raises error."""
        events = xr.DataArray(
            [[1.0, 2.0]],
            dims=("trial", "event"),
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
        result = align(da, events=events["t"], to="stim", window=(-0.3, 0.3))
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
            align(da, events=events["t"], to="stim", window=(-0.2, 0.2))

    def test_continuous_no_trial_array_like_events(self) -> None:
        """No-trial continuous alignment accepts array-like event times."""
        da = xr.DataArray(
            np.arange(10, dtype=float),
            dims=("time",),
            coords={"time": np.linspace(0.0, 0.9, 10)},
        )
        result = align(da, events=[0.5], to="stim", window=(-0.2, 0.2))
        np.testing.assert_allclose(
            result["time"].values, [-0.2, -0.1, 0.0, 0.1]
        )

    def test_continuous_no_trial_array_like_events_without_to(self) -> None:
        """Array-like event times should not require the to argument."""
        da = xr.DataArray(
            np.arange(10, dtype=float),
            dims=("time",),
            coords={"time": np.linspace(0.0, 0.9, 10)},
        )
        result = align(da, events=[0.5], window=(-0.2, 0.2))
        np.testing.assert_allclose(
            result["time"].values, [-0.2, -0.1, 0.0, 0.1]
        )

    def test_xarray_events_require_to(self) -> None:
        """xarray events should still require an explicit label via to."""
        da = xr.DataArray(
            np.arange(10, dtype=float),
            dims=("time",),
            coords={"time": np.linspace(0.0, 0.9, 10)},
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.5]])},
            coords={"trial": [0], "event": ["stim"]},
        )
        with self.assertRaisesRegex(EphysAlignError, "to is required"):
            _ = align(da, events=events["t"], window=(-0.2, 0.2))


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
        result = align(da, events=events["t"], to="stim", window=(-0.3, 0.3))
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
        result = align(
            spikes, events=events["t"], to="stim", window=(-0.2, 0.2)
        )
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
        result = align(
            spikes, events=events["t"], to="stim", window=(-0.1, 0.1)
        )
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
            align(spikes, events=events["t"], to="stim", window=(-0.1, 0.1))

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
        result = align(
            spikes, events=events["t"], to="stim", window=(-0.1, 0.1)
        )
        self.assertEqual(len(result.values[0, 0]), 0)
        self.assertEqual(len(result.values[0, 1]), 1)

    def test_ragged_spikes_array_like_events(self) -> None:
        """Ragged spikes alignment accepts array-like event times."""
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([0.1, 0.3, 0.5, 0.7])
        spikes = xr.DataArray(
            data=data,
            dims=("trial", "unit"),
            coords={"trial": [0], "unit": [0]},
        )
        result = align(
            spikes, events=np.array([0.4]), to="stim", window=(-0.2, 0.2)
        )
        np.testing.assert_allclose(result.values[0, 0], np.array([-0.1, 0.1]))

    def test_ragged_spikes_multi_trial_input_raises(self) -> None:
        """Ragged align should reject already-trialized ragged input."""
        data = np.empty((2, 1), dtype=object)
        data[0, 0] = np.array([0.1, 0.2])
        data[1, 0] = np.array([0.3, 0.4])
        spikes = xr.DataArray(
            data=data,
            dims=("trial", "unit"),
            coords={"trial": [0, 1], "unit": [0]},
        )
        events = xr.Dataset(
            {"t": (("trial", "event"), [[0.2], [0.3]])},
            coords={"trial": [0, 1], "event": ["stim"]},
        )
        with self.assertRaisesRegex(EphysAlignError, "single 'trial' entry"):
            _ = align(
                spikes, events=events["t"], to="stim", window=(-0.1, 0.1)
            )


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
            align(da, events=events["t"], to="stim", window=(0.2, 0.2))

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
            align(da, events=events["t"], to="stim", window=(0.3, 0.1))
