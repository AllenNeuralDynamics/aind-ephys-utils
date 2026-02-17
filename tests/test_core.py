"""Tests for the pure-numpy core module.

No xarray is imported anywhere in this file. This is by design: it
verifies that the core operates independently of xarray.
"""

from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from aind_ephys_utils.core import (
    align,
    align_unit,
    baseline,
    bin_spikes,
    convolve_1d,
    infer_dt,
    make_bin_edges,
    make_kernel,
    normalize,
    psth,
    restrict_dense,
    restrict_ragged,
    smooth,
)


# ---- smooth ----


class InferDtTest(unittest.TestCase):
    def test_uniform(self):
        t = np.array([0.0, 0.1, 0.2, 0.3])
        assert_allclose(infer_dt(t), 0.1)

    def test_non_uniform_raises(self):
        t = np.array([0.0, 0.1, 0.3])
        with self.assertRaises(ValueError):
            infer_dt(t)

    def test_too_few_raises(self):
        with self.assertRaises(ValueError):
            infer_dt(np.array([1.0]))

    def test_decreasing_raises(self):
        with self.assertRaises(ValueError):
            infer_dt(np.array([1.0, 0.0]))


class MakeKernelTest(unittest.TestCase):
    def test_gaussian_sums_to_one(self):
        k = make_kernel("gaussian", 0.01, sigma=0.05)
        assert_allclose(k.sum(), 1.0, atol=1e-10)

    def test_boxcar_sums_to_one(self):
        k = make_kernel("boxcar", 0.01, window=0.05)
        assert_allclose(k.sum(), 1.0, atol=1e-10)

    def test_boxcar_size(self):
        k = make_kernel("boxcar", 0.01, window=0.05)
        self.assertEqual(len(k), 5)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            make_kernel("hamming", 0.01)


class Convolve1dTest(unittest.TestCase):
    def test_identity_kernel(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = convolve_1d(y, np.array([1.0]))
        assert_array_equal(out, y)

    def test_gaussian_preserves_total(self):
        y = np.zeros(100)
        y[50] = 1.0
        k = make_kernel("gaussian", 0.01, sigma=0.03)
        out = convolve_1d(y, k)
        assert_allclose(out.sum(), y.sum(), atol=0.05)

    def test_causal_leading_edge(self):
        """First sample should equal itself (no attenuation)."""
        y = np.ones(20)
        k = make_kernel("boxcar", 0.01, window=0.05)
        out = convolve_1d(y, k, causal=True)
        assert_allclose(out[0], 1.0, atol=1e-10)


class SmoothTest(unittest.TestCase):
    def test_3d_shape_preserved(self):
        data = np.random.default_rng(0).standard_normal((3, 2, 50))
        dt = 0.01
        out = smooth(data, dt, sigma=0.03)
        self.assertEqual(out.shape, data.shape)

    def test_1d(self):
        data = np.zeros(100)
        data[50] = 1.0
        out = smooth(data, 0.01, sigma=0.03)
        assert_allclose(out.sum(), 1.0, atol=0.05)

    def test_gaussian_requires_sigma(self):
        with self.assertRaises(ValueError):
            smooth(np.zeros(10), 0.01, method="gaussian")


# ---- align ----


class AlignUnitTest(unittest.TestCase):
    def test_basic(self):
        # 10 spikes at 0.0, 1.0, ..., 9.0
        spikes = np.arange(10, dtype=float)
        anchors = np.array([2.0, 5.0])
        result = align_unit(spikes, anchors, (-0.5, 0.5))
        # Trial 0 centered at 2.0: spike at 2.0 → 0.0
        assert_allclose(result[0], [0.0])
        # Trial 1 centered at 5.0: spike at 5.0 → 0.0
        assert_allclose(result[1], [0.0])

    def test_empty_window(self):
        spikes = np.array([1.0, 2.0, 3.0])
        anchors = np.array([10.0])
        result = align_unit(spikes, anchors, (-0.1, 0.1))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].size, 0)

    def test_none_input(self):
        result = align_unit(None, np.array([1.0, 2.0]), (-0.5, 0.5))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].size, 0)
        self.assertEqual(result[1].size, 0)


class AlignTest(unittest.TestCase):
    def test_multi_unit(self):
        spike_times = [
            np.array([0.5, 1.5, 2.5]),  # unit 0
            np.array([0.3, 1.3, 2.3]),  # unit 1
        ]
        anchors = np.array([1.0, 2.0])
        result = align(spike_times, anchors, (-0.6, 0.6))
        # result[trial][unit]
        self.assertEqual(len(result), 2)  # 2 trials
        self.assertEqual(len(result[0]), 2)  # 2 units
        # Trial 0, unit 0: spikes at 0.5 and 1.5 within [0.4, 1.6]
        assert_allclose(result[0][0], [-0.5, 0.5])
        # Trial 0, unit 1: spike at 1.3 within [0.4, 1.6] → 0.3
        # (0.3 is below 0.4, so excluded)
        assert_allclose(result[0][1], [0.3])

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            align([np.array([1.0])], np.array([0.0]), (1.0, -1.0))


# ---- bin ----


class MakeBinEdgesTest(unittest.TestCase):
    def test_basic(self):
        edges, centers = make_bin_edges(0.1, (0.0, 0.5))
        assert_allclose(edges, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        assert_allclose(centers, [0.05, 0.15, 0.25, 0.35, 0.45])

    def test_invalid_dt(self):
        with self.assertRaises(ValueError):
            make_bin_edges(-0.1, (0.0, 1.0))


class BinSpikesTest(unittest.TestCase):
    def test_count(self):
        # 1 trial, 1 unit, spikes at 0.05 and 0.15
        spikes = [[np.array([0.05, 0.15])]]
        data, centers = bin_spikes(spikes, 0.1, (0.0, 0.3), output="count")
        self.assertEqual(data.shape, (1, 1, 3))
        assert_allclose(data[0, 0], [1.0, 1.0, 0.0])

    def test_rate(self):
        spikes = [[np.array([0.05, 0.15])]]
        data, centers = bin_spikes(spikes, 0.1, (0.0, 0.3), output="rate")
        assert_allclose(data[0, 0], [10.0, 10.0, 0.0])

    def test_empty_trial(self):
        spikes = [[np.array([])]]
        data, centers = bin_spikes(spikes, 0.1, (0.0, 0.3), output="count")
        assert_allclose(data[0, 0], [0.0, 0.0, 0.0])

    def test_invalid_dt_raises(self):
        with self.assertRaises(ValueError):
            bin_spikes([[np.array([])]], 0.0, (0.0, 1.0))


# ---- restrict ----


class RestrictDenseTest(unittest.TestCase):
    def test_basic(self):
        data = np.arange(10, dtype=float).reshape(1, 1, 10)
        time = np.arange(10, dtype=float)
        rd, rt = restrict_dense(data, time, (2.0, 5.0))
        self.assertEqual(rd.shape, (1, 1, 4))
        assert_array_equal(rt, [2.0, 3.0, 4.0, 5.0])
        assert_array_equal(rd[0, 0], [2.0, 3.0, 4.0, 5.0])


class RestrictRaggedTest(unittest.TestCase):
    def test_basic(self):
        spikes = [[np.array([0.1, 0.5, 0.9, 1.5])]]
        out = restrict_ragged(spikes, (0.2, 1.0))
        assert_allclose(out[0][0], [0.5, 0.9])

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            restrict_ragged([[np.array([])]], (1.0, 0.0))


# ---- baseline ----


class BaselineTest(unittest.TestCase):
    def test_subtract(self):
        # (1 trial, 1 unit, 4 time)
        data = np.array([[[1.0, 2.0, 3.0, 4.0]]])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        out = baseline(data, time, (0.0, 1.0), mode="subtract")
        # baseline mean of [1, 2] = 1.5
        assert_allclose(out[0, 0], [-0.5, 0.5, 1.5, 2.5])

    def test_divide(self):
        data = np.array([[[2.0, 4.0, 6.0, 8.0]]])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        out = baseline(data, time, (0.0, 1.0), mode="divide")
        assert_allclose(out[0, 0], [2.0 / 3.0, 4.0 / 3.0, 2.0, 8.0 / 3.0])

    def test_zscore(self):
        data = np.array([[[1.0, 3.0, 5.0, 7.0]]])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        out = baseline(data, time, (0.0, 1.0), mode="zscore")
        bl = np.array([1.0, 3.0])
        mean, std = bl.mean(), bl.std()
        assert_allclose(out[0, 0, 2], (5.0 - mean) / std)

    def test_zscore_zero_std(self):
        """Constant baseline should produce 0, not NaN."""
        data = np.array([[[5.0, 5.0, 10.0, 15.0]]])
        time = np.array([0.0, 1.0, 2.0, 3.0])
        out = baseline(data, time, (0.0, 1.0), mode="zscore")
        self.assertFalse(np.any(np.isnan(out)))
        assert_allclose(out[0, 0, :2], [0.0, 0.0])

    def test_empty_window_raises(self):
        data = np.array([[[1.0, 2.0]]])
        time = np.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            baseline(data, time, (5.0, 6.0))


# ---- normalize ----


class NormalizeTest(unittest.TestCase):
    def test_zscore_across_trials(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 3, 20))
        out = normalize(data, across="trials", method="zscore")
        # Mean across trials should be ~0
        assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)

    def test_zscore_zero_variance(self):
        data = np.ones((5, 2, 10))
        out = normalize(data, across="trials", method="zscore")
        self.assertFalse(np.any(np.isnan(out)))
        assert_allclose(out, 0.0)

    def test_minmax_range(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 3, 20))
        out = normalize(data, across="trials", method="minmax")
        # Along trial axis, each (unit,time) should be in [0,1]
        self.assertGreaterEqual(out.min(), -1e-10)
        self.assertLessEqual(out.max(), 1.0 + 1e-10)

    def test_robust(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 3, 20))
        out = normalize(data, across="trials", method="robust")
        self.assertEqual(out.shape, data.shape)

    def test_multi_axis(self):
        data = np.ones((5, 3, 10)) * 3.0
        out = normalize(data, across=("trials", "time"), method="zscore")
        assert_allclose(out, 0.0)

    def test_unknown_axis_raises(self):
        with self.assertRaises(ValueError):
            normalize(np.zeros((5, 3, 10)), across="channels")


# ---- psth ----


class PsthTest(unittest.TestCase):
    def test_mean_no_labels(self):
        data = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])  # (2 trials, 2 units, 2 time)
        out = psth(data, method="mean")
        assert_allclose(out, [[3.0, 4.0], [5.0, 6.0]])

    def test_median_no_labels(self):
        data = np.array([
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ])  # (3, 1, 2)
        out = psth(data, method="median")
        assert_allclose(out, [[3.0, 4.0]])

    def test_with_labels(self):
        data = np.array([
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
            [[7.0], [8.0]],
        ])  # (4 trials, 2 units, 1 time)
        labels = np.array(["A", "B", "A", "B"])
        result, unique = psth(data, labels=labels)
        assert_array_equal(unique, ["A", "B"])
        # Group A: trials 0 and 2 → mean
        assert_allclose(result[0], [[(1 + 5) / 2], [(2 + 6) / 2]])
        # Group B: trials 1 and 3 → mean
        assert_allclose(result[1], [[(3 + 7) / 2], [(4 + 8) / 2]])

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            psth(np.zeros((2, 1, 5)), method="mode")


# ---- import isolation ----


class ImportIsolationTest(unittest.TestCase):
    def test_no_xarray_in_core(self):
        """Verify that core/ modules do not import xarray.

        Note: the parent package ``aind_ephys_utils.__init__`` does
        import xarray, so we can't check ``sys.modules`` naively.
        Instead we inspect each core submodule's namespace.
        """
        import importlib
        import types as stdlib_types

        core_modules = [
            "aind_ephys_utils.core.align",
            "aind_ephys_utils.core.baseline",
            "aind_ephys_utils.core.bin",
            "aind_ephys_utils.core.normalize",
            "aind_ephys_utils.core.psth",
            "aind_ephys_utils.core.restrict",
            "aind_ephys_utils.core.smooth",
        ]
        for modname in core_modules:
            mod = importlib.import_module(modname)
            for key, val in mod.__dict__.items():
                if isinstance(val, stdlib_types.ModuleType):
                    self.assertFalse(
                        "xarray" in val.__name__,
                        f"{modname} has xarray in its namespace: {key}",
                    )


if __name__ == "__main__":
    unittest.main()
