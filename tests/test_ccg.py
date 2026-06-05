"""Tests for the cross-correlogram (ccg) module."""

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from aind_ephys_utils.adapters import from_dataframe
from aind_ephys_utils.metrics.ccg import (
    ccg,
    ccg_allpairs_sparse,
    ccg_between_sets_sparse,
    ccg_trial_paired,
    clip_spikes_to_trials,
    rescale_ccgs,
    smooth_ccgs,
)

try:
    import numba  # noqa: F401

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

needs_numba = unittest.skipUnless(
    HAS_NUMBA, "requires the numba optional extra"
)


def _ragged_da(
    data: np.ndarray,
    *,
    units: list[str] | None = None,
    window: tuple[float, float] = (0.0, 1.0),
) -> xr.DataArray:
    """Build a ragged spikes DataArray for wrapper tests."""
    if units is None:
        units = [f"u{i}" for i in range(data.shape[0])]
    da = xr.DataArray(
        data,
        dims=("unit", "trial"),
        coords={"unit": units, "trial": np.arange(data.shape[1])},
    )
    da.attrs["ephys.valid_intervals"] = [window]
    return da


class CCGTest(unittest.TestCase):
    """Tests for sparse CCG computation and post-processing."""

    @needs_numba
    def test_between_sets_lag_convention(self) -> None:
        """C[i, j] histograms t_j - t_i, so a +delta lag is positive."""
        delta = 0.005
        a = np.arange(0.1, 1.0, 0.1)
        b = a + delta
        lags, C = ccg_between_sets_sparse(
            [a],
            [b],
            bin_size=0.001,
            max_lag=0.05,
            observation_window=(0.0, 1.0),
        )
        self.assertEqual(C.shape, (1, 1, lags.size))
        peak_lag = lags[np.argmax(C[0, 0])]
        # set2 fires delta AFTER set1 -> peak at positive lag = +delta.
        self.assertLess(abs(peak_lag - delta), 0.001 + 1e-9)
        # Every one of the 9 source spikes contributes exactly one pair.
        self.assertEqual(int(C[0, 0].max()), a.size)

    @needs_numba
    def test_allpairs_symmetry(self) -> None:
        """C[i, j](tau) must equal C[j, i](-tau)."""
        rng = np.random.default_rng(0)
        u0 = np.sort(rng.uniform(0, 10, size=200))
        u1 = np.sort(rng.uniform(0, 10, size=150))
        lags, C = ccg_allpairs_sparse(
            [u0, u1],
            bin_size=0.001,
            max_lag=0.02,
            observation_window=(0.0, 10.0),
        )
        self.assertEqual(C.shape, (2, 2, lags.size))
        np.testing.assert_array_almost_equal(C[0, 1], C[1, 0][::-1])

    @needs_numba
    def test_allpairs_autocorr_zero_lag_excluded(self) -> None:
        """Zero-lag autocorr bin is blanked when requested."""
        rng = np.random.default_rng(1)
        u0 = np.sort(rng.uniform(0, 10, size=300))
        lags, C = ccg_allpairs_sparse(
            [u0],
            bin_size=0.001,
            max_lag=0.02,
            exclude_zero_lag_autocorr=True,
            observation_window=(0.0, 10.0),
        )
        center = lags.size // 2
        self.assertEqual(C[0, 0, center], 0.0)

    @needs_numba
    def test_corrcoef_normalization_is_finite(self) -> None:
        """corrcoef normalization returns finite correlation values."""
        rng = np.random.default_rng(2)
        u0 = np.sort(rng.uniform(0, 10, size=200))
        u1 = np.sort(rng.uniform(0, 10, size=200))
        _, C = ccg_allpairs_sparse(
            [u0, u1],
            bin_size=0.001,
            max_lag=0.02,
            normalize="corrcoef",
            observation_window=(0.0, 10.0),
        )
        self.assertTrue(np.all(np.isfinite(C)))

    def test_unsorted_spikes_raise(self) -> None:
        """Unsorted spike trains are rejected."""
        with self.assertRaises(ValueError):
            ccg_allpairs_sparse(
                [np.array([0.3, 0.1, 0.2])],
                observation_window=(0.0, 1.0),
            )

    def test_smooth_and_rescale(self) -> None:
        """Smoothing preserves shape; rescaling maps into [0, 1]."""
        rng = np.random.default_rng(3)
        C = rng.uniform(0, 5, size=(2, 2, 41))
        smoothed = smooth_ccgs(C, bin_size=0.001, kernel_width=0.003)
        self.assertEqual(smoothed.shape, C.shape)
        scaled = rescale_ccgs(C, axis=-1)
        self.assertTrue(np.all(scaled >= -1e-9))
        self.assertTrue(np.all(scaled <= 1 + 1e-9))


@needs_numba
class CCGWrapperTest(unittest.TestCase):
    """The xarray ccg wrapper dispatches to the lower-level implementations."""

    def test_single_trial_allpairs_matches_sparse(self) -> None:
        """One-trial ragged input with no selection uses all-pairs CCG."""
        data = np.empty((2, 1), dtype=object)
        data[0, 0] = np.array([0.1, 0.2, 0.4])
        data[1, 0] = np.array([0.105, 0.205, 0.7])
        spikes = _ragged_da(data, units=["a", "b"])

        lags, out = ccg(
            spikes, bin_size=0.001, max_lag=0.02, window=(0.0, 1.0)
        )
        lags_ref, ref = ccg_allpairs_sparse(
            [data[0, 0], data[1, 0]],
            bin_size=0.001,
            max_lag=0.02,
            observation_window=(0.0, 1.0),
        )

        np.testing.assert_array_equal(lags, lags_ref)
        np.testing.assert_array_equal(out.values, ref)
        self.assertEqual(out.dims, ("source_unit", "target_unit", "lag"))
        self.assertEqual(out.coords["source_unit"].values.tolist(), ["a", "b"])

    def test_single_trial_source_target_matches_between_sets(self) -> None:
        """Source/target selections use between-set sparse CCG."""
        data = np.empty((3, 1), dtype=object)
        data[0, 0] = np.array([0.1, 0.2, 0.4])
        data[1, 0] = np.array([0.105, 0.205, 0.7])
        data[2, 0] = np.array([0.15, 0.25])
        spikes = _ragged_da(data, units=["a", "b", "c"])

        _, out = ccg(
            spikes,
            source_units=["a"],
            target_units=["b", "c"],
            bin_size=0.001,
            max_lag=0.02,
            window=(0.0, 1.0),
        )
        _, ref = ccg_between_sets_sparse(
            [data[0, 0]],
            [data[1, 0], data[2, 0]],
            bin_size=0.001,
            max_lag=0.02,
            observation_window=(0.0, 1.0),
        )

        np.testing.assert_array_equal(out.values, ref)
        self.assertEqual(out.coords["source_unit"].values.tolist(), ["a"])
        self.assertEqual(out.coords["target_unit"].values.tolist(), ["b", "c"])

    def test_explicit_pairs_return_pair_layout(self) -> None:
        """Explicit label pairs return a compact pair x lag DataArray."""
        data = np.empty((3, 1), dtype=object)
        data[0, 0] = np.array([0.1, 0.2, 0.4])
        data[1, 0] = np.array([0.105, 0.205, 0.7])
        data[2, 0] = np.array([0.15, 0.25])
        spikes = _ragged_da(data, units=["a", "b", "c"])

        _, out = ccg(
            spikes,
            pairs=[("a", "b"), ("c", "b")],
            bin_size=0.001,
            max_lag=0.02,
            window=(0.0, 1.0),
        )

        self.assertEqual(out.dims, ("pair", "lag"))
        self.assertEqual(out.coords["source_unit"].values.tolist(), ["a", "c"])
        self.assertEqual(out.coords["target_unit"].values.tolist(), ["b", "b"])

    def test_multi_trial_identity_matches_trial_paired(self) -> None:
        """Multi-trial ragged input uses trial-paired CCG with identity pairing."""
        rng = np.random.default_rng(4)
        n_units = 3
        n_trials = 5
        trial_len = 1.0
        starts = np.arange(n_trials) * 2.0
        data = np.empty((n_units, n_trials), dtype=object)
        abs_spikes = []
        for i in range(n_units):
            per_trial_abs = []
            for k, start in enumerate(starts):
                rel = np.sort(rng.uniform(0.0, trial_len, size=10 + i))
                data[i, k] = rel
                per_trial_abs.append(start + rel)
            abs_spikes.append(np.sort(np.concatenate(per_trial_abs)))
        spikes = _ragged_da(data, window=(0.0, trial_len))
        trial_epochs = np.column_stack([starts, starts + trial_len])
        trial_segments = clip_spikes_to_trials(
            abs_spikes, trial_epochs, align_times=starts
        )

        lags, out = ccg(
            spikes, bin_size=0.001, max_lag=0.02, window=(0.0, trial_len)
        )
        lags_ref, ref = ccg_trial_paired(
            trial_segments,
            np.arange(n_trials),
            bin_size=0.001,
            max_lag=0.02,
            normalize="none",
        )

        np.testing.assert_array_equal(lags, lags_ref)
        np.testing.assert_array_equal(out.values, ref)

    def test_adapter_ragged_output_is_accepted(self) -> None:
        """ccg can directly ingest from_dataframe ragged spikes."""
        units_df = pd.DataFrame(
            {
                "unit_id": ["a", "b"],
                "spike_times": [
                    np.array([0.1, 0.2, 1.1, 1.2]),
                    np.array([0.105, 0.205, 1.105, 1.205]),
                ],
            }
        )
        trials_df = pd.DataFrame(
            {
                "trial_id": [0, 1],
                "start_time": [0.0, 1.0],
                "end_time": [0.5, 1.5],
            }
        )
        spikes = from_dataframe(
            units_df,
            trials_df,
            unit_id_col="unit_id",
            trial_id_col="trial_id",
            window=(0.0, 0.5),
        )

        _, out = ccg(spikes, pairs=[("a", "b")], bin_size=0.001, max_lag=0.02)

        self.assertEqual(out.dims, ("pair", "lag"))
        self.assertEqual(out.sizes["pair"], 1)
        self.assertEqual(out.coords["source_unit"].values.tolist(), ["a"])
        self.assertEqual(out.coords["target_unit"].values.tolist(), ["b"])


if __name__ == "__main__":
    unittest.main()
