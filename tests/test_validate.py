"""Tests for core validation utilities."""

import unittest

import numpy as np
import xarray as xr

from aind_ephys_utils.core.validate import (
    EphysValidationError,
    infer_kind,
    validate,
)


class ValidateTest(unittest.TestCase):
    """Coverage for validate helpers."""

    def test_infer_kind_ragged(self) -> None:
        """Infer ragged spikes from object dtype with trial/unit dims."""
        data = np.empty((1, 1), dtype=object)
        data[0, 0] = np.array([0.1])
        da = xr.DataArray(data=data, dims=("trial", "unit"))
        self.assertEqual(infer_kind(da), "spikes_ragged")

    def test_validate_binned_requires_time_coord(self) -> None:
        """Binned data without time coords should error."""
        da = xr.DataArray(
            np.zeros((2, 2, 3)),
            dims=("trial", "unit", "time"),
            coords={"trial": [0, 1], "unit": [0, 1]},
        )
        with self.assertRaises(EphysValidationError):
            validate(da)

    def test_validate_kind_mismatch(self) -> None:
        """Explicit kind mismatch raises validation error."""
        da = xr.DataArray(
            np.zeros((2, 2, 3)),
            dims=("trial", "unit", "time"),
            coords={
                "trial": [0, 1],
                "unit": [0, 1],
                "time": [0.0, 1.0, 2.0],
            },
        )
        with self.assertRaises(EphysValidationError):
            validate(da, kind="spikes_ragged")
