"""Tests for explicit global configuration helpers."""

import unittest

import matplotlib as mpl
import numpy as np
import xarray as xr

from aind_ephys_utils import configure_defaults


class ConfigTest(unittest.TestCase):
    """Validate explicit configuration behavior."""

    def test_configure_defaults_applies_requested_settings(self) -> None:
        """configure_defaults should update plot/display/print settings."""
        old_pdf = mpl.rcParams["pdf.fonttype"]
        old_ps = mpl.rcParams["ps.fonttype"]
        old_dpi = mpl.rcParams["savefig.dpi"]
        old_bbox = mpl.rcParams["savefig.bbox"]
        old_np = np.get_printoptions()
        old_xr_expand = xr.get_options()["display_expand_data"]

        try:
            mpl.rcParams["pdf.fonttype"] = 3
            mpl.rcParams["ps.fonttype"] = 3
            mpl.rcParams["savefig.dpi"] = 72
            mpl.rcParams["savefig.bbox"] = None
            np.set_printoptions(threshold=1000, edgeitems=10)
            xr.set_options(display_expand_data=old_xr_expand)

            configure_defaults(savefig_dpi=300)

            self.assertEqual(mpl.rcParams["pdf.fonttype"], 42)
            self.assertEqual(mpl.rcParams["ps.fonttype"], 42)
            self.assertEqual(mpl.rcParams["savefig.dpi"], 300.0)
            self.assertEqual(mpl.rcParams["savefig.bbox"], "tight")
            self.assertEqual(np.get_printoptions()["threshold"], 10)
            self.assertEqual(np.get_printoptions()["edgeitems"], 2)
            self.assertFalse(xr.get_options()["display_expand_data"])
        finally:
            mpl.rcParams["pdf.fonttype"] = old_pdf
            mpl.rcParams["ps.fonttype"] = old_ps
            mpl.rcParams["savefig.dpi"] = old_dpi
            mpl.rcParams["savefig.bbox"] = old_bbox
            np.set_printoptions(**old_np)
            xr.set_options(display_expand_data=True)
