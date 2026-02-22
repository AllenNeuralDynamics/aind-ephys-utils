"""Configuration helpers for optional global defaults."""

from __future__ import annotations

from typing import Optional


def configure_defaults(
    *,
    plot: bool = True,
    display: bool = True,
    print_options: bool = True,
    savefig_dpi: Optional[int] = 300,
) -> None:
    """Apply recommended global defaults explicitly.

    Parameters
    ----------
    plot:
        If True, apply matplotlib defaults for export-friendly figures.
    display:
        If True, set xarray display options for compact DataArray rendering.
    print_options:
        If True, set numpy print options for concise array display.
    savefig_dpi:
        Save DPI to set when ``plot=True``. If None, leaves current value.
    """
    if plot:
        import matplotlib as mpl

        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["savefig.bbox"] = "tight"
        if savefig_dpi is not None:
            mpl.rcParams["savefig.dpi"] = int(savefig_dpi)

    if display:
        import xarray as xr

        xr.set_options(display_expand_data=False)

    if print_options:
        import numpy as np

        np.set_printoptions(threshold=10, edgeitems=2)
