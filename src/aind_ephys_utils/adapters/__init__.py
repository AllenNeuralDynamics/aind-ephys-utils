"""Adapters for converting between common data formats and canonical types.

Adapters are responsible for building canonical representations used by
``.ephys`` methods and the :mod:`~aind_ephys_utils.types` dataclass types.
"""

from .dataframe import from_dataframe
from .xarray import (
    binned_to_xarray,
    ragged_to_xarray,
    xarray_to_binned,
    xarray_to_ragged,
)

__all__ = [
    "from_dataframe",
    "binned_to_xarray",
    "ragged_to_xarray",
    "xarray_to_binned",
    "xarray_to_ragged",
]
