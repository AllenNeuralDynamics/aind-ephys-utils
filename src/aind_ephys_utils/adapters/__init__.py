"""Adapters for converting common data formats into xarray objects.

Adapters are responsible for building canonical xarray representations used by
`.ephys` methods.
"""

from .dataframe import from_dataframe

__all__ = ["from_dataframe"]
