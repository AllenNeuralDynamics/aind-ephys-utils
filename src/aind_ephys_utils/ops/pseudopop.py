"""Pseudopopulation assembly helpers."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import xarray as xr

from ..standards.conventions import C
from .psth import psth


def pseudopop(
    das: Sequence[xr.DataArray],
    *,
    group_by: Union[str, Sequence[str]],
    session_ids: Optional[Sequence[object]] = None,
    unit_dim: str = C.unit,
) -> xr.DataArray:
    """
    Build a pseudopopulation by PSTH-averaging and concatenating sessions.

    Parameters
    ----------
    das:
        Session DataArrays to combine.
    group_by:
        Trial coord(s) used to compute per-condition PSTHs in each session.
    session_ids:
        Optional per-session identifiers. If omitted, IDs are ``s0``, ``s1``, ...
    unit_dim:
        Unit dimension name used for concatenation.
    """
    if len(das) == 0:
        raise ValueError("das must contain at least one DataArray.")
    if session_ids is not None and len(session_ids) != len(das):
        raise ValueError("session_ids length must match len(das).")

    psths: list[xr.DataArray] = []
    for i, da in enumerate(das):
        if unit_dim not in da.dims:
            raise ValueError(
                f"unit_dim {unit_dim!r} not found in session {i}."
            )
        sid = session_ids[i] if session_ids is not None else f"s{i}"
        p = psth(da, dim=C.trial, method="mean", group_by=group_by)
        p = p.assign_coords({"session": (unit_dim, [sid] * p.sizes[unit_dim])})
        psths.append(p)

    out = xr.concat(
        psths,
        dim=unit_dim,
        coords="minimal",
        compat="override",
        join="outer",
    )
    out = out.assign_coords({unit_dim: np.arange(out.sizes[unit_dim])})
    return out
