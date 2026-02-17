"""PSTH operations -- pure numpy, no xarray."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def psth(
    data: NDArray[np.float64],
    *,
    method: str = "mean",
    labels: NDArray | None = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray]:
    """Reduce across trials (axis 0) to compute a PSTH.

    Parameters
    ----------
    data : ndarray, shape (n_trials, ...)
        Dense data. First axis is trials.
    method : str
        ``"mean"`` or ``"median"``.
    labels : ndarray, shape (n_trials,), optional
        Per-trial group labels. If provided, returns one summary per
        unique label.

    Returns
    -------
    If *labels* is ``None``:
        ndarray, shape ``data.shape[1:]``
            Trial-averaged data.
    If *labels* is provided:
        ``(result, unique_labels)`` where *result* has shape
        ``(n_groups, *data.shape[1:])`` and *unique_labels* has shape
        ``(n_groups,)``.
    """
    method = method.lower()
    if method not in ("mean", "median"):
        raise ValueError(f"Unknown PSTH method {method!r}.")

    reduce_fn = np.mean if method == "mean" else np.median

    if labels is None:
        return reduce_fn(data, axis=0)

    labels = np.asarray(labels)
    unique = np.unique(labels)
    n_groups = len(unique)

    result = np.empty((n_groups,) + data.shape[1:], dtype=data.dtype)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        result[i] = reduce_fn(data[mask], axis=0)

    return result, unique
