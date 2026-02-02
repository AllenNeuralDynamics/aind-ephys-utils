"""Dimensionality reduction operations.

This module will contain xarray-native reduction helpers (e.g., PCA) used by
the `.ephys.reduce` accessor.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from ..standards.conventions import C
from .dpca import dPCA
from .utils import preserve_coords


def reduce(  # noqa: C901
    da: xr.DataArray,
    *,
    method: str,
    dim: Optional[str] = None,
    n_components: Optional[int] = None,
    stack: Optional[Tuple[str, ...]] = (C.trial, C.time),
    unstack: bool = True,
    return_dataset: bool = True,
    window: Optional[
        Union[Tuple[float, float], Sequence[Tuple[float, float]]]
    ] = None,
    window_apply: str = "fit_only",
    orthogonalize: str = "none",
    orthogonalize_across: str = "none",
    trial_dim: str = C.trial,
    time_dim: str = C.time,
    trial_average: bool = True,
    labels: Optional[Union[Sequence[str], str]] = None,
    targets: Optional[xr.DataArray] = None,
    rank: Optional[int] = None,
    regularization: Optional[float] = None,
    cv: Optional[int] = None,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Reduce data dimensionality in an xarray-friendly way.

    Parameters
    ----------
    da:
        Input DataArray.
    method:
        Reduction method (e.g. "pca", "dpca",
        "coding_direction", "logistic", "lda", "rrr").
    dim:
        Dimension to reduce across for methods that operate on a single axis.
    n_components:
        Number of components to keep (PCA, dPCA).
    stack:
        Dims to stack before reduction (default: (trial, time)).
    unstack:
        If True, unstack stacked dims in the output.
    return_dataset:
        If True, return a Dataset with projections and weights (and explained variance
        for PCA). For dPCA and supervised methods, a Dataset is always returned.
    window:
        Optional (tmin, tmax) window used for fitting supervised methods.
    window_apply:
        "fit_only" (default) fits on the window but projects all samples;
        "fit_and_project" fits and projects only within the window.
    orthogonalize:
        How to orthogonalize supervised components: "none", "qr", or "svd".
    orthogonalize_across:
        How to orthogonalize across multiple windows/labels: "none", "windows",
        "labels", or "all".
    trial_dim:
        Trial dimension name.
    time_dim:
        Time dimension name.
    trial_average:
        If True (default), average across trials before dPCA marginalization.
    labels:
        Coordinate name(s) used for dPCA and supervised methods
        (coding direction, logistic, lda). Must exist in `da.coords`.
    targets:
        Target matrix for reduced-rank regression.
    rank:
        Rank for reduced-rank regression.
    regularization:
        Regularization strength for supervised methods.
    cv:
        Cross-validation folds for supervised methods.
    """
    method = method.lower()
    if method not in (
        "pca",
        "dpca",
        "coding_direction",
        "logistic",
        "lda",
        "rrr",
    ):
        raise NotImplementedError(
            f"Reduction method {method!r} is not implemented yet."
        )

    if stack is not None:
        stack = tuple(d for d in stack if d in da.dims)

    if method == "dpca":
        if n_components is None:
            raise ValueError("n_components is required for method='dpca'.")
        if labels is None:
            raise ValueError("labels is required for method='dpca'.")
        return _reduce_dpca(
            da,
            n_components=n_components,
            dim=dim,
            trial_dim=trial_dim,
            time_dim=time_dim,
            labels=labels,
            trial_average=trial_average,
        )

    if method in ("coding_direction", "logistic", "lda", "rrr"):
        return _reduce_supervised(
            da,
            method=method,
            dim=dim,
            n_components=n_components,
            stack=stack,
            unstack=unstack,
            window=window,
            window_apply=window_apply,
            orthogonalize=orthogonalize,
            orthogonalize_across=orthogonalize_across,
            trial_dim=trial_dim,
            time_dim=time_dim,
            labels=labels,
            targets=targets,
            rank=rank,
            regularization=regularization,
            cv=cv,
        )

    if n_components is None:
        raise ValueError("n_components is required for PCA methods.")

    if method == "pca":
        if dim is None:
            raise ValueError("dim is required for method='pca'.")
        if dim not in da.dims:
            raise ValueError(f"dim {dim!r} not found in DataArray dims.")
        if stack is None:
            stack = tuple(d for d in da.dims if d != dim)
        sample_dims = stack
        feature_dims = _feature_dims(da.dims, sample_dims, dim=dim)

    da_stack = _stack_for_pca(da, sample_dims, feature_dims)
    X = np.asarray(da_stack.data)
    if np.isnan(X).any():
        raise ValueError("PCA does not support NaN values.")

    model = PCA(n_components=n_components)
    scores = model.fit_transform(X)

    out = xr.DataArray(
        scores,
        dims=("_sample", "component"),
        coords={
            "_sample": da_stack["_sample"],
            "component": np.arange(n_components),
        },
        name=da.name,
        attrs=dict(da.attrs),
    ).transpose("component", "_sample")

    if unstack:
        out = out.unstack("_sample")

    out = preserve_coords(da, out)
    if return_dataset:
        evr = xr.DataArray(
            model.explained_variance_ratio_,
            dims=("component",),
            coords={"component": np.arange(n_components)},
            name="explained_variance_ratio",
        )
        weights = _weights_to_dataarray(
            model.components_, da_stack, component_count=n_components
        )
        ds = xr.Dataset(
            {
                "projections": out,
                "weights": weights,
                "explained_variance_ratio": evr,
            }
        )
        return ds
    return out


def _feature_dims(
    dims: Sequence[str], sample_dims: Sequence[str], dim: Optional[str]
) -> Tuple[str, ...]:
    """Resolve feature dimensions given sample dims and optional preferred dim."""
    feature_dims = [d for d in dims if d not in sample_dims]
    if dim is not None:
        if dim not in feature_dims:
            raise ValueError(f"dim {dim!r} not found in feature dims.")
        feature_dims = [dim] + [d for d in feature_dims if d != dim]
    if not feature_dims:
        raise ValueError("Need at least one feature dimension for PCA.")
    return tuple(feature_dims)


def _stack_for_pca(
    da: xr.DataArray, sample_dims: Sequence[str], feature_dims: Sequence[str]
) -> xr.DataArray:
    """Stack sample/feature dims for PCA-style linear algebra."""
    if not sample_dims:
        raise ValueError("Need at least one sample dimension for PCA.")
    if not feature_dims:
        raise ValueError("Need at least one feature dimension for PCA.")
    missing = [d for d in sample_dims if d not in da.dims]
    if missing:
        raise ValueError(f"Sample dims {missing} not found in DataArray.")
    missing = [d for d in feature_dims if d not in da.dims]
    if missing:
        raise ValueError(f"Feature dims {missing} not found in DataArray.")

    da_stack = da.stack(_sample=tuple(sample_dims)).stack(
        _feature=tuple(feature_dims)
    )
    return da_stack.transpose("_sample", "_feature")


def _reduce_dpca(  # noqa: C901
    da: xr.DataArray,
    *,
    n_components: int,
    dim: Optional[str],
    trial_dim: str,
    time_dim: str,
    labels: Optional[Union[Sequence[str], str]],
    trial_average: bool,
) -> xr.Dataset:
    """dPCA implementation using the original solver."""
    if labels is None:
        raise ValueError("labels is required for method='dpca'.")
    if isinstance(labels, str):
        condition_dims = (labels,)
    else:
        condition_dims = tuple(labels)
    if len(condition_dims) == 0:
        raise ValueError("labels is required for method='dpca'.")
    if trial_dim not in da.dims:
        raise ValueError(
            f"trial_dim {trial_dim!r} not found in DataArray dims."
        )
    if time_dim not in da.dims:
        raise ValueError(f"time_dim {time_dim!r} not found in DataArray dims.")
    if dim is None:
        if "unit" in da.dims:
            dim = "unit"
        else:
            raise ValueError("dim is required for method='dpca'.")

    if trial_average:
        da_cond = _condition_mean(
            da, trial_dim=trial_dim, condition_dims=condition_dims
        )
    else:
        if trial_dim in da.dims:
            raise ValueError(
                "trial_average=False requires data without the trial dimension. "
                "Average across trials first or set trial_average=True."
            )
        missing = [d for d in condition_dims if d not in da.dims]
        if missing:
            raise ValueError(
                "trial_average=False requires condition_dims to be dimensions; "
                f"missing dims: {missing}."
            )
        da_cond = da

    factor_dims = list(condition_dims) + [time_dim]
    missing = [d for d in factor_dims if d not in da_cond.dims]
    if missing:
        raise ValueError(f"factor dims {missing} not found in DataArray dims.")

    da_dpca = da_cond.transpose(dim, *factor_dims)
    X = np.asarray(da_dpca.data)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)

    dpca = dPCA(labels=len(factor_dims), n_components=n_components)
    dpca.fit(X)
    transformed = dpca.transform(X)

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    label_map = {alphabet[i]: factor_dims[i] for i in range(len(factor_dims))}

    proj_list = []
    weights_list = []
    labels = []
    for key in dpca.marginalizations.keys():
        factors = tuple(label_map[ch] for ch in key)
        label = _marginal_label(factors, time_dim=time_dim)
        Z = transformed[key]
        proj = xr.DataArray(
            Z,
            dims=("component",) + tuple(factor_dims),
            coords={
                "component": np.arange(Z.shape[0]),
                **{d: da_dpca.coords[d] for d in factor_dims},
            },
            name=da.name,
            attrs=dict(da.attrs),
        )
        D = dpca.D[key]
        weights = xr.DataArray(
            D.T,
            dims=("component", dim),
            coords={
                "component": np.arange(D.shape[1]),
                dim: da_dpca.coords[dim],
            },
            name="weights",
        )
        proj_list.append(proj)
        weights_list.append(weights)
        labels.append(label)

    projections = xr.concat(proj_list, dim="marginal")
    weights = xr.concat(weights_list, dim="marginal")
    projections = projections.assign_coords(
        marginal=np.asarray(labels, dtype=object)
    )
    weights = weights.assign_coords(marginal=np.asarray(labels, dtype=object))
    return xr.Dataset({"projections": projections, "weights": weights})


def _reduce_supervised(  # noqa: C901
    da: xr.DataArray,
    *,
    method: str,
    dim: Optional[str],
    n_components: Optional[int],
    stack: Optional[Tuple[str, ...]],
    unstack: bool,
    window: Optional[
        Union[Tuple[float, float], Sequence[Tuple[float, float]]]
    ],
    window_apply: str,
    orthogonalize: str,
    orthogonalize_across: str,
    trial_dim: str,
    time_dim: str,
    labels: Optional[Union[Sequence[str], str]],
    targets: Optional[xr.DataArray],
    rank: Optional[int],
    regularization: Optional[float],
    cv: Optional[int],
) -> xr.Dataset:
    """Supervised linear reductions (coding direction, logistic, lda, rrr)."""
    if cv is not None:
        raise NotImplementedError(
            "cv is not implemented for supervised methods."
        )
    if dim is None:
        if C.unit in da.dims:
            dim = C.unit
        else:
            raise ValueError("dim is required for supervised methods.")

    if stack is None:
        if trial_dim not in da.dims:
            raise ValueError(
                f"trial_dim {trial_dim!r} not found in DataArray dims."
            )
        sample_dims = (trial_dim,)
    else:
        sample_dims = stack

    feature_dims = _feature_dims(da.dims, sample_dims, dim=dim)
    da_stack = _stack_for_pca(da, sample_dims, feature_dims)
    da_stack_full = da_stack
    X_full = np.asarray(da_stack_full.data)
    X = X_full
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
        X_full = X

    windows = _normalize_windows(window)
    if len(windows) > 1 and window_apply == "fit_and_project":
        raise ValueError(
            "window_apply='fit_and_project' is not supported with multiple windows."
        )
    for w in windows:
        if w is not None and time_dim not in sample_dims:
            raise ValueError("window requires time_dim to be in sample dims.")

    if labels is None:
        label_sets = []
    else:
        if isinstance(labels, str):
            label_names = [labels]
        else:
            label_names = list(labels)
        label_sets = []
        for name in label_names:
            if name not in da.coords:
                raise ValueError(
                    f"label coord {name!r} not found in DataArray coords."
                )
            label_sets.append((name, da.coords[name]))
    mode_entries: list[Dict[str, object]] = []

    if method in ("coding_direction", "logistic", "lda"):
        if not label_sets:
            raise ValueError("labels are required for supervised methods.")
        for label_name, label_da in label_sets:
            y_full = _stack_labels(label_da, sample_dims=sample_dims, ref=da)
            base_mask_full = _label_mask(y_full)
            for w in windows:
                window_mask = (
                    _window_mask(da_stack_full, time_dim=time_dim, window=w)
                    if w is not None
                    else None
                )
                if w is None:
                    X_fit = X[base_mask_full]
                    y_fit = y_full[base_mask_full]
                else:
                    in_window = (da[time_dim] >= w[0]) & (da[time_dim] <= w[1])
                    da_fit = da.where(in_window, drop=True)
                    da_stack_fit = _stack_for_pca(
                        da_fit, sample_dims, feature_dims
                    )
                    X_fit = np.asarray(da_stack_fit.data)
                    if np.isnan(X_fit).any():
                        X_fit = np.nan_to_num(X_fit, nan=0.0)
                    y_fit = _stack_labels(
                        label_da, sample_dims=sample_dims, ref=da_fit
                    )
                    base_mask = _label_mask(y_fit)
                    X_fit = X_fit[base_mask]
                    y_fit = y_fit[base_mask]
                if X_fit.size == 0:
                    raise ValueError(
                        "No samples available after window/label filtering."
                    )
                if method == "coding_direction":
                    classes = np.unique(y_fit)
                    if classes.size != 2:
                        raise ValueError(
                            "coding_direction requires exactly 2 classes."
                        )
                    mean0 = X_fit[y_fit == classes[0]].mean(axis=0)
                    mean1 = X_fit[y_fit == classes[1]].mean(axis=0)
                    wts = (mean1 - mean0)[None, :]
                elif method == "logistic":
                    if np.unique(y_fit).size < 2:
                        raise ValueError(
                            "logistic requires at least 2 classes."
                        )
                    if regularization is None:
                        C_val = 1.0
                    elif regularization <= 0:
                        C_val = 1e12
                    else:
                        C_val = 1.0 / regularization
                    try:
                        model = LogisticRegression(
                            max_iter=1000, C=C_val, multi_class="auto"
                        )
                    except TypeError:
                        model = LogisticRegression(max_iter=1000, C=C_val)
                    model.fit(X_fit, y_fit)
                    wts = model.coef_
                    bias = model.intercept_
                else:  # lda
                    if np.unique(y_fit).size < 2:
                        raise ValueError("lda requires at least 2 classes.")
                    model = LinearDiscriminantAnalysis()
                    model.fit(X_fit, y_fit)
                    if model.coef_.shape[0] == 2:
                        wts = (model.coef_[1] - model.coef_[0])[None, :]
                        bias = (model.intercept_[1] - model.intercept_[0])[
                            None
                        ]
                    else:
                        wts = model.coef_
                        bias = model.intercept_

                wts = _orthogonalize_weights(wts, method=orthogonalize)
                if method not in ("logistic", "lda"):
                    bias = None
                X_proj, da_proj = _select_projection(
                    X_full, da_stack_full, window_mask, window_apply
                )
                mode_entries.append(
                    {
                        "label": label_name,
                        "window": w,
                        "weights": wts,
                        "bias": bias,
                        "X_proj": X_proj,
                        "da_proj": da_proj,
                    }
                )

    if method in ("coding_direction", "logistic", "lda"):
        return _assemble_supervised_output(
            da,
            mode_entries=mode_entries,
            orthogonalize_across=orthogonalize_across,
            orthogonalize=orthogonalize,
            unstack=unstack,
            da_stack_full=da_stack_full,
        )

    if method == "rrr":
        if targets is None:
            raise ValueError("targets are required for method='rrr'.")
        Y, target_dim = _stack_targets(targets, sample_dims=sample_dims)
        mask = _target_mask(Y)
        for w in windows:
            window_mask = (
                _window_mask(da_stack_full, time_dim=time_dim, window=w)
                if w is not None
                else None
            )
            mask_w = mask
            if window_mask is not None:
                mask_w = mask_w & window_mask
            X_fit, Y_fit = X[mask_w], Y[mask_w]
            da_stack = da_stack_full.isel(_sample=mask_w)
            if X_fit.size == 0:
                raise ValueError(
                    "No samples available after window/target filtering."
                )
            if regularization is None:
                reg = 0.0
            else:
                reg = float(regularization)
            XtX = X_fit.T @ X_fit
            if reg > 0:
                XtX = XtX + reg * np.eye(XtX.shape[0])
            B = np.linalg.pinv(XtX) @ X_fit.T @ Y_fit
            U, S, Vt = np.linalg.svd(B, full_matrices=False)
            r = rank if rank is not None else n_components
            if r is None:
                raise ValueError(
                    "rank or n_components is required for method='rrr'."
                )
            r = int(r)
            U_r = U[:, :r]
            U_r = _orthogonalize_weights(U_r.T, method=orthogonalize).T
            X_proj, da_proj = _select_projection(
                X_full, da_stack_full, window_mask, window_apply
            )
            mode_entries.append(
                {
                    "label": "rrr",
                    "window": w,
                    "weights": U_r.T,
                    "X_proj": X_proj,
                    "da_proj": da_proj,
                }
            )
        return _assemble_supervised_output(
            da,
            mode_entries=mode_entries,
            orthogonalize_across=orthogonalize_across,
            orthogonalize=orthogonalize,
            unstack=unstack,
            da_stack_full=da_stack_full,
        )

    raise NotImplementedError(f"Unhandled supervised method {method!r}.")


def _condition_mean(
    da: xr.DataArray, *, trial_dim: str, condition_dims: Sequence[str]
) -> xr.DataArray:
    """Compute condition means across trials."""
    for c in condition_dims:
        if c not in da.coords:
            raise ValueError(f"condition coord {c!r} not found in DataArray.")
        if trial_dim not in da[c].dims:
            raise ValueError(
                f"condition coord {c!r} must have dims including {trial_dim!r}."
            )
    cond_index = pd.MultiIndex.from_arrays(
        [da[c].values for c in condition_dims], names=list(condition_dims)
    )
    da2 = da.assign_coords(_cond=(trial_dim, cond_index))
    mean = da2.groupby("_cond").mean(dim=trial_dim, keep_attrs=True)
    return mean.unstack("_cond").fillna(0.0)


def _marginal_label(factors: Sequence[str], *, time_dim: str) -> str:
    """Build a readable marginalization label."""
    parts = ["time" if f == time_dim else str(f) for f in factors]
    return "_".join(parts)


def _marginalize(
    da: xr.DataArray, *, factor_dims: Sequence[str]
) -> Tuple[xr.DataArray, Dict[Tuple[str, ...], xr.DataArray]]:
    """ANOVA-style marginalization over factor subsets."""
    grand = da.mean(dim=list(factor_dims), keep_attrs=True)
    marginals: Dict[Tuple[str, ...], xr.DataArray] = {}
    subsets = _nonempty_subsets(list(factor_dims))
    for subset in subsets:
        mean = da.mean(
            dim=[d for d in factor_dims if d not in subset], keep_attrs=True
        )
        for smaller in _proper_subsets(list(subset)):
            mean = mean - marginals[tuple(smaller)]
        marginals[tuple(subset)] = mean
    return grand, marginals


def _pca_marginal(
    da: xr.DataArray,
    *,
    feature_dim: str,
    n_components: int,
    project_da: Optional[xr.DataArray] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """PCA for a single marginalization."""
    sample_dims = [d for d in da.dims if d != feature_dim]
    da_stack = _stack_for_pca(da, sample_dims, [feature_dim])
    X = np.asarray(da_stack.data)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    model = PCA(n_components=n_components)
    model.fit(X)
    if project_da is None:
        X_proj = X
        da_proj = da_stack
    else:
        da_proj = _stack_for_pca(project_da, sample_dims, [feature_dim])
        X_proj = np.asarray(da_proj.data)
        if np.isnan(X_proj).any():
            X_proj = np.nan_to_num(X_proj, nan=0.0)
    scores = X_proj @ model.components_.T
    out = _scores_to_dataarray(
        scores,
        da_proj,
        component_count=n_components,
        unstack=True,
    )
    weights = _weights_to_dataarray(
        model.components_, da_stack, component_count=n_components
    )
    return out, weights


def _nonempty_subsets(items: Sequence[str]) -> Sequence[Tuple[str, ...]]:
    """Return all non-empty subsets of items."""
    out: list[Tuple[str, ...]] = []
    n = len(items)
    for k in range(1, n + 1):
        for idx in combinations(range(n), k):
            out.append(tuple(items[i] for i in idx))
    return out


def _proper_subsets(items: Sequence[str]) -> Sequence[Tuple[str, ...]]:
    """Return all proper (non-empty, non-full) subsets."""
    out: list[Tuple[str, ...]] = []
    n = len(items)
    for k in range(1, n):
        for idx in combinations(range(n), k):
            out.append(tuple(items[i] for i in idx))
    return out


def _stack_labels(
    labels: xr.DataArray, *, sample_dims: Sequence[str], ref: xr.DataArray
) -> np.ndarray:
    """Stack labels to sample axis, broadcasting across missing sample dims."""
    if not isinstance(labels, xr.DataArray):
        raise TypeError("labels must be an xarray.DataArray.")
    missing = [d for d in sample_dims if d not in labels.dims]
    if missing:
        if all(d in ref.dims for d in missing):
            ref_sample = ref
            for d in ref.dims:
                if d not in sample_dims:
                    ref_sample = ref_sample.isel({d: 0})
            labels = labels.broadcast_like(ref_sample)
        else:
            raise ValueError(f"labels missing sample dims {missing}.")
    lab = labels.stack(_sample=tuple(sample_dims))
    return np.asarray(lab.values)


def _label_mask(y: np.ndarray) -> np.ndarray:
    """Mask NaN labels."""
    return ~pd.isna(y)


def _stack_targets(
    targets: xr.DataArray, *, sample_dims: Sequence[str]
) -> Tuple[np.ndarray, str]:
    """Stack targets to sample axis."""
    if not isinstance(targets, xr.DataArray):
        raise TypeError("targets must be an xarray.DataArray.")
    missing = [d for d in sample_dims if d not in targets.dims]
    if missing:
        raise ValueError(f"targets missing sample dims {missing}.")
    target_dims = [d for d in targets.dims if d not in sample_dims]
    if len(target_dims) != 1:
        raise ValueError("targets must have exactly one non-sample dim.")
    tdim = target_dims[0]
    tstack = targets.stack(_sample=tuple(sample_dims)).transpose(
        "_sample", tdim
    )
    Y = np.asarray(tstack.data)
    if np.isnan(Y).any():
        Y = np.nan_to_num(Y, nan=0.0)
    return Y, tdim


def _target_mask(Y: np.ndarray) -> np.ndarray:
    """Mask rows with NaN targets."""
    return ~np.isnan(Y).any(axis=1)


def _scores_to_dataarray(
    scores: np.ndarray,
    da_stack: xr.DataArray,
    *,
    component_count: int,
    unstack: bool,
) -> xr.DataArray:
    """Wrap projected scores into a DataArray with sample coords."""
    out = xr.DataArray(
        scores,
        dims=("_sample", "component"),
        coords={
            "_sample": da_stack["_sample"],
            "component": np.arange(component_count),
        },
        name=da_stack.name,
        attrs=dict(da_stack.attrs),
    ).transpose("component", "_sample")
    if unstack:
        out = out.unstack("_sample")
    return out


def _weights_to_dataarray(
    weights: np.ndarray,
    da_stack: xr.DataArray,
    *,
    component_count: int,
) -> xr.DataArray:
    """Wrap weights into a DataArray with feature coords."""
    if weights.ndim == 1:
        weights = weights[None, :]
    out = xr.DataArray(
        weights,
        dims=("component", "_feature"),
        coords={
            "component": np.arange(component_count),
            "_feature": da_stack["_feature"],
        },
        name="weights",
    )
    return out.unstack("_feature")


def _expand_marginal_dims(
    da: xr.DataArray,
    *,
    factor_dims: Sequence[str],
    ref: xr.DataArray,
) -> xr.DataArray:
    """Ensure all factor dims exist, expanding to full coords."""
    out = da
    for d in factor_dims:
        if d not in out.dims:
            if d in ref.coords:
                coord = ref.coords[d]
            elif d in ref.dims:
                coord = np.arange(ref.sizes[d])
            else:
                coord = ["__all__"]
            out = out.expand_dims({d: coord})
    return out


def _window_mask(
    da_stack: xr.DataArray,
    *,
    time_dim: str,
    window: Optional[Tuple[float, float]],
) -> Optional[np.ndarray]:
    """Build a boolean mask for sample times inside the window."""
    if window is None:
        return None
    index = da_stack["_sample"].to_index()
    if time_dim not in index.names:
        return None
    tvals = index.get_level_values(time_dim).to_numpy()
    tmin, tmax = window
    return (tvals >= tmin) & (tvals <= tmax)


def _select_projection(
    X: np.ndarray,
    da_stack: xr.DataArray,
    window_mask: Optional[np.ndarray],
    window_apply: str,
) -> Tuple[np.ndarray, xr.DataArray]:
    """Select which samples to project based on window_apply."""
    if window_apply not in ("fit_only", "fit_and_project"):
        raise ValueError(
            "window_apply must be 'fit_only' or 'fit_and_project'."
        )
    if window_apply == "fit_and_project" and window_mask is not None:
        return X[window_mask], da_stack.isel(_sample=window_mask)
    return X, da_stack


def _orthogonalize_weights(weights: np.ndarray, *, method: str) -> np.ndarray:
    """Orthogonalize weight vectors."""
    method = method.lower()
    if method == "none":
        return weights
    if weights.ndim == 1:
        weights = weights[None, :]
    if method == "qr":
        q, _ = np.linalg.qr(weights.T)
        return q.T
    if method == "svd":
        u, _, vt = np.linalg.svd(weights, full_matrices=False)
        return u @ vt
    raise ValueError("orthogonalize must be 'none', 'qr', or 'svd'.")


def _normalize_windows(
    window: Optional[
        Union[Tuple[float, float], Sequence[Tuple[float, float]]]
    ],
) -> Sequence[Optional[Tuple[float, float]]]:
    """Normalize window input to a list."""
    if window is None:
        return [None]
    if isinstance(window, tuple) and len(window) == 2:
        return [window]
    return list(window)


def _assemble_supervised_output(  # noqa: C901
    da: xr.DataArray,
    *,
    mode_entries: Sequence[Dict[str, object]],
    orthogonalize_across: str,
    orthogonalize: str,
    unstack: bool,
    da_stack_full: xr.DataArray,
) -> xr.Dataset:
    """Assemble projections/weights across windows or labels."""
    if not mode_entries:
        raise ValueError("No supervised modes were computed.")

    orthogonalize_across = orthogonalize_across.lower()
    if orthogonalize_across not in ("none", "windows", "labels", "all"):
        raise ValueError(
            "orthogonalize_across must be 'none', 'windows', 'labels', or 'all'."
        )

    labels = []
    weights_list = []
    proj_list = []

    def _label_for(entry: Dict[str, object], comp_idx: int) -> str:
        """Compose a component label for window/label combos."""
        label = str(entry["label"])
        win = entry.get("window")
        if win is not None:
            label = f"{label}|{win[0]}:{win[1]}"
        return f"{label}|c{comp_idx}"

    if orthogonalize_across == "none":
        for entry in mode_entries:
            wts = np.asarray(entry["weights"])
            X_proj = np.asarray(entry["X_proj"])
            bias_val = entry.get("bias")
            if orthogonalize == "none" and bias_val is not None:
                bias = np.asarray(bias_val)
            else:
                bias = None
            for ci in range(wts.shape[0]):
                labels.append(_label_for(entry, ci))
                weights_list.append(wts[ci])
                if bias is not None:
                    proj_list.append(X_proj @ wts[ci] + bias[ci])
                else:
                    proj_list.append(X_proj @ wts[ci])
    else:
        groups: Dict[str, list[Tuple[Dict[str, object], int]]] = {}
        for entry in mode_entries:
            key = "all"
            if orthogonalize_across == "windows":
                key = str(entry.get("label"))
            elif orthogonalize_across == "labels":
                key = str(entry.get("window"))
            for ci in range(np.asarray(entry["weights"]).shape[0]):
                groups.setdefault(key, []).append((entry, ci))

        for _, items in groups.items():
            W = np.stack(
                [np.asarray(e["weights"])[ci] for e, ci in items], axis=0
            )
            W_ortho = _orthogonalize_weights(W, method=orthogonalize)
            for wi, (entry, ci) in enumerate(items):
                X_proj = np.asarray(entry["X_proj"])
                labels.append(_label_for(entry, ci))
                weights_list.append(W_ortho[wi])
                proj_list.append(X_proj @ W_ortho[wi])

    da_proj = mode_entries[0]["da_proj"]
    proj_mat = np.vstack(proj_list).T
    projections = _scores_to_dataarray(
        proj_mat,
        da_proj,  # type: ignore[arg-type]
        component_count=proj_mat.shape[1],
        unstack=unstack,
    )
    projections = preserve_coords(da, projections)
    projections = projections.assign_coords(
        component_label=("component", np.asarray(labels, dtype=object))
    )

    weights = _weights_to_dataarray(
        np.vstack(weights_list),
        da_stack_full,
        component_count=len(weights_list),
    )
    weights = weights.assign_coords(
        component_label=("component", np.asarray(labels, dtype=object))
    )

    return xr.Dataset({"projections": projections, "weights": weights})
