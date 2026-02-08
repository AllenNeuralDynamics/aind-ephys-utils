"""Gaussian Process Factor Analysis (GPFA) helpers.

This implementation is adapted from the Elephant library GPFA code:
https://github.com/NeuralEnsemble/elephant

The adaptation here is intentionally minimal for aind-ephys-utils:
- Input is pre-binned ``xarray.DataArray`` data (no spike binning step).
- Public API mirrors ``reduce(method="pca")`` style usage.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize
import scipy.sparse as sparse
import xarray as xr
from sklearn.decomposition import FactorAnalysis

from .utils import preserve_coords


def gpfa(
    da: xr.DataArray,
    *,
    n_components: int,
    dim: str,
    trial_dim: str,
    time_dim: str,
    gpfa_options: Optional[Dict[str, object]] = None,
) -> xr.Dataset:
    """Run GPFA on pre-binned ``DataArray`` and return projections/weights."""
    if trial_dim not in da.dims:
        raise ValueError(
            f"trial_dim {trial_dim!r} not found in DataArray dims."
        )
    if time_dim not in da.dims:
        raise ValueError(f"time_dim {time_dim!r} not found in DataArray dims.")
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not found in DataArray dims.")
    if da.dtype == object:
        raise ValueError(
            "GPFA expects numeric pre-binned data. "
            "Bin ragged spikes first (e.g. da.ephys.bin(...))."
        )
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")

    y, unit_coords = _prepare_binned_counts(
        da, dim=dim, trial_dim=trial_dim, time_dim=time_dim
    )
    seqs = _as_seqs_from_y(y)
    params_est, _fit_info = _fit_gpfa(
        seqs=seqs,
        x_dim=n_components,
        gpfa_options=gpfa_options,
    )
    seqs_latent, _ll = _exact_inference_with_ll(seqs, params_est, get_ll=True)
    corth, seqs_latent = _orthonormalize(params_est, seqs_latent)

    proj = np.stack(
        [
            seqs_latent[i]["latent_variable_orth"]
            for i in range(len(seqs_latent))
        ],
        axis=1,
    )
    projections = xr.DataArray(
        proj,
        dims=("component", trial_dim, time_dim),
        coords={
            "component": np.arange(proj.shape[0]),
            trial_dim: da.coords[trial_dim],
            time_dim: da.coords[time_dim],
        },
        name=da.name,
        attrs=dict(da.attrs),
    )
    projections = preserve_coords(da, projections)

    weights = xr.DataArray(
        corth.T,
        dims=("component", dim),
        coords={"component": np.arange(corth.shape[1]), dim: unit_coords},
        name="weights",
    )
    return xr.Dataset({"projections": projections, "weights": weights})


def _prepare_binned_counts(
    da: xr.DataArray, *, dim: str, trial_dim: str, time_dim: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare binned count-like data to match Elephant GPFA preprocessing."""
    da_use = da.transpose(trial_dim, dim, time_dim)
    y = np.asarray(da_use.data, dtype=float)
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=0.0)
    if (y < 0).any():
        raise ValueError(
            "GPFA expects non-negative binned count-like data. "
            "Negative values are not supported."
        )

    # Elephant GPFA operates on spike counts.
    # If inputs are not count-like, assume they are rates and convert by dt.
    frac = np.abs(y - np.round(y))
    is_count_like = float(np.nanmedian(frac)) < 1e-6
    if (
        (not is_count_like)
        and time_dim in da_use.coords
        and da_use.sizes.get(time_dim, 0) > 1
    ):
        t = np.asarray(da_use.coords[time_dim].values, dtype=float)
        dt = float(np.median(np.diff(t)))
        if np.isfinite(dt) and dt > 0:
            y = y * dt

    y = np.sqrt(y)

    has_spikes = y.any(axis=(0, 2))
    if not has_spikes.any():
        raise ValueError("No active units found for GPFA.")
    y = y[:, has_spikes, :]
    unit_coords = np.asarray(da.coords[dim].values)[has_spikes]
    return y, unit_coords


def _as_seqs_from_y(y: np.ndarray) -> np.ndarray:
    """Convert 3D trial x unit x time array into Elephant-style recarray."""
    n_trials = y.shape[0]
    n_time = y.shape[2]
    seqs = np.empty(n_trials, dtype=[("T", int), ("y", object)])
    seqs["T"] = n_time
    for i in range(n_trials):
        seqs[i]["y"] = y[i]
    return seqs


def _fit_gpfa(  # noqa C901
    seqs: np.ndarray,
    *,
    x_dim: int,
    gpfa_options: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Fit GPFA parameters with configurable options."""
    opts = dict(gpfa_options or {})
    allowed = {
        "max_iters",
        "freq_ll",
        "em_tol",
        "min_var_frac",
        "tau_init",
        "eps_init",
        "seg_length",
        "learn_kernel_params",
        "learn_gp_noise",
        "r_force_diagonal",
        "fast_mode",
        "gp_param_update_every",
    }
    unknown = set(opts).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown gpfa_options keys: {sorted(unknown)}")

    max_iters = int(opts.get("max_iters", 200))
    freq_ll = int(opts.get("freq_ll", 5))
    em_tol = float(opts.get("em_tol", 1.0e-8))
    min_var_frac = float(opts.get("min_var_frac", 0.01))
    tau_init = float(opts.get("tau_init", 5.0))
    eps_init = float(opts.get("eps_init", 1.0e-3))
    seg_length = opts.get("seg_length", 20)
    learn_kernel_params = bool(opts.get("learn_kernel_params", True))
    learn_gp_noise = bool(opts.get("learn_gp_noise", False))
    r_force_diagonal = bool(opts.get("r_force_diagonal", True))
    fast_mode = bool(opts.get("fast_mode", False))
    gp_param_update_every = opts.get("gp_param_update_every", None)
    if gp_param_update_every is None:
        gp_param_update_every = 5 if fast_mode else 1
    gp_param_update_every = int(gp_param_update_every)

    if max_iters < 1:
        raise ValueError("gpfa_options['max_iters'] must be >= 1.")
    if freq_ll < 1:
        raise ValueError("gpfa_options['freq_ll'] must be >= 1.")
    if em_tol <= 0:
        raise ValueError("gpfa_options['em_tol'] must be > 0.")
    if min_var_frac < 0:
        raise ValueError("gpfa_options['min_var_frac'] must be >= 0.")
    if tau_init <= 0:
        raise ValueError("gpfa_options['tau_init'] must be > 0.")
    if eps_init < 0:
        raise ValueError("gpfa_options['eps_init'] must be >= 0.")
    if gp_param_update_every < 1:
        raise ValueError("gpfa_options['gp_param_update_every'] must be >= 1.")

    seqs_train = _cut_trials(seqs, seg_length=float(seg_length))
    if len(seqs_train) == 0:
        seqs_train = _cut_trials(seqs, seg_length=np.inf)

    y_all = np.hstack(seqs_train["y"])
    fa = FactorAnalysis(
        n_components=x_dim,
        copy=True,
        noise_variance_init=np.diag(np.cov(y_all, bias=True)),
    )
    fa.fit(y_all.T)

    params_init: Dict[str, np.ndarray] = {}
    params_init["covType"] = "rbf"
    params_init["gamma"] = (1.0 / tau_init) ** 2 * np.ones(x_dim)
    params_init["eps"] = eps_init * np.ones(x_dim)
    params_init["d"] = y_all.mean(axis=1)
    params_init["C"] = fa.components_.T
    params_init["R"] = np.diag(fa.noise_variance_)
    params_init["notes"] = {
        "learnKernelParams": learn_kernel_params,
        "learnGPNoise": learn_gp_noise,
        "RforceDiagonal": r_force_diagonal,
    }

    params_est, _seqs_latent, ll, iter_time = _em(
        params_init=params_init,
        seqs=seqs_train,
        max_iters=max_iters,
        tol=em_tol,
        min_var_frac=min_var_frac,
        freq_ll=freq_ll,
        gp_param_update_every=gp_param_update_every,
    )
    fit_info = {"iteration_time": iter_time, "log_likelihoods": ll}
    return params_est, fit_info


def _em(
    params_init: Dict[str, np.ndarray],
    seqs: np.ndarray,
    *,
    max_iters: int,
    tol: float,
    min_var_frac: float,
    freq_ll: int,
    gp_param_update_every: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[float], List[float]]:
    """EM loop for GPFA parameter fitting."""
    params = params_init
    if params["notes"].get("learnGPNoise", False):
        raise ValueError(
            "gpfa_options['learn_gp_noise']=True is not supported."
        )
    t = seqs["T"]
    y_dim, x_dim = params["C"].shape
    lls: List[float] = []
    ll_old = ll_base = ll = 0.0
    iter_time: List[float] = []
    var_floor = min_var_frac * np.diag(np.cov(np.hstack(seqs["y"])))
    seqs_latent = seqs

    for iter_id in range(1, max_iters + 1):
        tic = time.time()
        get_ll = (np.fmod(iter_id, freq_ll) == 0) or (iter_id <= 2)
        if not np.isnan(ll):
            ll_old = ll
        seqs_latent, ll = _exact_inference_with_ll(seqs, params, get_ll=get_ll)
        lls.append(float(ll))

        sum_p_auto = np.zeros((x_dim, x_dim))
        for seq_latent in seqs_latent:
            sum_p_auto += seq_latent["Vsm"].sum(axis=2) + seq_latent[
                "latent_variable"
            ].dot(seq_latent["latent_variable"].T)
        y = np.hstack(seqs["y"])
        latent_variable = np.hstack(seqs_latent["latent_variable"])
        sum_yx = y.dot(latent_variable.T)
        sum_x = latent_variable.sum(axis=1)[:, np.newaxis]
        sum_y = y.sum(axis=1)[:, np.newaxis]

        term = np.vstack(
            [
                np.hstack([sum_p_auto, sum_x]),
                np.hstack([sum_x.T, t.sum().reshape((1, 1))]),
            ]
        )
        cd = _rdiv(np.hstack([sum_yx, sum_y]), term)
        params["C"] = cd[:, :x_dim]
        params["d"] = cd[:, -1]

        c = params["C"]
        d = params["d"][:, np.newaxis]
        if params["notes"]["RforceDiagonal"]:
            sum_yy = (y * y).sum(axis=1)[:, np.newaxis]
            yd = sum_y * d
            term = ((sum_yx - d.dot(sum_x.T)) * c).sum(axis=1)[:, np.newaxis]
            r = d**2 + (sum_yy - 2 * yd - term) / t.sum()
            r = np.maximum(var_floor, r)
            params["R"] = np.diag(r[:, 0])
        else:
            sum_yy = y.dot(y.T)
            yd = sum_y.dot(d.T)
            term = (sum_yx - d.dot(sum_x.T)).dot(c.T)
            r = d.dot(d.T) + (sum_yy - yd - yd.T - term) / t.sum()
            params["R"] = (r + r.T) / 2

        if params["notes"]["learnKernelParams"] and (
            np.fmod(iter_id, gp_param_update_every) == 0
        ):
            params["gamma"] = _learn_gp_params(seqs_latent, params)["gamma"]

        iter_time.append(time.time() - tic)
        if iter_id <= 2:
            ll_base = ll
        elif (ll - ll_base) < (1 + tol) * (ll_old - ll_base):
            break

    return params, seqs_latent, lls, iter_time


def _exact_inference_with_ll(
    seqs: np.ndarray,
    params: Dict[str, np.ndarray],
    *,
    get_ll: bool,
) -> Tuple[np.ndarray, float]:
    """Infer latent trajectories given parameters."""
    y_dim, x_dim = params["C"].shape

    dtype_out = [(x, seqs[x].dtype) for x in seqs.dtype.names]
    dtype_out.extend(
        [("latent_variable", object), ("Vsm", object), ("VsmGP", object)]
    )
    seqs_latent = np.empty(len(seqs), dtype=dtype_out)
    for name in seqs.dtype.names:
        seqs_latent[name] = seqs[name]

    rinv = np.diag(1.0 / np.diag(params["R"]))
    logdet_r = np.log(np.diag(params["R"])).sum()
    c_rinv = params["C"].T.dot(rinv)
    c_rinv_c = c_rinv.dot(params["C"])
    t_all = seqs_latent["T"]
    t_uniq = np.unique(t_all)
    ll = 0.0

    for t in t_uniq:
        k_big, k_big_inv, logdet_k_big = _make_k_big(params, int(t))
        k_big = sparse.csr_matrix(k_big)
        c_rinv_c_big = linalg.block_diag(*[c_rinv_c for _ in range(int(t))])
        minv, logdet_m = _inv_persymm(k_big_inv + c_rinv_c_big, x_dim)

        vsm = np.full((x_dim, x_dim, int(t)), np.nan)
        idx = np.arange(0, x_dim * int(t) + 1, x_dim)
        for i in range(int(t)):
            vsm[:, :, i] = minv[idx[i]: idx[i + 1], idx[i]: idx[i + 1]]  # fmt: skip
        vsm_gp = np.full((int(t), int(t), x_dim), np.nan)
        for i in range(x_dim):
            vsm_gp[:, :, i] = minv[i::x_dim, i::x_dim]

        n_list = np.where(t_all == t)[0]
        dif = np.hstack(seqs_latent[n_list]["y"]) - params["d"][:, np.newaxis]
        term1 = c_rinv.dot(dif).reshape((x_dim * int(t), -1), order="F")

        t_half = int(np.ceil(t / 2.0))
        blk_prod = np.zeros((x_dim * t_half, x_dim * int(t)))
        idxh = range(0, x_dim * t_half + 1, x_dim)
        for i in range(t_half):
            blk_prod[idxh[i]: idxh[i + 1], :] = c_rinv_c.dot(
                minv[idxh[i]: idxh[i + 1], :]
            )  # fmt: skip
        eye_top = np.eye(x_dim * t_half, x_dim * int(t))
        blk_prod = k_big[: x_dim * t_half, :].dot(
            _fill_persymm(eye_top - blk_prod, x_dim, int(t))
        )
        latent_mat = _fill_persymm(blk_prod, x_dim, int(t)).dot(term1)

        for i, n in enumerate(n_list):
            seqs_latent[n]["latent_variable"] = latent_mat[:, i].reshape(
                (x_dim, int(t)), order="F"
            )
            seqs_latent[n]["Vsm"] = vsm
            seqs_latent[n]["VsmGP"] = vsm_gp

        if get_ll:
            val = (
                -t * logdet_r
                - logdet_k_big
                - logdet_m
                - y_dim * t * np.log(2 * np.pi)
            )
            ll = ll + len(n_list) * val - (rinv.dot(dif) * dif).sum()
            ll = ll + (term1.T.dot(minv) * term1.T).sum()

    if get_ll:
        ll /= 2.0
    else:
        ll = np.nan
    return seqs_latent, float(ll)


def _learn_gp_params(
    seqs_latent: np.ndarray, params: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Update GP kernel parameters."""
    x_dim = params["C"].shape[1]
    gamma_opt = np.empty_like(params["gamma"])
    precomp = _make_precomp(seqs_latent, x_dim)
    eps = np.asarray(params["eps"], dtype=float)
    gamma = np.asarray(params["gamma"], dtype=float)
    for i in range(x_dim):
        initp = np.log(gamma[i])
        res = optimize.fmin_l_bfgs_b(
            func=_grad_betgam,
            x0=np.array([initp], dtype=float),
            args=(precomp[i], float(eps[i])),
            approx_grad=False,
        )
        gamma_opt[i] = np.exp(res[0].item())
    param_opt = {"gamma": gamma_opt}
    return param_opt


def _orthonormalize(
    params_est: Dict[str, np.ndarray], seqs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormalize C and transform latent trajectories."""
    c = params_est["C"]
    x = np.hstack(seqs["latent_variable"])
    x_dim = c.shape[1]
    if x_dim == 1:
        tt = np.sqrt(np.dot(c.T, c))
        corth = _rdiv(c, tt)
        x_orth = np.dot(tt, x)
    else:
        u, s, vh = linalg.svd(c, full_matrices=False)
        tt = np.dot(np.diag(s), vh)
        corth = u
        x_orth = np.dot(tt, x)
    seqs_new = _segment_by_trial(seqs, x_orth, "latent_variable_orth")
    return corth, seqs_new


def _segment_by_trial(seqs: np.ndarray, x: np.ndarray, fn: str) -> np.ndarray:
    """Store concatenated latents back into trial-wise object arrays."""
    if np.sum(seqs["T"]) != x.shape[1]:
        raise ValueError("size of X incorrect.")
    dtype_new = [(i, seqs[i].dtype) for i in seqs.dtype.names] + [(fn, object)]
    seqs_new = np.empty(len(seqs), dtype=dtype_new)
    for name in seqs.dtype.names:
        seqs_new[name] = seqs[name]
    ctr = 0
    for n, t in enumerate(seqs["T"]):
        seqs_new[n][fn] = x[:, ctr: ctr + t]  # fmt: skip
        ctr += t
    return seqs_new


def _rdiv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve right matrix division ``x @ b = a`` (MATLAB ``a / b``)."""
    return np.linalg.solve(b.T, a.T).T


def _logdet(a: np.ndarray) -> float:
    """Compute ``log(det(a))`` for positive-definite matrices via Cholesky."""
    u = np.linalg.cholesky(a)
    return float(2 * np.log(np.diag(u)).sum())


def _make_k_big(
    params: Dict[str, np.ndarray], n_timesteps: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Construct GP covariance matrix."""
    x_dim = params["C"].shape[1]
    k_big = np.zeros((x_dim * n_timesteps, x_dim * n_timesteps))
    k_big_inv = np.zeros_like(k_big)
    tdif = np.subtract.outer(np.arange(n_timesteps), np.arange(n_timesteps))
    logdet_k_big = 0.0
    for i in range(x_dim):
        k = (1 - params["eps"][i]) * np.exp(-params["gamma"][i] / 2 * tdif**2)
        k = k + params["eps"][i] * np.eye(n_timesteps)
        k_big[i::x_dim, i::x_dim] = k
        k_big_inv[i::x_dim, i::x_dim] = np.linalg.inv(k)
        logdet_k_big += _logdet(k)
    return k_big, k_big_inv, float(logdet_k_big)


def _inv_persymm(m: np.ndarray, blk_size: int) -> Tuple[np.ndarray, float]:
    """Invert a block-persymmetric matrix."""
    t = int(m.shape[0] / blk_size)
    t_half = int(np.ceil(t / 2.0))
    mkr = blk_size * t_half
    inv_a11 = np.linalg.inv(m[:mkr, :mkr])
    inv_a11 = (inv_a11 + inv_a11.T) / 2
    a12 = m[:mkr, mkr:]
    term = inv_a11.dot(a12)
    f22 = m[mkr:, mkr:] - a12.T.dot(term)
    res12 = _rdiv(-term, f22)
    res11 = inv_a11 - res12.dot(term.T)
    res11 = (res11 + res11.T) / 2
    inv_m = _fill_persymm(np.hstack([res11, res12]), blk_size, t)
    logdet_m = -_logdet(inv_a11) + _logdet(f22)
    return inv_m, float(logdet_m)


def _fill_persymm(
    p_in: np.ndarray,
    blk_size: int,
    n_blocks: int,
    blk_size_vert: Optional[int] = None,
) -> np.ndarray:
    """Fill the full block-persymmetric matrix from its top half."""
    if blk_size_vert is None:
        blk_size_vert = blk_size
    nh = blk_size * n_blocks
    nv = blk_size_vert * n_blocks
    t_half = int(np.floor(n_blocks / 2.0))
    t_half_up = int(np.ceil(n_blocks / 2.0))

    p_out = np.empty((blk_size_vert * n_blocks, blk_size * n_blocks))
    p_out[: blk_size_vert * t_half_up, :] = p_in
    for i in range(t_half):
        for j in range(n_blocks):
            p_out[
                nv - (i + 1) * blk_size_vert: nv - i * blk_size_vert,  # fmt: skip
                nh - (j + 1) * blk_size: nh - j * blk_size,  # fmt: skip
            ] = p_in[
                i * blk_size_vert: (i + 1) * blk_size_vert,  # fmt: skip
                j * blk_size: (j + 1) * blk_size,  # fmt: skip
            ]
    return p_out


def _make_precomp(seqs: np.ndarray, x_dim: int) -> np.ndarray:
    """Precompute trial-length-grouped sufficient stats for GP timescale updates.

    Parameters
    ----------
    seqs : np.ndarray
        Elephant-style sequence records containing ``T``, ``latent_variable``,
        and ``VsmGP`` from the E-step posteriors.
    x_dim : int
        Number of latent dimensions.

    Returns
    -------
    np.ndarray
        Structured array (length ``x_dim``) with reusable terms for the
        M-step GP kernel parameter objective/gradient, grouped by unique trial
        length to avoid recomputing per-trial matrix products.
    """
    tall = seqs["T"]
    tmax = tall.max()
    tdif = np.subtract.outer(np.arange(tmax), np.arange(tmax))
    abs_dif = np.abs(tdif)
    dif_sq = tdif**2
    tu = np.unique(tall)

    precomp = np.empty(
        x_dim,
        dtype=[
            ("absDif", object),
            ("difSq", object),
            ("Tall", object),
            ("Tu", object),
        ],
    )

    for i in range(x_dim):
        precomp[i]["absDif"] = abs_dif
        precomp[i]["difSq"] = dif_sq
        precomp[i]["Tall"] = tall
        precomp[i]["Tu"] = np.empty(
            len(tu),
            dtype=[
                ("nList", object),
                ("T", int),
                ("numTrials", int),
                ("PautoSUM", object),
            ],
        )

    grouped: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for j, t in enumerate(tu):
        n_list = np.where(tall == t)[0]
        latent_block = np.stack(
            [np.asarray(seqs[n]["latent_variable"]) for n in n_list], axis=0
        )  # nTrial x xDim x T
        vsmgp_block = np.stack(
            [np.asarray(seqs[n]["VsmGP"]) for n in n_list], axis=0
        )  # nTrial x T x T x xDim
        grouped[int(t)] = (n_list, latent_block, vsmgp_block)

        for i in range(x_dim):
            precomp[i]["Tu"][j]["nList"] = n_list
            precomp[i]["Tu"][j]["T"] = t
            precomp[i]["Tu"][j]["numTrials"] = len(n_list)

    for i in range(x_dim):
        for j, t in enumerate(tu):
            n_list, latent_block, vsmgp_block = grouped[int(t)]
            latent_i = latent_block[:, i, :]  # nTrial x T
            pauto = vsmgp_block[:, :, :, i].sum(axis=0) + latent_i.T.dot(
                latent_i
            )
            precomp[i]["Tu"][j]["PautoSUM"] = pauto

    return precomp


def _cut_trials(seq_in: np.ndarray, seg_length: float = 20) -> np.ndarray:
    """Extract equal-length trial segments, matching Elephant's approach."""
    if seg_length == 0:
        raise ValueError("At least 1 extracted trial must be returned.")
    if np.isinf(seg_length):
        return seq_in
    seg_len = int(seg_length)
    if seg_len <= 0:
        raise ValueError("seg_length must be positive.")

    dtype_seq_out = [("segId", int), ("T", int), ("y", object)]
    seq_out_buff: List[np.ndarray] = []
    for n, seq_n in enumerate(seq_in):
        t = seq_n["T"]
        if t < seg_len:
            continue

        num_seg = int(np.ceil(float(t) / seg_len))
        if num_seg == 1:
            cum_ol = np.array([0])
        else:
            total_ol = (seg_len * num_seg) - t
            probs = np.ones(num_seg - 1, float) / (num_seg - 1)
            rand_ol = np.random.multinomial(total_ol, probs)
            cum_ol = np.hstack([0, np.cumsum(rand_ol)])

        seg = np.empty(num_seg, dtype=dtype_seq_out)
        seg["segId"] = n
        seg["T"] = seg_len
        for s in range(num_seg):
            t_start = seg_len * s - cum_ol[s]
            seg[s]["y"] = seq_n["y"][:, t_start: t_start + seg_len]  # fmt: skip
        seq_out_buff.append(seg)

    if len(seq_out_buff) > 0:
        return np.hstack(seq_out_buff)
    return np.empty(0, dtype=dtype_seq_out)


def _grad_betgam(
    p: Union[np.ndarray, float], pre_comp: np.ndarray, eps: float
) -> Tuple[float, np.ndarray]:
    """Evaluate negative log-likelihood and gradient for one GP timescale.

    Parameters
    ----------
    p : np.ndarray | float
        Log-timescale parameter (``log(gamma)`` in Elephant notation).
    pre_comp : np.ndarray
        One latent dimension entry returned by :func:`_make_precomp`.
    eps : float
        GP observation noise floor term used in the RBF kernel.

    Returns
    -------
    tuple[float, np.ndarray]
        Objective value and 1-element gradient array, formatted for
        ``scipy.optimize.fmin_l_bfgs_b``.
    """
    p_val = float(np.asarray(p).item())
    tall = pre_comp["Tall"]
    tmax = tall.max()
    temp = (1 - eps) * np.exp(-np.exp(p_val) / 2 * pre_comp["difSq"])
    kmax = temp + eps * np.eye(tmax)
    dkdg = -0.5 * temp * pre_comp["difSq"]

    dedg = 0.0
    f = 0.0
    for j in range(len(pre_comp["Tu"])):
        t = pre_comp["Tu"][j]["T"]
        t_half = int(np.ceil(t / 2.0))
        kinv = np.linalg.inv(kmax[:t, :t])
        logdet_k = _logdet(kmax[:t, :t])
        kinv_m = kinv[:t_half, :].dot(dkdg[:t, :t])
        kinv_m_kinv = (kinv_m.dot(kinv)).T

        dg_kinv = np.diag(kinv_m)
        tr_kinv = 2 * dg_kinv.sum() - np.fmod(t, 2) * dg_kinv[-1]
        mkr = int(np.ceil(0.5 * t**2))
        n_trials = pre_comp["Tu"][j]["numTrials"]
        pauto = pre_comp["Tu"][j]["PautoSUM"]
        dot1 = pauto.ravel("F")[:mkr].dot(kinv_m_kinv.ravel("F")[:mkr])
        dot2 = pauto.ravel("F")[-1: mkr - 1: -1].dot(
            kinv_m_kinv.ravel("F")[: (t**2 - mkr)]
        )  # fmt: skip
        dedg = dedg - 0.5 * n_trials * tr_kinv + 0.5 * dot1 + 0.5 * dot2
        f = f - 0.5 * n_trials * logdet_k - 0.5 * (pauto * kinv).sum()

    f = -f
    df = -dedg * np.exp(p_val)
    return float(np.asarray(f).item()), np.array(
        [float(np.asarray(df).item())]
    )
