"""Demixed PCA implementation adapted from github.com/machenslab/dPCA

Author: Wieland Brendel <wieland.brendel@neuro.fchampalimaud.org>

License: BSD 3 clause

"""

from __future__ import annotations

from collections import OrderedDict
from itertools import chain, combinations

import numpy as np
from scipy.linalg import pinv
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import randomized_svd


def shuffle2D(X: np.ndarray) -> None:
    """In-place shuffle of rows where first column is not NaN."""
    idx = np.where(~np.isnan(X[:, 0]))[0]
    k = X.shape[1]
    t = len(idx)
    randints = np.random.rand(t) * np.arange(t)
    for i in range(t - 1, 0, -1):
        j = int(round(randints[i]))
        n, m = idx[i], idx[j]
        for col in range(k):
            X[n, col], X[m, col] = X[m, col], X[n, col]


def classification(class_means: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Simple nearest-class-mean classification per time point."""
    q = class_means.shape[0]
    t = class_means.shape[1]
    performance = np.zeros(t)
    for ti in range(t):
        for p in range(q):
            argmin = 0
            distance = abs(class_means[argmin, ti] - test[p, ti])
            for qi in range(1, q):
                if abs(class_means[qi, ti] - test[p, ti]) < distance:
                    distance = abs(class_means[qi, ti] - test[p, ti])
                    argmin = qi
            if argmin == p:
                performance[ti] += 1.0
        performance[ti] /= q
    return performance


def denoise_mask(mask: np.ndarray, n_consecutive: int) -> np.ndarray:
    """Remove short True segments from a 1D mask."""
    subseq = 0
    n = mask.shape[0]
    for i in range(n):
        if mask[i] == 1:
            subseq += 1
        else:
            if subseq < n_consecutive:
                for k in range(i - subseq, i):
                    mask[k] = 0
            subseq = 0
    return mask


class dPCA(BaseEstimator):
    """Demixed PCA (dPCA) implementation."""

    def __init__(
        self,
        labels=None,
        join=None,
        n_components=10,
        regularizer=None,
        copy=True,
        n_iter=0,
    ):
        """Initialize a dPCA model."""
        if isinstance(labels, str):
            self.labels = labels
        elif isinstance(labels, int):
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            self.labels = alphabet[:labels]
        else:
            raise TypeError(
                "labels must be int or string (e.g. 'ts' for time/stimulus)."
            )

        self._join = join
        self.join = join
        self.regularizer = 0 if regularizer is None else regularizer
        self.opt_regularizer_flag = regularizer == "auto"
        self.n_components = n_components
        self.copy = copy
        self.marginalizations = self._get_parameter_combinations()
        self.n_iter = n_iter
        self.debug = 0

    def fit(self, X, trialX=None):
        """Fit dPCA on data X (and optional trial-wise data)."""
        self._fit(X, trialX=trialX)
        return self

    def fit_transform(self, X, trialX=None):
        """Fit dPCA and return transformed projections."""
        self._fit(X, trialX=trialX)
        return self.transform(X)

    def _get_parameter_combinations(self, join=True):
        """Return mapping of marginalization keys to factor indices."""
        subsets = list(
            chain.from_iterable(
                combinations(list(range(len(self.labels))), r)
                for r in range(len(self.labels))
            )
        )
        del subsets[0]
        subsets.append(list(range(len(self.labels))))
        pcombs = OrderedDict()
        for subset in subsets:
            key = "".join([self.labels[i] for i in subset])
            pcombs[key] = set(subset)
        if isinstance(self._join, dict) and join:
            for key, combs in self._join.items():
                tmp = [pcombs[comb] for comb in combs]
                for comb in combs:
                    del pcombs[comb]
                pcombs[key] = tmp
        return pcombs

    def _marginalize(self, X, save_memory=False):  # noqa: C901
        """Compute ANOVA-style marginalizations of X."""

        def mmean(X, axes, expand=False):
            """Mean over axes, optionally keeping dimensions."""
            Z = X.copy()
            for ax in np.sort(axes)[::-1]:
                Z = np.mean(Z, ax)
                if expand:
                    Z = np.expand_dims(Z, ax)
            return Z

        def dense_marg(Y, mYs):
            """Expand marginals to match the full data shape."""
            tmp = np.zeros_like(Y)
            for key in list(mYs.keys()):
                mYs[key] = (tmp + mYs[key]).reshape((Y.shape[0], -1))
            return mYs

        Xres = X.copy()
        Xres -= np.mean(Xres.reshape((Xres.shape[0], -1)), -1).reshape(
            (Xres.shape[0],) + (len(Xres.shape) - 1) * (1,)
        )
        Xmargs = OrderedDict()
        pcombs = self._get_parameter_combinations(join=False)
        S = list(pcombs.values())[-1]

        if save_memory:
            for key, phi in pcombs.items():
                S_without_phi = list(S - phi)
                Xmargs[key] = mmean(
                    Xres, np.array(S_without_phi) + 1, expand=True
                )
                Xres -= Xmargs[key]
        else:
            pre_mean = {}
            for key, phi in pcombs.items():
                if len(key) == 1:
                    pre_mean[key] = mmean(
                        Xres, np.array(list(phi)) + 1, expand=True
                    )
                else:
                    pre_mean[key] = mmean(
                        pre_mean[key[:-1]],
                        np.array([list(phi)[-1]]) + 1,
                        expand=True,
                    )

            for key, phi in pcombs.items():
                key_without_phi = "".join(
                    filter(lambda ch: ch not in key, self.labels)
                )
                Xloc = (
                    pre_mean[key_without_phi]
                    if len(key_without_phi) > 0
                    else Xres
                )
                if len(key) > 1:
                    subsets = list(
                        chain.from_iterable(
                            combinations(key, r) for r in range(1, len(key))
                        )
                    )
                    subsets = ["".join(subset) for subset in subsets]
                    Xm = Xloc
                    for subset in subsets:
                        Xm = Xm - Xmargs[subset]
                    Xmargs[key] = Xm
                else:
                    Xmargs[key] = Xloc

        if isinstance(self._join, dict):
            for key, combs in self._join.items():
                Xshape = np.ones(len(self.labels) + 1, dtype="int")
                for comb in combs:
                    sh = np.array(Xmargs[comb].shape)
                    Xshape[(sh - 1).nonzero()] = sh[(sh - 1).nonzero()]
                tmp = np.zeros(Xshape)
                for comb in combs:
                    tmp += Xmargs[comb]
                    del Xmargs[comb]
                Xmargs[key] = tmp

        Xmargs = dense_marg(X, Xmargs)
        return Xmargs

    def _optimize_regularization(self, X, trialX, center=True, lams="auto"):
        """Select a regularization value via simple cross-validation."""
        if lams == "auto":
            lams = np.logspace(-7, 7, 30)
        elif isinstance(lams, (float, int)):
            lams = [lams]

        if trialX is None:
            raise ValueError(
                "To optimize the regularization parameter, trialX must be provided."
            )

        self.n_trials = 3
        self.protect = None

        n_trials = self.n_trials
        n_unq = trialX.shape[1]

        for _ in range(n_trials):
            shuffle2D(trialX)

        pcombs = self._get_parameter_combinations(join=False)
        keys = list(pcombs.keys())

        mean = np.mean(X, axis=-1).reshape(
            (X.shape[0],) + (len(X.shape) - 1) * (1,)
        )
        if center:
            X -= mean

        scores = np.zeros((self.n_trials, len(lams)))
        for j, lam in enumerate(lams):
            self.regularizer = lam
            for i in range(self.n_trials):
                trainX = X[..., : n_unq - 1]
                validX = X[..., n_unq - 1:]
                trainmXs, _ = (
                    self._marginalize(trainX),
                    self._marginalize(validX),
                )
                self._fit(trainX, mXs=trainmXs, optimize=False)
                trainZ = self.fit_transform(trainX)
                validZ = self.transform(validX)
                for key in keys:
                    for comp in range(trainZ[key].shape[0]):
                        scores[i, j] += np.sum(
                            (trainZ[key][comp] - validZ[key][comp]) ** 2
                        )

        score = scores.mean(axis=0)
        self.regularizer = lams[np.argmin(score)]

    def _randomized_dpca(self, X, mXs, pinvX=None):
        """Closed-form randomized dPCA solver for each marginalization."""
        n_features = X.shape[0]
        rX = X.reshape((n_features, -1))
        pinvX = pinv(rX) if pinvX is None else pinvX

        P, D = {}, {}

        for key in list(mXs.keys()):
            mX = mXs[key].reshape((n_features, -1))
            C = np.dot(mX, pinvX)

            if isinstance(self.n_components, dict):
                U, s, V = randomized_svd(
                    np.dot(C, rX),
                    n_components=self.n_components[key],
                    n_iter=self.n_iter,
                    random_state=np.random.randint(10e5),
                )
            else:
                U, s, V = randomized_svd(
                    np.dot(C, rX),
                    n_components=self.n_components,
                    n_iter=self.n_iter,
                    random_state=np.random.randint(10e5),
                )

            P[key] = U
            D[key] = np.dot(U.T, C).T

        return P, D

    def _add_regularization(self, Y, mYs, lam, SVD=None, pre_reg=False):
        """Prepare regularized data/marginals for the solver."""
        n_features = Y.shape[0]

        if not pre_reg:
            regY = np.hstack(
                [Y.reshape((n_features, -1)), lam * np.eye(n_features)]
            )
        else:
            regY = Y
            regY[:, -n_features:] = lam * np.eye(n_features)

        if not pre_reg:
            regmYs = OrderedDict()

            for key in list(mYs.keys()):
                regmYs[key] = np.hstack(
                    [mYs[key], np.zeros((n_features, n_features))]
                )
        else:
            regmYs = mYs

        if SVD is not None:
            U, s, V = SVD

            M = ((s**2 + lam**2) ** -1)[:, None] * U.T
            pregY = np.dot(np.vstack([V.T * s[None, :], lam * U]), M)
        else:
            pregY = np.dot(
                regY.reshape((n_features, -1)).T,
                np.linalg.inv(
                    np.dot(
                        Y.reshape((n_features, -1)),
                        Y.reshape((n_features, -1)).T,
                    )
                    + lam**2 * np.eye(n_features)
                ),
            )

        return regY, regmYs, pregY

    def _fit(
        self, X, trialX=None, mXs=None, center=True, SVD=None, optimize=True
    ):
        """Internal fit routine implementing the dPCA solution."""

        def flat2d(A):
            """Flatten all but the first axis."""
            return A.reshape((A.shape[0], -1))

        n_features = X.shape[0]

        if center:
            X = X - np.mean(flat2d(X), 1).reshape(
                (n_features,) + len(self.labels) * (1,)
            )

        if mXs is None:
            mXs = self._marginalize(X)

        if self.opt_regularizer_flag and optimize:
            if trialX is None:
                raise ValueError(
                    "To optimize regularization, trialX must be provided."
                )
            self._optimize_regularization(X, trialX)

        if self.regularizer > 0:
            regX, regmXs, pregX = self._add_regularization(
                X, mXs, self.regularizer * np.sum(X**2), SVD=SVD
            )
        else:
            regX, regmXs, pregX = X, mXs, pinv(X.reshape((n_features, -1)))

        self.P, self.D = self._randomized_dpca(regX, regmXs, pinvX=pregX)

    def transform(self, X, marginalization=None):
        """Project X onto dPCA components for each marginalization."""

        def marginal_variances(marginal):
            """Compute variance explained within a marginalization."""
            D, Xr = self.D[marginal], X.reshape((X.shape[0], -1))
            Z = np.dot(D.T, Xr).reshape((D.shape[1],) + X.shape[1:])
            return np.sum(Z.reshape((Z.shape[0], -1)) ** 2, axis=1) / np.sum(
                Xr**2
            )

        self.explained_variance_ratio_ = {}
        if marginalization is not None:
            D, Xr = self.D[marginalization], X.reshape((X.shape[0], -1))
            X_transformed = np.dot(D.T, Xr).reshape(
                (D.shape[1],) + X.shape[1:]
            )
            self.explained_variance_ratio_ = {
                marginalization: marginal_variances(marginalization)
            }
        else:
            X_transformed = {}
            for key in list(self.marginalizations.keys()):
                X_transformed[key] = np.dot(
                    self.D[key].T, X.reshape((X.shape[0], -1))
                ).reshape((self.D[key].shape[1],) + X.shape[1:])
                self.explained_variance_ratio_[key] = marginal_variances(key)
        return X_transformed

    def inverse_transform(self, X, marginalization):
        """Map dPCA components back to data space for one marginalization."""
        return np.dot(
            self.P[marginalization], X.reshape((X.shape[0], -1))
        ).reshape((self.P[marginalization].shape[0],) + X.shape[1:])

    def reconstruct(self, X, marginalization):
        """Reconstruct data using only one marginalization."""
        return self.inverse_transform(
            self.transform(X, marginalization), marginalization
        )
