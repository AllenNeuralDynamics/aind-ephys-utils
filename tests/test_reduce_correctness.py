"""Ground-truth correctness tests for dimensionality reduction methods."""

import unittest

import numpy as np
import xarray as xr

from aind_ephys_utils.ops import reduce


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return unit-norm vector."""
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector.")
    return v / n


def _abs_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute cosine similarity between two vectors."""
    a_n = _normalize(np.asarray(a, dtype=float))
    b_n = _normalize(np.asarray(b, dtype=float))
    return float(abs(np.dot(a_n, b_n)))


def _subspace_min_sv(
    est_components: np.ndarray, true_components: np.ndarray
) -> float:
    """Return minimum singular value between two row-subspaces."""
    q_est, _ = np.linalg.qr(np.asarray(est_components).T)
    q_true, _ = np.linalg.qr(np.asarray(true_components).T)
    s = np.linalg.svd(q_est.T @ q_true, compute_uv=False)
    return float(np.min(s))


class ReduceCorrectnessTest(unittest.TestCase):
    """Behavioral correctness tests using synthetic data with known structure."""

    @staticmethod
    def _make_data(
        seed: int = 0, n_trial: int = 24, n_time: int = 8, n_unit: int = 6
    ) -> xr.DataArray:
        """Create trial x unit x time data with choice/block labels."""
        rng = np.random.default_rng(seed)
        time = np.linspace(-0.3, 0.3, n_time)
        trials = np.arange(n_trial)

        choice = np.tile(np.array([0, 1]), n_trial // 2)
        block = np.tile(np.array([0, 0, 1, 1]), n_trial // 4)

        w_choice = _normalize(np.array([1.0, -0.8, 0.2, 0.0, 0.0, 0.0]))
        w_block = _normalize(np.array([0.0, 0.1, 0.0, 1.0, -0.6, 0.0]))
        w_time = _normalize(np.array([0.0, 0.0, 0.2, 0.1, 0.0, 1.0]))

        X = np.zeros((n_trial, n_time, n_unit), dtype=float)
        for ti in range(n_trial):
            c = 2 * choice[ti] - 1
            b = 2 * block[ti] - 1
            for ki, t in enumerate(time):
                X[ti, ki, :] = (
                    2.2 * c * w_choice
                    + 1.7 * b * w_block
                    + 1.5 * np.sin(np.pi * t / time.max()) * w_time
                    + 0.10 * rng.standard_normal(n_unit)
                )

        da = xr.DataArray(
            X.transpose(0, 2, 1),
            dims=("trial", "unit", "time"),
            coords={
                "trial": trials,
                "unit": np.arange(n_unit),
                "time": time,
                "choice": ("trial", choice),
                "block": ("trial", block),
            },
        )
        da.attrs["w_choice"] = w_choice
        da.attrs["w_block"] = w_block
        da.attrs["w_time"] = w_time
        return da

    def test_pca_recovers_low_rank_subspace(self) -> None:
        """PCA should recover the dominant 2D latent subspace."""
        da = self._make_data(seed=1)
        out = reduce(
            da,
            method="pca",
            dim="unit",
            n_components=2,
            stack=("trial", "time"),
        )

        w_est = out["weights"].values
        w_true = np.vstack([da.attrs["w_choice"], da.attrs["w_block"]])
        similarity = _subspace_min_sv(w_est, w_true)
        self.assertGreater(similarity, 0.85)
        self.assertGreater(float(out["explained_variance_ratio"].sum()), 0.75)

    def test_dpca_recovers_choice_and_time_marginals(self) -> None:
        """dPCA marginal weights should align with known factors."""
        da = self._make_data(seed=2)
        out = reduce(
            da,
            method="dpca",
            dim="unit",
            n_components=1,
            labels="choice",
            trial_average=True,
        )
        marginals = set(out["weights"]["marginal"].values.tolist())
        self.assertIn("choice", marginals)
        self.assertIn("time", marginals)

        choice_w = out["weights"].sel(marginal="choice", component=0).values
        time_w = out["weights"].sel(marginal="time", component=0).values
        self.assertGreater(_abs_cosine(choice_w, da.attrs["w_choice"]), 0.75)
        self.assertGreater(_abs_cosine(time_w, da.attrs["w_time"]), 0.70)

    def test_dpca_accepts_preaveraged_condition_data(self) -> None:
        """dPCA should run when trial-averaged condition dims are already present."""
        da = self._make_data(seed=8)
        da_pre = da.groupby("choice").mean("trial")
        out = reduce(
            da_pre,
            method="dpca",
            dim="unit",
            n_components=1,
            labels="choice",
            trial_average=True,
        )
        self.assertIn("projections", out)
        self.assertIn("weights", out)
        self.assertIn("choice", out["projections"].dims)
        marginals = set(out["weights"]["marginal"].values.tolist())
        self.assertIn("choice", marginals)
        self.assertIn("time", marginals)

    def test_coding_direction_recovers_separating_axis(self) -> None:
        """Coding-direction weight should align to known choice axis."""
        da = self._make_data(seed=3)
        out = reduce(
            da,
            method="coding_direction",
            dim="unit",
            labels="choice",
            stack=("trial", "time"),
        )
        w = out["weights"].isel(component=0).values
        self.assertGreater(_abs_cosine(w, da.attrs["w_choice"]), 0.90)

    def test_logistic_recovers_separating_axis(self) -> None:
        """Logistic weight should align to known choice axis."""
        da = self._make_data(seed=4)
        out = reduce(
            da,
            method="logistic",
            dim="unit",
            labels="choice",
            stack=("trial", "time"),
            regularization=0.1,
        )
        w = out["weights"].isel(component=0).values
        self.assertGreater(_abs_cosine(w, da.attrs["w_choice"]), 0.85)

    def test_lda_recovers_separating_axis(self) -> None:
        """LDA weight should align to known choice axis."""
        da = self._make_data(seed=5)
        out = reduce(
            da,
            method="lda",
            dim="unit",
            labels="choice",
            stack=("trial", "time"),
        )
        w = out["weights"].isel(component=0).values
        self.assertGreater(_abs_cosine(w, da.attrs["w_choice"]), 0.85)

    def test_rrr_recovers_predictor_subspace(self) -> None:
        """RRR should recover low-rank predictor directions used for targets."""
        da = self._make_data(seed=6)
        w_choice = da.attrs["w_choice"]
        w_block = da.attrs["w_block"]

        X = da.transpose("trial", "time", "unit").values.reshape(
            -1, da.sizes["unit"]
        )
        b_true = np.stack(
            [
                1.3 * w_choice + 0.2 * w_block,
                -0.4 * w_choice + 1.1 * w_block,
            ],
            axis=1,
        )  # unit x target
        y = X @ b_true
        targets = xr.DataArray(
            y.reshape(da.sizes["trial"], da.sizes["time"], 2),
            dims=("trial", "time", "target"),
            coords={
                "trial": da.coords["trial"],
                "time": da.coords["time"],
                "target": ["y0", "y1"],
            },
        )

        out = reduce(
            da,
            method="rrr",
            dim="unit",
            stack=("trial", "time"),
            targets=targets,
            rank=2,
        )
        w_est = out["weights"].values
        u_true, _, _ = np.linalg.svd(b_true, full_matrices=False)
        similarity = _subspace_min_sv(w_est, u_true[:, :2].T)
        self.assertGreater(similarity, 0.90)

    def test_supervised_qr_orthogonalize_across_labels(self) -> None:
        """QR orthogonalization across labels should produce orthonormal rows."""
        da = self._make_data(seed=7)
        out = reduce(
            da,
            method="coding_direction",
            dim="unit",
            labels=("choice", "block"),
            stack=("trial", "time"),
            orthogonalize="qr",
            orthogonalize_across="labels",
        )
        w = out["weights"].values  # component x unit
        gram = w @ w.T
        eye = np.eye(gram.shape[0])
        np.testing.assert_allclose(gram, eye, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
