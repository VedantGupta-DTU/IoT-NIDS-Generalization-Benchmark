"""E4 data generators: per-class Gaussian-copula synthesizer and drift
perturbations.

The copula synthesizer (SDV/GaussianCopula-style, no extra dependencies)
captures each class's marginal distributions via empirical quantiles and the
inter-feature dependence via a Gaussian rank correlation. CTGAN is the heavier
literature-standard alternative; the copula gives a fast, reproducible first
pass with the same experimental role (synthetic test data / synthetic
training data).
"""
import numpy as np

from common import SEED


class ClassCopulaSynthesizer:
    def __init__(self, n_quantiles: int = 512, seed: int = SEED):
        self.n_quantiles = n_quantiles
        self.rng = np.random.default_rng(seed)
        self.per_class = {}

    def fit(self, X: np.ndarray, y: np.ndarray, max_per_class: int = 100_000):
        from scipy.stats import rankdata

        for c in np.unique(y):
            Xc = X[y == c]
            if len(Xc) > max_per_class:
                Xc = Xc[self.rng.choice(len(Xc), max_per_class, replace=False)]
            # marginals: empirical quantile grid per feature
            qs = np.quantile(Xc, np.linspace(0, 1, self.n_quantiles), axis=0)
            # dependence: Gaussian rank correlation (Spearman -> Pearson approx)
            # computed manually; constant features get zero correlation
            if len(Xc) > 2:
                ranks = rankdata(Xc, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rho = np.corrcoef(ranks, rowvar=False)
                rho = np.nan_to_num(rho, nan=0.0)
                rho = 2 * np.sin(np.pi / 6 * rho)  # rank -> linear corr
                np.fill_diagonal(rho, 1.0)
                # nearest PD fix
                w, V = np.linalg.eigh(rho)
                w = np.clip(w, 1e-6, None)
                rho = V @ np.diag(w) @ V.T
                d = np.sqrt(np.diag(rho))
                rho = rho / np.outer(d, d)
                L = np.linalg.cholesky(rho)
            else:
                L = np.eye(X.shape[1])
            self.per_class[int(c)] = (qs, L)
        return self

    def sample(self, n_per_class: dict) -> tuple:
        from scipy import stats

        Xs, ys = [], []
        grid = np.linspace(0, 1, self.n_quantiles)
        for c, n in n_per_class.items():
            qs, L = self.per_class[int(c)]
            z = self.rng.standard_normal((n, qs.shape[1])) @ L.T
            u = stats.norm.cdf(z)
            x = np.empty_like(u, dtype=np.float32)
            for j in range(qs.shape[1]):
                x[:, j] = np.interp(u[:, j], grid, qs[:, j])
            Xs.append(x)
            ys.append(np.full(n, c, dtype=np.int64))
        idx = self.rng.permutation(sum(len(a) for a in Xs))
        return np.concatenate(Xs)[idx], np.concatenate(ys)[idx]


def perturb_noise(X: np.ndarray, sigma: float, seed: int = SEED) -> np.ndarray:
    """Additive Gaussian noise on standardized features (distribution drift)."""
    rng = np.random.default_rng(seed)
    return (X + rng.normal(0, sigma, X.shape)).astype(np.float32)


def perturb_dropout(X: np.ndarray, frac: float, seed: int = SEED) -> np.ndarray:
    """Randomly zero a fraction of feature values (sensor/extractor failure)."""
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) < frac
    Xp = X.copy()
    Xp[mask] = 0.0
    return Xp


def perturb_scale_drift(X: np.ndarray, factor_sigma: float, seed: int = SEED) -> np.ndarray:
    """Per-feature multiplicative drift (e.g. rate features shift between
    networks): each feature scaled by lognormal factor."""
    rng = np.random.default_rng(seed)
    factors = rng.lognormal(0, factor_sigma, X.shape[1]).astype(np.float32)
    return (X * factors).astype(np.float32)
