"""LightGBM isolated in a subprocess.

LightGBM's OpenMP fit/predict segfaults in this environment once PyTorch has
run Metal (MPS) ops in the same process (duplicate OpenMP runtimes). This
module keeps every LightGBM call in a clean child process: `SubprocessLGBM`
exposes sklearn-style fit/predict, shuttling arrays via npy files in a temp
dir. The module is also the child-process CLI and must not import torch.
"""
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

SEED = 42


def _fit(X_path, y_path, out_pkl):
    import lightgbm as lgb

    X = np.load(X_path)
    y = np.load(y_path)
    # lr=0.1 with 200 rounds degenerates on the full 1.5M-row training set
    # (acc 0.49); halving the step and doubling rounds is stable (acc 0.93).
    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63, n_jobs=8,
        random_state=SEED, class_weight="balanced", verbosity=-1,
    )
    clf.fit(X, y)
    with open(out_pkl, "wb") as f:
        pickle.dump(clf, f)


def _predict(pkl_path, X_path, out_path):
    with open(pkl_path, "rb") as f:
        clf = pickle.load(f)
    np.save(out_path, clf.predict(np.load(X_path)))


class SubprocessLGBM:
    """Duck-types the parts of the sklearn API the runners use."""

    def __init__(self, pkl_path: str):
        self.pkl_path = pkl_path

    @staticmethod
    def _run(args):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), *args],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"lgbm subprocess failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")

    @classmethod
    def fit_new(cls, X: np.ndarray, y: np.ndarray, pkl_path: str) -> "SubprocessLGBM":
        with tempfile.TemporaryDirectory() as td:
            xp, yp = os.path.join(td, "X.npy"), os.path.join(td, "y.npy")
            np.save(xp, X)
            np.save(yp, y)
            cls._run(["fit", xp, yp, pkl_path])
        return cls(pkl_path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            xp, op = os.path.join(td, "X.npy"), os.path.join(td, "out.npy")
            np.save(xp, X)
            self._run(["predict", self.pkl_path, xp, op])
            return np.load(op)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        _fit(*sys.argv[2:5])
    elif mode == "predict":
        _predict(*sys.argv[2:5])
    else:
        raise SystemExit(f"unknown mode {mode}")
