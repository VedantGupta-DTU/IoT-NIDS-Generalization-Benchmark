"""Dataset loading and preprocessing.

CICIoT2023 comes from the canonical HF mirror `lacg030175/CIC-IoT-2023-neto-full`
(46,686,579 rows x 46 features, Neto et al. distribution). We build a stratified
subsample with a per-class cap so the 72.8%-DDoS majority does not drown
training, then standardize with a scaler fit on train only.

Future datasets (CICIoMT2024 native 45-feature CSVs, TabularIoTAttacks-2024
slices) plug in through load_external_csv(), which aligns columns by name with
the CICIoT2023 feature list and reports anything it has to drop.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA_DIR, PROCESSED_DIR, SEED, set_seed

CICIOT_TRAIN = os.path.join(DATA_DIR, "ciciot2023", "random", "train-00000-of-00001.parquet")
CICIOT_TEST = os.path.join(DATA_DIR, "ciciot2023", "random", "test-00000-of-00001.parquet")

LABEL_COLS = {"Label", "Label_orig", "attack_class", "label"}
CLASS_ORDER = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based"]


def _stratified_sample_parquet(path: str, per_class_cap: int, seed: int) -> pd.DataFrame:
    """Read a large parquet in batches, keeping at most per_class_cap rows per
    attack_class via per-batch bernoulli thinning calibrated on a first
    counting pass (single column, cheap)."""
    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(path)

    counts = {}
    for batch in pf.iter_batches(batch_size=2_000_000, columns=["attack_class"]):
        vc = pd.Series(batch.column(0).to_pylist()).value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)
    keep_p = {k: min(1.0, per_class_cap / v) for k, v in counts.items()}
    print(f"[data] {os.path.basename(path)} class counts: {counts}")
    print(f"[data] keep probabilities: { {k: round(p, 4) for k, p in keep_p.items()} }")

    parts = []
    for batch in pf.iter_batches(batch_size=1_000_000):
        df = batch.to_pandas()
        p = df["attack_class"].map(keep_p).to_numpy()
        mask = rng.random(len(df)) < p
        parts.append(df.loc[mask])
    out = pd.concat(parts, ignore_index=True)
    # thinning is approximate; trim overshoot exactly to the cap
    out = (
        out.groupby("attack_class", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_class_cap), random_state=seed), include_groups=True)
        .reset_index(drop=True)
    )
    return out


def _uniform_sample_parquet(path: str, total_cap: int, seed: int) -> pd.DataFrame:
    """Uniform thinning across all rows -> preserves the natural class mix."""
    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(path)
    p = min(1.0, total_cap / pf.metadata.num_rows)
    parts = []
    for batch in pf.iter_batches(batch_size=1_000_000):
        df = batch.to_pandas()
        parts.append(df.loc[rng.random(len(df)) < p])
    return pd.concat(parts, ignore_index=True)


def _clean(df: pd.DataFrame, feature_cols: list) -> tuple:
    X = df[feature_cols].to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    bad = np.isnan(X).any(axis=1)
    n_bad = int(bad.sum())
    return df.loc[~bad].reset_index(drop=True), X[~bad], n_bad


def prepare_ciciot2023(train_cap: int = 250_000, test_cap: int = 62_500, force: bool = False) -> dict:
    """Build processed npz splits for CICIoT2023. Returns paths + metadata."""
    out_train = os.path.join(PROCESSED_DIR, "ciciot2023_train.npz")
    out_test = os.path.join(PROCESSED_DIR, "ciciot2023_test.npz")
    out_nat = os.path.join(PROCESSED_DIR, "ciciot2023_test_natural.npz")
    meta_path = os.path.join(PROCESSED_DIR, "ciciot2023_meta.json")
    if not force and all(os.path.exists(p) for p in (out_train, out_test, out_nat, meta_path)):
        with open(meta_path) as f:
            return json.load(f)

    set_seed(SEED)
    tr = _stratified_sample_parquet(CICIOT_TRAIN, train_cap, SEED)
    te = _stratified_sample_parquet(CICIOT_TEST, test_cap, SEED + 1)
    tn = _uniform_sample_parquet(CICIOT_TEST, 500_000, SEED + 2)

    feature_cols = sorted(c for c in tr.columns if c not in LABEL_COLS)
    tr, Xtr, bad_tr = _clean(tr, feature_cols)
    te, Xte, bad_te = _clean(te, feature_cols)
    tn, Xtn, bad_tn = _clean(tn, feature_cols)
    print(f"[data] dropped NaN/inf rows: train={bad_tr} test={bad_te} natural={bad_tn}")

    classes = [c for c in CLASS_ORDER if c in set(tr["attack_class"])]
    cls_to_id = {c: i for i, c in enumerate(classes)}
    ytr8 = tr["attack_class"].map(cls_to_id).to_numpy(dtype=np.int64)
    yte8 = te["attack_class"].map(cls_to_id).to_numpy(dtype=np.int64)
    ytn8 = tn["attack_class"].map(cls_to_id).to_numpy(dtype=np.int64)
    ytr2 = (tr["attack_class"] != "Benign").to_numpy(dtype=np.int64)
    yte2 = (te["attack_class"] != "Benign").to_numpy(dtype=np.int64)
    ytn2 = (tn["attack_class"] != "Benign").to_numpy(dtype=np.int64)

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std == 0] = 1.0
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    Xtn = (Xtn - mean) / std

    np.savez_compressed(out_train, X=Xtr, y8=ytr8, y2=ytr2)
    np.savez_compressed(out_test, X=Xte, y8=yte8, y2=yte2)
    np.savez_compressed(out_nat, X=Xtn, y8=ytn8, y2=ytn2)
    meta = {
        "feature_cols": feature_cols,
        "classes": classes,
        "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "n_test_natural": int(len(Xtn)),
        "train_cap": train_cap,
        "test_cap": test_cap,
        "train_npz": out_train,
        "test_npz": out_test,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data] train={len(Xtr)} test={len(Xte)} features={len(feature_cols)} classes={classes}")
    return meta


def load_processed(split: str) -> tuple:
    d = np.load(os.path.join(PROCESSED_DIR, f"ciciot2023_{split}.npz"))
    return d["X"], d["y8"], d["y2"]


def load_meta() -> dict:
    with open(os.path.join(PROCESSED_DIR, "ciciot2023_meta.json")) as f:
        return json.load(f)


def load_external_csv(glob_pattern: str, label_col: str | None = None,
                      benign_values: set | None = None,
                      label_from_filename=None,
                      rename_map: dict | None = None, per_file_cap: int = 200_000) -> tuple:
    """Load an external dataset (e.g. CICIoMT2024 native CSVs) and align it to
    the CICIoT2023 feature space by column name. Returns (X_scaled, y_binary,
    dropped_features). Scaling uses the CICIoT2023 train scaler so the exact
    trained models can be applied unchanged.

    Labels come either from a column (`label_col` + `benign_values`) or, for
    datasets shipped as one-CSV-per-attack (CICIoMT2024), from the filename:
    `label_from_filename(basename) -> 0 (benign) | 1 (attack)`."""
    meta = load_meta()
    feature_cols = meta["feature_cols"]
    mean = np.array(meta["scaler_mean"], dtype=np.float32)
    std = np.array(meta["scaler_std"], dtype=np.float32)

    files = sorted(glob.glob(glob_pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"no files match {glob_pattern}")
    rng = np.random.default_rng(SEED)
    parts, labels = [], []
    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        if rename_map:
            df = df.rename(columns=rename_map)
        if len(df) > per_file_cap:
            df = df.iloc[rng.choice(len(df), per_file_cap, replace=False)]
        parts.append(df)
        if label_from_filename is not None:
            labels.append(np.full(len(df), label_from_filename(os.path.basename(fp)), dtype=np.int64))
    df = pd.concat(parts, ignore_index=True)

    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0  # absent feature -> 0 after imputation (reported to caller)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    keep = ~np.isnan(X).any(axis=1)
    X = X[keep]
    if label_from_filename is not None:
        y = np.concatenate(labels)[keep]
    else:
        y = (~df.loc[keep, label_col].isin(benign_values)).to_numpy(dtype=np.int64)
    X = (X - mean) / std
    return X, y, missing
