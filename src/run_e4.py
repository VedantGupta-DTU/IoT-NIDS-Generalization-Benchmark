"""E4 — Synthetic & drifted data: does in-distribution performance survive
synthetic traffic and distribution drift?

Conditions (8-class task, models already trained in E1):
  a) TRTS: real-trained models tested on copula-synthetic test data
  b) drifted real test: Gaussian noise sigma in {0.05, 0.2}, feature dropout
     10%, per-feature scale drift
  c) TSTR: models trained on synthetic data only, tested on real test split
     (classic synthetic-data quality measure)
"""
import numpy as np
import torch

from common import SEED, compute_metrics, print_table, save_results, set_seed
from data import load_meta, load_processed
from models import build, param_count
from run_e1 import NN_MODELS, train_all
from synthetic import ClassCopulaSynthesizer, perturb_dropout, perturb_noise, perturb_scale_drift
from train import predict, train_model


def eval_models(models: dict, X, y, n_classes: int) -> dict:
    out = {}
    for name, m in models.items():
        yp = predict(m, X) if isinstance(m, torch.nn.Module) else m.predict(X)
        out[name] = compute_metrics(y, yp, n_classes)
    return out


def main():
    set_seed(SEED)
    meta = load_meta()
    Xtr, ytr8, _ = load_processed("train")
    Xte, yte8, _ = load_processed("test")
    n8 = len(meta["classes"])

    models = train_all("8class", Xtr, ytr8, n8)  # loads checkpoints

    results = {"classes": meta["classes"]}

    print("[e4] fitting copula synthesizer on train")
    synth = ClassCopulaSynthesizer().fit(Xtr, ytr8, max_per_class=60_000)
    test_counts = {int(c): int((yte8 == c).sum()) for c in np.unique(yte8)}
    Xsyn_te, ysyn_te = synth.sample(test_counts)
    print("[e4] (a) real-trained models on synthetic test")
    results["a_trts_synthetic_test"] = eval_models(models, Xsyn_te, ysyn_te, n8)

    print("[e4] (b) drifted real test sets")
    drift = {
        "noise_0.05": perturb_noise(Xte, 0.05),
        "noise_0.2": perturb_noise(Xte, 0.2),
        "dropout_10pct": perturb_dropout(Xte, 0.10),
        "scale_drift_0.2": perturb_scale_drift(Xte, 0.2),
    }
    results["b_drifted_real_test"] = {
        cond: eval_models(models, Xd, yte8, n8) for cond, Xd in drift.items()
    }

    print("[e4] (c) TSTR: train on synthetic, test on real")
    train_counts = {int(c): min(int((ytr8 == c).sum()), 100_000) for c in np.unique(ytr8)}
    Xsyn_tr, ysyn_tr = synth.sample(train_counts)
    tstr = {}
    for name in ["MLP14k", "TinyMLP"]:
        m = build(name, Xtr.shape[1], n8)
        print(f"  TSTR training {name} ({param_count(m):,} params)")
        train_model(m, Xsyn_tr, ysyn_tr, n8, epochs=3, verbose=False)
        yp = predict(m, Xte)
        tstr[name] = compute_metrics(yte8, yp, n8)
    results["c_tstr_real_test"] = tstr

    save_results("e4_synthetic_drift", results)

    rows = []
    base = {n: r for n, r in eval_models(models, Xte, yte8, n8).items()}
    for name in NN_MODELS + ["RandomForest", "LightGBM"]:
        rows.append([
            name,
            f"{base[name]['f1_macro']:.4f}",
            f"{results['a_trts_synthetic_test'][name]['f1_macro']:.4f}",
            f"{results['b_drifted_real_test']['noise_0.2'][name]['f1_macro']:.4f}",
            f"{results['b_drifted_real_test']['dropout_10pct'][name]['f1_macro']:.4f}",
        ])
    print_table(rows, ["model", "realF1", "synF1", "noise.2F1", "drop10F1"])


if __name__ == "__main__":
    main()
