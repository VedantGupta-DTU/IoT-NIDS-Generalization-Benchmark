"""E2 ablation — fair cross-dataset comparison on the 44 shared features.

CICIoMT2024 lacks `flow_duration` and `urg_count`, which rank 4th and 2nd in
the CICIoT2023 RandomForest's feature importances; zero-filling them (main E2)
may unfairly handicap models that rely on them. Here every model is retrained
from scratch on the 44-feature intersection and evaluated on both the
CICIoT2023 balanced test split and CICIoMT2024 — no imputation anywhere.
"""
import gc
import glob

import numpy as np
import torch

from common import compute_metrics, print_table, save_results, set_seed
from data import load_external_csv, load_meta, load_processed
from run_e1 import ALL_MODELS, get_model
from run_e2 import CICIOMT_GLOB, label_from_filename
from train import predict


def main():
    set_seed()
    meta = load_meta()
    drop = [meta["feature_cols"].index(c) for c in ("flow_duration", "urg_count")]
    keep = [i for i in range(len(meta["feature_cols"])) if i not in drop]

    Xtr, _, ytr2 = load_processed("train")
    Xte, _, yte2 = load_processed("test")
    Xext, yext, _ = load_external_csv(CICIOMT_GLOB, label_from_filename=label_from_filename)
    Xtr, Xte, Xext = Xtr[:, keep], Xte[:, keep], Xext[:, keep]
    print(f"[e2-ablated] features={Xtr.shape[1]} ext_rows={len(Xext)}")

    results, rows = {"n_features": len(keep)}, []
    for name in [m for m in ALL_MODELS if m != "KDTeacherCNN"]:
        m = get_model(name, "binary44", Xtr, ytr2, 2)
        r_cic = compute_metrics(yte2, predict(m, Xte) if isinstance(m, torch.nn.Module) else m.predict(Xte), 2)
        r_ext = compute_metrics(yext, predict(m, Xext) if isinstance(m, torch.nn.Module) else m.predict(Xext), 2)
        results[name] = {"ciciot_balanced": r_cic, "ciciomt": r_ext}
        rows.append([name, f"{r_cic['f1_macro']:.4f}", f"{r_ext['f1_macro']:.4f}",
                     f"{r_ext['mcc']:.4f}", f"{r_ext['detection_rate']:.4f}", f"{r_ext['fpr']:.4f}"])
        del m
        gc.collect()

    print_table(rows, ["model", "CICIoT F1", "ext F1", "ext MCC", "ext DR", "ext FPR"])
    save_results("e2_ablated_44feat", results)


if __name__ == "__main__":
    main()
