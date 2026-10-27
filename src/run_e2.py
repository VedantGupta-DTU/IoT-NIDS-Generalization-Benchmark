"""E2 — Cross-dataset: apply the exact CICIoT2023-trained models (binary task,
unchanged weights, CICIoT2023 scaler) to a second dataset.

Expected second datasets (either works once downloaded):
  1. CICIoMT2024 native ML CSVs  -> data/ciciomt2024/**/*.csv
     (same CIC feature extractor as CICIoT2023; columns align by name)
  2. TabularIoTAttacks-2024 slices -> data/tabular_iot_2024/<slice>/*.csv
     (unified schema across 9 datasets; use with retrain_on_unified=True
      workflow in run_e3.py instead for non-CICIoT feature spaces)

Both are gated behind the CIC registration form:
  https://cicresearch.ca/IOTDataset/CIC-BCCC-NRC-TabularIoTAttacks-2024/
  https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
Fill the form once, then place the extracted CSVs in the folders above.
"""
import glob
import os
import sys

import torch

from common import DATA_DIR, compute_metrics, print_table, save_results, set_seed
from data import load_external_csv, load_meta, load_processed
from run_e1 import NN_MODELS, train_all
from train import predict

CICIOMT_GLOB = os.path.join(DATA_DIR, "ciciomt2024", "**", "*.csv")


def label_from_filename(basename: str) -> int:
    """CICIoMT2024 ships one CSV per class; the filename carries the label."""
    return 0 if basename.lower().startswith("benign") else 1


def main():
    set_seed()
    if not glob.glob(CICIOMT_GLOB, recursive=True):
        print("E2 blocked: no external dataset found.")
        print(f"  expected CSVs at {CICIOMT_GLOB}")
        print("  Download CICIoMT2024 (WiFI_and_MQTT/attacks CSV folder) via the CIC form:")
        print("  https://www.unb.ca/cic/datasets/iomt-dataset-2024.html")
        sys.exit(1)

    meta = load_meta()
    Xtr, _, ytr2 = load_processed("train")
    models = train_all("binary", Xtr, ytr2, 2)

    print("[e2] loading external dataset aligned to CICIoT2023 features")
    X, y, missing = load_external_csv(CICIOMT_GLOB, label_from_filename=label_from_filename)
    print(f"[e2] rows={len(X)} attack_ratio={y.mean():.3f} missing_features={missing}")

    results = {"n_rows": int(len(X)), "attack_ratio": float(y.mean()), "missing_features": missing}
    rows = []
    for name in NN_MODELS + ["RandomForest", "LightGBM"]:
        m = models[name]
        yp = predict(m, X) if isinstance(m, torch.nn.Module) else m.predict(X)
        r = compute_metrics(y, yp, 2)
        results[name] = r
        rows.append([name, f"{r['accuracy']:.4f}", f"{r['f1_macro']:.4f}", f"{r['mcc']:.4f}",
                     f"{r['detection_rate']:.4f}", f"{r['fpr']:.4f}"])
    print_table(rows, ["model", "acc", "macroF1", "MCC", "DR", "FPR"])
    save_results("e2_cross_dataset", results)


if __name__ == "__main__":
    main()
