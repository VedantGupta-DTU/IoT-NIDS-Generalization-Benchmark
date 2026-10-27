"""E3 — Combined-dataset training: retrain the same architectures on
CICIoT2023 + a second dataset and test on (a) each dataset's own test split
and (b) the held-out external data, to measure whether multi-dataset training
restores transfer (binary task).

Requires the same external data as run_e2.py (see its docstring for the CIC
form links). Uses a 50/50 mixing ratio by default.
"""
import glob
import os
import sys

import numpy as np
import torch

from common import SEED, compute_metrics, print_table, save_results, set_seed
from data import load_external_csv, load_processed
from models import build
from run_e2 import CICIOMT_GLOB, label_from_filename
from run_e1 import NN_MODELS
from train import predict, train_model


def main():
    set_seed()
    if not glob.glob(CICIOMT_GLOB, recursive=True):
        print("E3 blocked: no external dataset found (see run_e2.py docstring).")
        sys.exit(1)

    Xtr, _, ytr2 = load_processed("train")
    Xte, _, yte2 = load_processed("test")

    Xext, yext, _ = load_external_csv(CICIOMT_GLOB, label_from_filename=label_from_filename)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(Xext))
    half = len(idx) // 2
    Xext_tr, yext_tr = Xext[idx[:half]], yext[idx[:half]]
    Xext_te, yext_te = Xext[idx[half:]], yext[idx[half:]]

    n_mix = min(len(Xtr), len(Xext_tr))
    Xmix = np.concatenate([Xtr[:n_mix], Xext_tr[:n_mix]])
    ymix = np.concatenate([ytr2[:n_mix], yext_tr[:n_mix]])
    print(f"[e3] combined train: {len(Xmix)} rows ({n_mix} from each dataset)")

    results = {}
    rows = []
    for name in NN_MODELS:
        m = build(name, Xtr.shape[1], 2)
        print(f"[e3] training {name} on combined data")
        train_model(m, Xmix, ymix, 2, epochs=3, verbose=False)
        r_cic = compute_metrics(yte2, predict(m, Xte), 2)
        r_ext = compute_metrics(yext_te, predict(m, Xext_te), 2)
        results[name] = {"ciciot_test": r_cic, "external_test": r_ext}
        rows.append([name, f"{r_cic['f1_macro']:.4f}", f"{r_ext['f1_macro']:.4f}"])

    print_table(rows, ["model", "CICIoT F1", "external F1"])
    save_results("e3_combined_training", results)


if __name__ == "__main__":
    main()
