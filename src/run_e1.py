"""E1 — In-distribution: train all models on CICIoT2023, evaluate on its
held-out test split. Saves checkpoints reused by E2/E4/E5.

Tasks: 8-class attack-category classification and binary benign/attack.
Models: MLP14k, KDStudentCNN (distilled from KDTeacherCNN), DLBiLSTM, TinyMLP,
plus RandomForest and LightGBM classical baselines.

Memory note: models are trained/evaluated one at a time and released
(a full roster held at once caused OOM-driven segfaults on 24 GB RAM).
"""
import gc
import os
import pickle
import time

import numpy as np
import torch

from common import RESULTS_DIR, SEED, compute_metrics, print_table, save_results, set_seed
from data import load_meta, load_processed, prepare_ciciot2023
from models import KDTeacherCNN, build, param_count
from train import predict, train_model

CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

NN_MODELS = ["MLP14k", "KDStudentCNN", "DLBiLSTM", "TinyMLP"]
ALL_MODELS = ["KDTeacherCNN"] + NN_MODELS + ["RandomForest", "LightGBM"]
EPOCHS = 4


def ckpt_path(name: str, task: str) -> str:
    ext = "pt" if name not in ("RandomForest", "LightGBM") else "pkl"
    return os.path.join(CKPT_DIR, f"{name}_{task}.{ext}")


def _rf():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_leaf=5, n_jobs=8,
        random_state=SEED, class_weight="balanced_subsample",
    )


def get_model(name: str, task: str, Xtr, ytr, n_classes: int, teacher=None):
    """Load a trained model from checkpoint, training (and saving) if absent.
    LightGBM runs fully in a subprocess (OpenMP/MPS conflict, see lgbm_iso)."""
    path = ckpt_path(name, task)
    if name == "LightGBM":
        from lgbm_iso import SubprocessLGBM

        if os.path.exists(path):
            return SubprocessLGBM(path)
        print(f"[{task}] training {name} (subprocess)")
        return SubprocessLGBM.fit_new(Xtr, ytr, path)
    if name == "RandomForest":
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        print(f"[{task}] training {name}")
        clf = _rf()
        clf.fit(Xtr, ytr)
        with open(path, "wb") as f:
            pickle.dump(clf, f)
        return clf

    in_dim = Xtr.shape[1]
    model = KDTeacherCNN(in_dim, n_classes) if name == "KDTeacherCNN" else build(name, in_dim, n_classes)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
        return model
    print(f"[{task}] training {name} ({param_count(model):,} params)")
    kw = {}
    if name == "KDStudentCNN":
        kw["teacher"] = teacher if teacher is not None else get_model("KDTeacherCNN", task, Xtr, ytr, n_classes)
    epochs = 3 if name == "KDTeacherCNN" else EPOCHS
    train_model(model, Xtr, ytr, n_classes, epochs=epochs, **kw)
    torch.save(model.cpu().state_dict(), path)
    return model


def train_all(task: str, Xtr, ytr, n_classes: int, names=None) -> dict:
    """Load/train several models at once (for runners that need the full
    roster in memory). Prefer get_model + release in loops when possible."""
    return {n: get_model(n, task, Xtr, ytr, n_classes) for n in (names or ALL_MODELS)}


def main():
    set_seed(SEED)
    meta = prepare_ciciot2023()
    Xtr, ytr8, ytr2 = load_processed("train")
    Xte, yte8, yte2 = load_processed("test")
    Xtn, ytn8, ytn2 = load_processed("test_natural")
    n8 = len(meta["classes"])
    print(f"train={Xtr.shape} test={Xte.shape} test_natural={Xtn.shape} classes={meta['classes']}")

    all_results = {"meta": {"classes": meta["classes"], "n_train": len(Xtr),
                            "n_test": len(Xte), "n_test_natural": len(Xtn)}}
    tasks = [("8class", ytr8, [("balanced", Xte, yte8), ("natural", Xtn, ytn8)], n8),
             ("binary", ytr2, [("balanced", Xte, yte2), ("natural", Xtn, ytn2)], 2)]
    for task, ytr, evalsets, nc in tasks:
        print(f"\n===== task: {task} =====")
        res = {}
        for name in ALL_MODELS:
            m = get_model(name, task, Xtr, ytr, nc)
            res[name] = {}
            for setname, X, y in evalsets:
                t0 = time.time()
                yp = predict(m, X) if isinstance(m, torch.nn.Module) else m.predict(X)
                r = compute_metrics(y, yp, nc)
                r["eval_seconds"] = round(time.time() - t0, 1)
                res[name][setname] = r
            print(f"  {name}: nat_acc={res[name]['natural']['accuracy']:.4f} "
                  f"bal_acc={res[name]['balanced']['accuracy']:.4f} "
                  f"bal_macroF1={res[name]['balanced']['f1_macro']:.4f}")
            del m
            gc.collect()
        all_results[task] = res
        rows = [
            [n, f"{r['natural']['accuracy']:.4f}", f"{r['balanced']['accuracy']:.4f}",
             f"{r['balanced']['f1_macro']:.4f}", f"{r['balanced']['mcc']:.4f}"]
            for n, r in res.items()
        ]
        print_table(rows, ["model", "natural acc", "balanced acc", "bal macroF1", "bal MCC"])

    save_results("e1_in_distribution", all_results)


if __name__ == "__main__":
    main()
