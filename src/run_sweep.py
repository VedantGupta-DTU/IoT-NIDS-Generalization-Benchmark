"""Seed x epoch-budget sweep hardening the headline binary results (E1/E2).

Retrains the four lightweight NNs (MLP14k, KDStudentCNN distilled from a
per-seed KDTeacherCNN, DLBiLSTM, TinyMLP) on seeds {13, 21, 42} x epochs
{4, 10}, and the tree baselines (RandomForest, LightGBM) on the same three
seeds, all on the binary task. Every run is evaluated on the CICIoT2023
balanced test split and on CICIoMT2024 (same loader, filename labels and
per-file cap as run_e2, so the external eval set is identical across runs).
An optional 8-class arm re-runs MLP14k per seed/epoch in-distribution to
check the finding-#2 reproducibility gap across seeds.

Per-run metrics accumulate in results/sweep.json (incremental and resumable:
a run with a record is skipped; a run with a checkpoint but no record is
re-evaluated without retraining). Checkpoints go to results/checkpoints_sweep/
and never touch results/checkpoints/.

Seeding without modifying existing files: train.train_model reseeds itself
from train.SEED and its loader's shuffle generator default was bound at
import, run_e1._rf reads run_e1.SEED, and lgbm_iso's CLI pins SEED=42 — so
this script points each module-level constant at the run's seed (for
LightGBM inside the same kind of clean, torch-free child process that
lgbm_iso.SubprocessLGBM uses; an in-process LightGBM fit segfaults after
torch has run MPS ops).

Usage:
  python3 run_sweep.py             # full sweep, then rebuild SWEEP.md tables
  python3 run_sweep.py summarize   # rebuild results/SWEEP.md from sweep.json
                                   # (regenerates the whole file)
  python3 run_sweep.py --smoke [--smoke-dir D]   # tiny end-to-end check
"""
import argparse
import gc
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

import run_e1
import train as train_mod
from common import DATA_DIR, RESULTS_DIR, compute_metrics, set_seed
from data import load_external_csv, load_meta, load_processed
from lgbm_iso import SubprocessLGBM
from models import KDTeacherCNN, build, param_count
from run_e2 import CICIOMT_GLOB, label_from_filename
from train import predict, train_model

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

SEEDS = [13, 21, 42]
EPOCH_ARMS = [4, 10]
NN_MODELS = ["MLP14k", "KDStudentCNN", "DLBiLSTM", "TinyMLP"]
TREE_MODELS = ["RandomForest", "LightGBM"]
TEACHER_EPOCHS = 3  # fixed as in run_e1; one teacher per seed, shared by both epoch arms


def default_cfg() -> dict:
    return {
        "ckpt_dir": os.path.join(RESULTS_DIR, "checkpoints_sweep"),
        "sweep_json": os.path.join(RESULTS_DIR, "sweep.json"),
        "sweep_md": os.path.join(RESULTS_DIR, "SWEEP.md"),
        "seeds": SEEDS,
        "epoch_arms": EPOCH_ARMS,
        "external_glob": CICIOMT_GLOB,
        "external_cap": 200_000,  # per-file cap, matches run_e2's eval set
        "train_subsample": None,
        "eightclass_mlp14k": True,
    }


def run_key(model: str, task: str, seed: int, epochs) -> str:
    return f"{model}|{task}|s{seed}|e{epochs if epochs is not None else '-'}"


def seed_everything(seed: int) -> None:
    """Make the shared training code honor this run's seed: train_model calls
    set_seed(train.SEED) internally, and _loader's shuffle-generator seed is a
    default argument bound to 42 at import time."""
    train_mod.SEED = seed
    train_mod._loader.__defaults__ = (seed,)
    set_seed(seed)


# ---------------------------------------------------------------- training


def get_nn(cfg, name, task, seed, epochs, Xtr, ytr, n_classes):
    path = os.path.join(cfg["ckpt_dir"], f"{name}_{task}_s{seed}_e{epochs}.pt")
    seed_everything(seed)
    in_dim = Xtr.shape[1]
    model = KDTeacherCNN(in_dim, n_classes) if name == "KDTeacherCNN" else build(name, in_dim, n_classes)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
        return model, 0.0
    kw = {}
    if name == "KDStudentCNN":
        teacher, tsec = get_nn(cfg, "KDTeacherCNN", task, seed, TEACHER_EPOCHS, Xtr, ytr, n_classes)
        if tsec:
            print(f"  (teacher s{seed} trained in {tsec:.0f}s)", flush=True)
        kw["teacher"] = teacher
    print(f"[train] {name} {task} s{seed} e{epochs} ({param_count(model):,} params)", flush=True)
    t0 = time.time()
    train_model(model, Xtr, ytr, n_classes, epochs=epochs, **kw)
    torch.save(model.cpu().state_dict(), path)
    return model, time.time() - t0


_LGBM_CHILD = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "import lgbm_iso\n"
    "lgbm_iso.SEED = int(sys.argv[2])\n"
    "lgbm_iso._fit(sys.argv[3], sys.argv[4], sys.argv[5])\n"
)


def lgbm_fit_seeded(X, y, pkl_path, seed):
    """lgbm_iso's own CLI pins SEED=42; run its _fit in the same kind of clean
    child process (never importing torch) with the constant pointed at `seed`."""
    with tempfile.TemporaryDirectory() as td:
        xp, yp = os.path.join(td, "X.npy"), os.path.join(td, "y.npy")
        np.save(xp, X)
        np.save(yp, y)
        r = subprocess.run(
            [sys.executable, "-c", _LGBM_CHILD, SRC_DIR, str(seed), xp, yp, pkl_path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"seeded lgbm fit failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    return SubprocessLGBM(pkl_path)


def get_tree(cfg, name, task, seed, Xtr, ytr):
    path = os.path.join(cfg["ckpt_dir"], f"{name}_{task}_s{seed}.pkl")
    if name == "LightGBM":
        if os.path.exists(path):
            return SubprocessLGBM(path), 0.0
        print(f"[train] {name} {task} s{seed} (subprocess)", flush=True)
        t0 = time.time()
        clf = lgbm_fit_seeded(Xtr, ytr, path, seed)
        return clf, time.time() - t0
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f), 0.0
    print(f"[train] {name} {task} s{seed}", flush=True)
    run_e1.SEED = seed  # _rf() reads its module global at call time
    t0 = time.time()
    clf = run_e1._rf()
    clf.fit(Xtr, ytr)
    tsec = time.time() - t0
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    return clf, tsec


# ---------------------------------------------------------------- bookkeeping


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"meta": {}, "runs": {}}


def flush_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=float)
    os.replace(tmp, path)


def do_run(state, cfg, name, task, seed, epochs, Xtr, ytr, n_classes, evalsets, is_tree=False):
    key = run_key(name, task, seed, epochs)
    if key in state["runs"]:
        print(f"[skip] {key} already recorded", flush=True)
        return
    if is_tree:
        model, tsec = get_tree(cfg, name, task, seed, Xtr, ytr)
        params = None
    else:
        model, tsec = get_nn(cfg, name, task, seed, epochs, Xtr, ytr, n_classes)
        params = param_count(model)
    rec = {"model": name, "task": task, "seed": seed, "epochs": epochs, "params": params,
           "train_seconds": round(tsec, 1), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "evals": {}}
    for ename, X, y, nc in evalsets:
        yp = predict(model, X) if isinstance(model, torch.nn.Module) else model.predict(X)
        rec["evals"][ename] = compute_metrics(y, yp, nc)
    state["runs"][key] = rec
    flush_state(cfg["sweep_json"], state)
    line = "  ".join(f"{e}: mcc={m['mcc']:.3f}" for e, m in rec["evals"].items())
    print(f"[done] {key} (train {tsec:.0f}s)  {line}", flush=True)
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------- summary


def _fmt(vals) -> str:
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{m:.3f} ± {s:.3f}"


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def summarize(cfg) -> None:
    state = load_state(cfg["sweep_json"])
    runs = list(state["runs"].values())
    meta = state.get("meta", {})
    seeds = meta.get("seeds", SEEDS)
    arms = meta.get("epoch_arms", EPOCH_ARMS)

    def select(model, task, epochs):
        return [r for r in runs if r["model"] == model and r["task"] == task and r["epochs"] == epochs]

    def table(eval_name, metrics, task="binary"):
        rows = []
        for model in NN_MODELS:
            for ep in arms:
                recs = [r for r in select(model, task, ep) if eval_name in r["evals"]]
                if recs:
                    rows.append([model, ep, f"n={len(recs)}"]
                                + [_fmt([r["evals"][eval_name][met] for r in recs]) for met in metrics])
        for model in TREE_MODELS:
            recs = [r for r in select(model, task, None) if eval_name in r["evals"]]
            if recs:
                rows.append([model, "—", f"n={len(recs)}"]
                            + [_fmt([r["evals"][eval_name][met] for r in recs]) for met in metrics])
        return rows

    binary_metrics = ["accuracy", "f1_macro", "mcc", "fpr"]
    headers = ["model", "epochs", "runs", "accuracy", "macro-F1", "MCC", "FPR"]
    lines = [
        "# Seed × Epoch Sweep — robustness of the headline numbers",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `src/run_sweep.py` from "
        f"[sweep.json](sweep.json). Seeds {seeds}, epoch budgets {arms} (binary task; "
        f"trees have no epoch arm), teacher fixed at {TEACHER_EPOCHS} epochs per seed. "
        "Same splits, scaler and CICIoMT2024 eval subsample as E1/E2 "
        f"({meta.get('external', {}).get('rows', '?')} external rows, attack ratio "
        f"{meta.get('external', {}).get('attack_ratio', '?')}). "
        "Values are mean ± sample std across seeds.",
        "",
        "## Binary — CICIoT2023 balanced test (in-distribution)",
        "",
        _md_table(headers, table("ciciot2023_balanced", binary_metrics)),
        "",
        "## Binary — CICIoMT2024 (cross-dataset, unchanged weights)",
        "",
        _md_table(headers, table("ciciomt2024", binary_metrics)),
        "",
        "## Cross-dataset MCC per seed (binary)",
        "",
    ]
    rows = []
    for model in NN_MODELS + TREE_MODELS:
        eps = arms if model in NN_MODELS else [None]
        for ep in eps:
            cells = [model, ep if ep is not None else "—"]
            found = False
            for s in seeds:
                rec = state["runs"].get(run_key(model, "binary", s, ep))
                if rec and "ciciomt2024" in rec["evals"]:
                    cells.append(f"{rec['evals']['ciciomt2024']['mcc']:.3f}")
                    found = True
                else:
                    cells.append("—")
            if found:
                rows.append(cells)
    lines += [_md_table(["model", "epochs"] + [f"seed {s}" for s in seeds], rows), ""]

    rows8 = table("ciciot2023_balanced", ["accuracy", "f1_macro", "mcc", "fpr_macro"], task="8class")
    rows8 = [r for r in rows8 if r[0] == "MLP14k"]
    if rows8:
        lines += [
            "## 8-class arm — MLP14k, CICIoT2023 balanced test",
            "",
            _md_table(["model", "epochs", "runs", "accuracy", "macro-F1", "MCC", "FPR (macro-OvR)"], rows8),
            "",
        ]
    lines += [
        "## Verdict vs. the single-seed findings",
        "",
        "_(filled in after the sweep completes — see results/FINDINGS.md §2–3)_",
        "",
    ]
    with open(cfg["sweep_md"], "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] {cfg['sweep_md']}", flush=True)


# ---------------------------------------------------------------- driver


def main(cfg) -> None:
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    state = load_state(cfg["sweep_json"])

    Xtr, ytr8, ytr2 = load_processed("train")
    Xte, yte8, yte2 = load_processed("test")
    if cfg["train_subsample"]:
        idx = np.random.default_rng(0).choice(len(Xtr), cfg["train_subsample"], replace=False)
        Xtr, ytr8, ytr2 = Xtr[idx], ytr8[idx], ytr2[idx]
    n8 = len(load_meta()["classes"])

    print("[sweep] loading CICIoMT2024 ...", flush=True)
    Xext, yext, missing = load_external_csv(
        cfg["external_glob"], label_from_filename=label_from_filename,
        per_file_cap=cfg["external_cap"],
    )
    print(f"[sweep] external rows={len(Xext)} attack_ratio={yext.mean():.3f} missing={missing}", flush=True)

    state["meta"] = {
        "seeds": cfg["seeds"], "epoch_arms": cfg["epoch_arms"], "teacher_epochs": TEACHER_EPOCHS,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "external": {"rows": int(len(Xext)), "attack_ratio": round(float(yext.mean()), 4),
                     "missing_features": missing, "per_file_cap": cfg["external_cap"]},
    }
    flush_state(cfg["sweep_json"], state)

    binary_evals = [("ciciot2023_balanced", Xte, yte2, 2), ("ciciomt2024", Xext, yext, 2)]
    for seed in cfg["seeds"]:
        for epochs in cfg["epoch_arms"]:
            for name in NN_MODELS:
                do_run(state, cfg, name, "binary", seed, epochs, Xtr, ytr2, 2, binary_evals)
        for name in TREE_MODELS:
            do_run(state, cfg, name, "binary", seed, None, Xtr, ytr2, 2, binary_evals, is_tree=True)
        if cfg["eightclass_mlp14k"]:
            for epochs in cfg["epoch_arms"]:
                do_run(state, cfg, "MLP14k", "8class", seed, epochs, Xtr, ytr8, n8,
                       [("ciciot2023_balanced", Xte, yte8, n8)])

    summarize(cfg)
    print("[sweep] complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "summarize"])
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end plumbing check")
    ap.add_argument("--smoke-dir", default=None, help="output dir for --smoke (default: a temp dir)")
    args = ap.parse_args()

    cfg = default_cfg()
    if args.smoke:
        out = args.smoke_dir or tempfile.mkdtemp(prefix="sweep_smoke_")
        os.makedirs(out, exist_ok=True)
        cfg.update(
            ckpt_dir=os.path.join(out, "ckpt"),
            sweep_json=os.path.join(out, "sweep.json"),
            sweep_md=os.path.join(out, "SWEEP.md"),
            seeds=[13], epoch_arms=[1],
            external_glob=os.path.join(DATA_DIR, "ciciomt2024", "test", "[AB]*.csv"),
            external_cap=5_000, train_subsample=20_000,
        )
        print(f"[smoke] outputs under {out}", flush=True)

    if args.mode == "summarize":
        summarize(cfg)
    else:
        main(cfg)
