"""E1 addendum — unweighted-loss NN variant (8-class).

The main E1 trains NNs with class-weighted CE (macro-F1-oriented). Published
papers train with plain CE on the natural class mix and report accuracy, which
the majority classes dominate. This variant reproduces that protocol so the
comparison against published numbers is apples-to-apples.
"""
import gc

from common import compute_metrics, print_table, save_results, set_seed
from data import load_meta, load_processed
from models import build, param_count
from run_e1 import NN_MODELS
from train import predict, train_model

EPOCHS = 4


def main():
    set_seed()
    meta = load_meta()
    Xtr, ytr8, _ = load_processed("train")
    Xte, yte8, _ = load_processed("test")
    Xtn, ytn8, _ = load_processed("test_natural")
    n8 = len(meta["classes"])

    results, rows = {}, []
    for name in NN_MODELS:
        m = build(name, Xtr.shape[1], n8)
        print(f"[unweighted] training {name} ({param_count(m):,} params)")
        train_model(m, Xtr, ytr8, n8, epochs=EPOCHS, weighted=False, verbose=False)
        r_nat = compute_metrics(ytn8, predict(m, Xtn), n8)
        r_bal = compute_metrics(yte8, predict(m, Xte), n8)
        results[name] = {"natural": r_nat, "balanced": r_bal}
        rows.append([name, f"{r_nat['accuracy']:.4f}", f"{r_nat['f1_macro']:.4f}",
                     f"{r_bal['accuracy']:.4f}", f"{r_bal['f1_macro']:.4f}"])
        del m
        gc.collect()

    print_table(rows, ["model", "nat acc", "nat F1", "bal acc", "bal F1"])
    save_results("e1_unweighted_8class", results)


if __name__ == "__main__":
    main()
