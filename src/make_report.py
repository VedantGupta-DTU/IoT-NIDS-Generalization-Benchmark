"""Collect all results/*.json into a single markdown results report."""
import json
import os

from common import PROJECT_ROOT, RESULTS_DIR

OUT = os.path.join(PROJECT_ROOT, "results", "RESULTS.md")
MODEL_ORDER = ["MLP14k", "KDStudentCNN", "KDTeacherCNN", "DLBiLSTM", "TinyMLP", "RandomForest", "LightGBM"]


def load(name):
    p = os.path.join(RESULTS_DIR, f"{name}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    lines = ["# Experimental Results — Lightweight IoT-NIDS Generalization Benchmark\n"]

    e1 = load("e1_in_distribution")
    if e1:
        lines.append(f"Training data: CICIoT2023 stratified subsample "
                     f"({e1['meta']['n_train']:,} train / {e1['meta']['n_test']:,} test rows, "
                     f"classes: {', '.join(e1['meta']['classes'])}).\n")
        for task in ["8class", "binary"]:
            lines.append(f"\n## E1 — In-distribution ({task})\n")
            lines.append("`natural` = unmodified class mix (comparable to published numbers); "
                         "`balanced` = per-class-capped stress test.\n")
            rows = []
            for m in MODEL_ORDER:
                r = e1.get(task, {}).get(m)
                if r:
                    nat, bal = r["natural"], r["balanced"]
                    rows.append([m, fmt(nat["accuracy"]), fmt(nat["f1_macro"]),
                                 fmt(bal["accuracy"]), fmt(bal["f1_macro"]), fmt(bal["mcc"]),
                                 fmt(bal.get("fpr", bal.get("fpr_macro")))])
            lines.append(table(["model", "nat acc", "nat F1(macro)", "bal acc", "bal F1(macro)",
                                "bal MCC", "bal FPR"], rows))

    e1u = load("e1_unweighted_8class")
    if e1u:
        lines.append("\n## E1 addendum — unweighted CE (papers' protocol), 8-class\n")
        rows = []
        for m in MODEL_ORDER:
            r = e1u.get(m)
            if r:
                rows.append([m, fmt(r["natural"]["accuracy"]), fmt(r["natural"]["f1_macro"]),
                             fmt(r["balanced"]["accuracy"]), fmt(r["balanced"]["f1_macro"])])
        lines.append(table(["model", "nat acc", "nat F1(macro)", "bal acc", "bal F1(macro)"], rows))

    e2 = load("e2_cross_dataset")
    lines.append("\n## E2 — Cross-dataset (exact models, unchanged weights)\n")
    if e2:
        lines.append(f"External rows: {e2['n_rows']:,}, attack ratio {fmt(e2['attack_ratio'], 3)}, "
                     f"missing features: {e2['missing_features'] or 'none'}\n")
        rows = [[m, fmt(e2[m]["accuracy"]), fmt(e2[m]["f1_macro"]), fmt(e2[m]["mcc"]),
                 fmt(e2[m]["detection_rate"]), fmt(e2[m]["fpr"])]
                for m in MODEL_ORDER if m in e2]
        lines.append(table(["model", "acc", "F1(macro)", "MCC", "DR", "FPR"], rows))
    else:
        lines.append("*Pending — needs the gated CICIoMT2024 / TabularIoTAttacks download "
                     "(see src/run_e2.py).*\n")

    e2a = load("e2_ablated_44feat")
    if e2a:
        lines.append("\n### E2 ablation — retrained on the 44 shared features (no imputation)\n")
        rows = []
        for m in MODEL_ORDER:
            r = e2a.get(m)
            if isinstance(r, dict) and "ciciomt" in r:
                rows.append([m, fmt(r["ciciot_balanced"]["f1_macro"]), fmt(r["ciciomt"]["f1_macro"]),
                             fmt(r["ciciomt"]["mcc"]), fmt(r["ciciomt"]["detection_rate"]),
                             fmt(r["ciciomt"]["fpr"])])
        lines.append(table(["model", "CICIoT F1", "CICIoMT F1", "CICIoMT MCC", "DR", "FPR"], rows))

    e3 = load("e3_combined_training")
    lines.append("\n## E3 — Combined-dataset training\n")
    if e3:
        rows = [[m, fmt(e3[m]["ciciot_test"]["f1_macro"]), fmt(e3[m]["external_test"]["f1_macro"])]
                for m in MODEL_ORDER if m in e3]
        lines.append(table(["model", "CICIoT2023 test F1", "external test F1"], rows))
    else:
        lines.append("*Pending — same gated data as E2.*\n")

    e4 = load("e4_synthetic_drift")
    if e4:
        lines.append("\n## E4 — Synthetic & drifted data (8-class, macro-F1)\n")
        conds = [("a_trts_synthetic_test", "synthetic test"),
                 *[(f"b::{k}", k) for k in e4.get("b_drifted_real_test", {})]]
        headers = ["model"] + [c[1] for c in conds]
        rows = []
        for m in MODEL_ORDER:
            row = [m]
            ok = False
            for key, _ in conds:
                r = (e4["b_drifted_real_test"][key[3:]] if key.startswith("b::")
                     else e4.get(key, {}))
                if m in r:
                    row.append(fmt(r[m]["f1_macro"]))
                    ok = True
                else:
                    row.append("—")
            if ok:
                rows.append(row)
        lines.append(table(headers, rows))
        if "c_tstr_real_test" in e4:
            lines.append("\n**TSTR** (train on synthetic only, test on real):\n")
            rows = [[m, fmt(r["accuracy"]), fmt(r["f1_macro"]), fmt(r["mcc"])]
                    for m, r in e4["c_tstr_real_test"].items()]
            lines.append(table(["model", "acc", "F1(macro)", "MCC"], rows))

    e5 = load("e5_edge_cost")
    if e5:
        lines.append("\n## E5 — Edge cost (CPU proxy; not a Raspberry Pi)\n")
        rows = []
        for m, r in e5.items():
            if m.startswith("_"):
                continue
            rows.append([m, f"{r['params']:,}", fmt(r["fp32_mb"], 3), fmt(r["int8_mb"], 3),
                         fmt(r["lat_b1_ms"], 3), fmt(r["lat_b256_ms_per_sample"], 4),
                         fmt(r["fp32_f1_macro"]), fmt(r["int8_f1_macro"])])
        lines.append(table(["model", "params", "FP32 MB", "INT8 MB", "lat b=1 (ms)",
                            "lat b=256 (ms/sample)", "F1", "INT8 F1"], rows))

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
