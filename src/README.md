# Lightweight IoT-NIDS Generalization Benchmark

Code for the B.Tech project: reproduce published lightweight CICIoT2023 models
and measure whether their performance survives new / combined / synthetic data.

## Setup

Uses the system anaconda env (torch, sklearn, lightgbm, pandas, pyarrow,
scipy, huggingface_hub — all already installed). No extra installs needed.

## Data

- **CICIoT2023** (auto-downloaded): canonical mirror
  `lacg030175/CIC-IoT-2023-neto-full` on Hugging Face — 46,686,579 rows x 46
  features, Neto et al. (Sensors 2023) distribution, CC-BY-4.0. Stored in
  `../data/ciciot2023/`. `data.prepare_ciciot2023()` builds a stratified
  subsample (cap 250k/class train, 62.5k/class test -> 1.53M / 382k rows),
  standardized with a train-only scaler.
- **CICIoMT2024 / TabularIoTAttacks-2024** (manual, gated by the CIC
  registration form): see `run_e2.py` docstring for links and target folders.

## Models (see models.py for provenance)

| name | source paper | params (8-class) |
|---|---|---|
| MLP14k | Electronics 15(8):1604, 2026 | ~14.3k |
| KDStudentCNN | Applied Soft Computing 2024 (KD) | ~1.9k |
| DLBiLSTM | PeerJ CS 2023 (+ dynamic INT8) | ~40k |
| TinyMLP | reliable-edge-iot-ids artifact | ~7.6k |
| RandomForest / LightGBM | classical baselines | — |

All are re-implementations from paper descriptions (no official code exists);
parameter budgets match the papers' stated sizes. Undocumented details are
flagged inline.

## Experiments

```bash
python3 run_e1.py   # E1 in-distribution: train + eval on CICIoT2023 (saves checkpoints)
python3 run_e2.py   # E2 cross-dataset: exact trained models on external data (needs gated data)
python3 run_e3.py   # E3 combined-dataset retraining (needs gated data)
python3 run_e4.py   # E4 synthetic (copula TRTS/TSTR) + drift perturbations
python3 run_e5.py   # E5 edge cost: params, FP32/INT8 size, CPU latency, pruning
```

Run E1 first; E4/E5 reuse its checkpoints in `../results/checkpoints/`.
Results land in `../results/*.json`. Seed 42 everywhere.

## Protocol notes

- Tasks: 8-class attack category + binary benign/attack. Metrics: accuracy,
  macro-P/R/F1, weighted F1, MCC, FPR (macro-OvR for multiclass) — accuracy
  alone is misleading at 72.8% DDoS share.
- CICIoT2023's 46 features contain no IPs/ports/timestamps, so the shortcut
  features flagged by the generalization literature are absent by design.
- E5 latency is measured on this machine's CPU as an edge proxy, not a
  Raspberry Pi; rankings are indicative, absolute numbers are not.
