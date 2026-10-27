# Seed × Epoch Sweep — robustness of the headline numbers

Generated 2026-10-27 01:46 by `src/run_sweep.py` from [sweep.json](sweep.json). Seeds [13, 21, 42], epoch budgets [4, 10] (binary task; trees have no epoch arm), teacher fixed at 3 epochs per seed. Same splits, scaler and CICIoMT2024 eval subsample as E1/E2 (8718012 external rows, attack ratio 0.9736). Values are mean ± sample std across seeds.

## Binary — CICIoT2023 balanced test (in-distribution)

| model | epochs | runs | accuracy | macro-F1 | MCC | FPR |
|---|---|---|---|---|---|---|
| MLP14k | 4 | n=3 | 0.899 ± 0.005 | 0.846 ± 0.006 | 0.722 ± 0.008 | 0.041 ± 0.006 |
| MLP14k | 10 | n=3 | 0.910 ± 0.005 | 0.860 ± 0.005 | 0.742 ± 0.007 | 0.045 ± 0.008 |
| KDStudentCNN | 4 | n=3 | 0.832 ± 0.036 | 0.765 ± 0.031 | 0.586 ± 0.025 | 0.096 ± 0.059 |
| KDStudentCNN | 10 | n=3 | 0.830 ± 0.021 | 0.765 ± 0.018 | 0.597 ± 0.017 | 0.059 ± 0.030 |
| DLBiLSTM | 4 | n=3 | 0.896 ± 0.009 | 0.842 ± 0.010 | 0.715 ± 0.013 | 0.043 ± 0.012 |
| DLBiLSTM | 10 | n=3 | 0.908 ± 0.005 | 0.856 ± 0.006 | 0.736 ± 0.007 | 0.049 ± 0.009 |
| TinyMLP | 4 | n=3 | 0.902 ± 0.007 | 0.849 ± 0.008 | 0.725 ± 0.010 | 0.047 ± 0.009 |
| TinyMLP | 10 | n=3 | 0.907 ± 0.004 | 0.855 ± 0.005 | 0.736 ± 0.007 | 0.043 ± 0.006 |
| RandomForest | — | n=3 | 0.958 ± 0.000 | 0.929 ± 0.000 | 0.863 ± 0.000 | 0.025 ± 0.000 |
| LightGBM | — | n=3 | 0.958 ± 0.001 | 0.928 ± 0.001 | 0.864 ± 0.002 | 0.019 ± 0.000 |

## Binary — CICIoMT2024 (cross-dataset, unchanged weights)

| model | epochs | runs | accuracy | macro-F1 | MCC | FPR |
|---|---|---|---|---|---|---|
| MLP14k | 4 | n=3 | 0.977 ± 0.003 | 0.672 ± 0.045 | 0.396 ± 0.104 | 0.760 ± 0.059 |
| MLP14k | 10 | n=3 | 0.974 ± 0.001 | 0.538 ± 0.002 | 0.157 ± 0.025 | 0.951 ± 0.003 |
| KDStudentCNN | 4 | n=3 | 0.976 ± 0.001 | 0.600 ± 0.039 | 0.292 ± 0.059 | 0.872 ± 0.057 |
| KDStudentCNN | 10 | n=3 | 0.976 ± 0.000 | 0.594 ± 0.008 | 0.287 ± 0.012 | 0.884 ± 0.011 |
| DLBiLSTM | 4 | n=3 | 0.991 ± 0.001 | 0.915 ± 0.007 | 0.830 ± 0.013 | 0.141 ± 0.009 |
| DLBiLSTM | 10 | n=3 | 0.972 ± 0.001 | 0.573 ± 0.041 | 0.186 ± 0.083 | 0.898 ± 0.058 |
| TinyMLP | 4 | n=3 | 0.975 ± 0.002 | 0.612 ± 0.035 | 0.287 ± 0.086 | 0.852 ± 0.044 |
| TinyMLP | 10 | n=3 | 0.973 ± 0.001 | 0.551 ± 0.019 | 0.169 ± 0.057 | 0.934 ± 0.024 |
| RandomForest | — | n=3 | 0.974 ± 0.000 | 0.500 ± 0.005 | 0.069 ± 0.034 | 0.993 ± 0.005 |
| LightGBM | — | n=3 | 0.974 ± 0.000 | 0.522 ± 0.008 | 0.146 ± 0.021 | 0.970 ± 0.008 |

## Cross-dataset MCC per seed (binary)

| model | epochs | seed 13 | seed 21 | seed 42 |
|---|---|---|---|---|
| MLP14k | 4 | 0.306 | 0.510 | 0.372 |
| MLP14k | 10 | 0.129 | 0.179 | 0.163 |
| KDStudentCNN | 4 | 0.250 | 0.267 | 0.360 |
| KDStudentCNN | 10 | 0.275 | 0.298 | 0.290 |
| DLBiLSTM | 4 | 0.841 | 0.833 | 0.816 |
| DLBiLSTM | 10 | 0.150 | 0.281 | 0.126 |
| TinyMLP | 4 | 0.312 | 0.358 | 0.192 |
| TinyMLP | 10 | 0.204 | 0.200 | 0.103 |
| RandomForest | — | 0.032 | 0.099 | 0.077 |
| LightGBM | — | 0.123 | 0.148 | 0.165 |

## 8-class arm — MLP14k, CICIoT2023 balanced test

| model | epochs | runs | accuracy | macro-F1 | MCC | FPR (macro-OvR) |
|---|---|---|---|---|---|---|
| MLP14k | 4 | n=3 | 0.754 ± 0.006 | 0.641 ± 0.005 | 0.720 ± 0.006 | 0.034 ± 0.001 |
| MLP14k | 10 | n=3 | 0.767 ± 0.006 | 0.653 ± 0.006 | 0.734 ± 0.006 | 0.032 ± 0.001 |

## Verdict vs. the single-seed findings

**1. The tree cross-dataset collapse (FINDINGS §3) holds across seeds.**
RF MCC 0.069 ± 0.034 with benign FPR 0.993 ± 0.005; LightGBM 0.146 ± 0.021 /
0.970 ± 0.008. The single-seed values (0.077 / 0.166) were not seed luck.
Caveat: this LightGBM config uses no bagging or feature subsampling, so it is
near-deterministic across seeds by construction — RandomForest is the honest
tree seed-variance measure, and even its best seed (MCC 0.099) stays in
collapse territory.

**2. The in-distribution leaderboard is stable.** Trees win on every seed
(MCC 0.863–0.864, std ≤ 0.002) over the best NN (MLP14k, 0.742 ± 0.007), with
NN seed-std ≤ 0.03 throughout. Finding §1's protocol point is unaffected.

**3. At the original 4-epoch budget, "DLBiLSTM transfers best" replicates
cleanly; the rest of the NN ordering does not.** DLBiLSTM cross-dataset MCC is
0.830 ± 0.013 (0.816–0.841 across seeds; the original 0.816 is in-range). All
four NNs beat both trees on MCC and FPR on *every* seed at 4 epochs. But the
MLP14k / student / TinyMLP mid-order shuffles with seed (0.31–0.51 / 0.25–0.36 /
0.19–0.36), confirming FINDINGS' own caveat. The original student value of
0.628 (from the E1 checkpoint) lies outside this sweep's 3-seed range and
should be treated as run-to-run variance — quote the sweep means instead.

**4. New result: more epochs destroy transfer.** 10 vs 4 epochs buys ~0.01–0.02
in-distribution MCC but collapses cross-dataset MCC for three of four NNs
(DLBiLSTM 0.830 → 0.186 ± 0.083, MLP14k 0.396 → 0.157 ± 0.025, TinyMLP
0.287 → 0.169 ± 0.057), down to tree territory with FPR ≥ 0.88. Only the KD
student is budget-stable (0.292 → 0.287). The NN transfer advantage therefore
behaves like implicit early stopping — a *training-budget* property that
distillation preserves, not pure architecture — and the paper should report
epoch budget as a first-class variable alongside seed.

**5. The 8-class reproducibility gap (FINDINGS §2) is robust.** MLP14k lands
at 75.4 ± 0.6% (4 ep) and 76.7 ± 0.6% (10 ep) balanced accuracy across seeds —
the 96–98% paper claims are not reachable via seed luck or 2.5× more epochs.

**Bottom line for the paper:** findings §1, §2 and the tree collapse of §3
survive the sweep unchanged; §3's NN ordering should be weakened to
"DLBiLSTM best at 4 epochs (stable), remaining NNs unordered"; and the
epoch-budget sensitivity (point 4) is a new, citable result that strengthens
the threshold-memorization narrative.
