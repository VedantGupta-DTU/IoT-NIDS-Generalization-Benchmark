# Key Findings — Lightweight IoT-NIDS Generalization Benchmark

Interpretation of [RESULTS.md](RESULTS.md). Four published lightweight
CICIoT2023 architectures re-implemented at their papers' stated parameter
budgets, plus RandomForest/LightGBM baselines, all trained and evaluated under
one shared protocol (seed 42, same splits, same scaler). Cross-dataset target:
CICIoMT2024 (official WiFi+MQTT attack CSVs, 8.7M flows, 45/46 features shared
with CICIoT2023 by name).

## 1. Published headline numbers reproduce only under the papers' easy protocol

On the **natural class mix** (72.8% DDoS), binary accuracies land at
98.6–99.7% — right where the papers claim. The tree baselines also reproduce
the ~99% 8-class claims (RF 99.26%, LightGBM 99.50%). But the same trained
models on a **class-balanced test** drop hard: the KD student falls from 98.6%
to 78.4% binary accuracy. Published accuracy on CICIoT2023 substantially
measures the DDoS majority, not detection ability; any comparison quoting
natural-mix accuracy is comparing class priors.

## 2. A genuine reproducibility gap for the tiny neural models (8-class)

The tiny NNs plateau at 69–79% 8-class accuracy whether trained with weighted
or plain CE — far from the 96–98% their papers report — while trees reach 99%.
Errors concentrate in DDoS↔DoS confusion and Recon/Spoofing→Web-based
leakage: sharp-threshold splits that trees find and 2k–14k-parameter networks
do not. *Stated architectures at stated sizes do not reproduce stated numbers
under a controlled protocol.*

## 3. Cross-dataset transfer (E2): the benchmark ranking inverts — headline result

Exact CICIoT2023-trained binary models applied to CICIoMT2024, weights
unchanged. Accuracy is meaningless there (97.4% of flows are attacks; an
"everything is an attack" detector scores 97.4%), so MCC and benign-FPR carry
the story:

| model | in-dist F1 (balanced) | CICIoMT MCC | benign FPR |
|---|---|---|---|
| DLBiLSTM | 0.843 | **0.816** | **0.132** |
| KDStudentCNN | 0.724 | 0.628 | 0.508 |
| MLP14k | 0.840 | 0.372 | 0.779 |
| TinyMLP | 0.840 | 0.192 | 0.903 |
| RandomForest | **0.929** | 0.077 | 0.992 |
| LightGBM | 0.927 | 0.166 | 0.963 |

The in-distribution winners (trees) collapse to near-random: RF flags 99.2% of
benign medical-IoT traffic as attack. The BiLSTM — bottom-tier on parts of the
benchmark — transfers best by a wide margin. Detection rate stays ≥0.993 for
every model; what breaks cross-domain is the *benign* model, i.e. false
alarms, exactly the operational failure mode the AL Mukhaini SLR warns about.

**Not an imputation artifact.** CICIoMT2024 lacks `flow_duration` and
`urg_count` (RF's #4 and #2 features; zero-filled above). Retraining everything
on the fair 44-feature intersection changes nothing for the trees (RF MCC
0.172, FPR 0.957; LightGBM 0.193/0.949) while the NNs hold or improve
(TinyMLP MCC 0.775/FPR 0.248, MLP14k 0.665, DLBiLSTM 0.610, student 0.585).
The tree collapse is a property of axis-aligned threshold memorization, which
E4's noise experiment predicted: trees lost the most under feature
perturbation (LightGBM macro-F1 0.831→0.292 at sigma=0.2) while NNs degraded
gracefully.

**Confirmed by the 3-seed sweep** ([SWEEP.md](SWEEP.md), seeds 13/21/42): the
tree collapse is not seed luck (RF cross-dataset MCC 0.069 ± 0.034, benign FPR
0.993 ± 0.005; even RF's best seed stays collapsed), the in-distribution tree
win is rock-stable, and at the 4-epoch budget every NN beats both trees on MCC
and FPR on every seed. "DLBiLSTM transfers best" replicates cleanly
(0.830 ± 0.013); the MLP14k/student/TinyMLP mid-order shuffles with seed, so
quote sweep means and claim only "trees collapse, NNs survive, DLBiLSTM best
at 4 epochs."

**New sweep result — more training destroys transfer.** Going from 4 to 10
epochs buys ~0.01–0.02 in-distribution MCC but collapses cross-dataset MCC for
three of four NNs into tree territory (DLBiLSTM 0.830→0.186 ± 0.083, MLP14k
0.396→0.157, TinyMLP 0.287→0.169; FPR ≥ 0.88). Only the distilled student is
budget-stable (0.292→0.287). The NN transfer advantage behaves like implicit
early stopping — a training-budget property that distillation preserves, not
pure architecture — so epoch budget must be reported as a first-class
experimental variable alongside seed.

## 4. Combined-dataset training (E3) restores transfer almost for free

Retraining the same architectures on 50/50 CICIoT2023+CICIoMT2024:
external-test F1 reaches 0.957–0.970 for every NN while the CICIoT2023
balanced F1 stays within ~0.02–0.03 of single-dataset training (e.g. MLP14k
0.840→0.820). Multi-dataset training is close to a free lunch here,
replicating the federated-study finding at model level.

## 5. Copula-synthetic traffic is a usable evaluation surrogate (E4)

Testing real-trained models on class-conditional Gaussian-copula synthetic
data costs all models only 0.03–0.09 macro-F1; TSTR (train on synthetic only)
reaches 0.59 vs 0.65 real-trained.

## 6. Edge deployment is a non-issue for size, but INT8 is not free (E5)

All four models are 11–60 KB FP32 with 0.013–0.17 ms single-flow CPU latency —
"lightweight" was never the binding constraint. Dynamic INT8 quantization
costs 6–10 macro-F1 points at this scale (DLBiLSTM 0.644→0.538), contradicting
the "nearly lossless" framing; 50% magnitude pruning is nearly free (TinyMLP
0.640→0.627).

## Paper narrative in one line

Benchmark-topping lightweight IoT-NIDS models owe their headline numbers to
easy protocols and threshold memorization; under domain shift the leaderboard
inverts, false alarms explode, and the cheapest fixes are training across
datasets and stopping training early — not a bigger model.

## Caveats

- Headline claims verified across seeds {13, 21, 42} and epoch budgets {4, 10}
  ([SWEEP.md](SWEEP.md)); the E1/E4/E5 tables themselves remain single-seed
  (42). One subsample protocol throughout.
- This LightGBM config uses no bagging/feature subsampling, so it is
  near-deterministic across seeds; RandomForest is the honest tree
  seed-variance measure.
- Cross-dataset evaluation is binary (label taxonomies differ across datasets).
- CICIoMT2024 is attack-heavy (97.4%); MCC/FPR are the meaningful metrics.
- Latency/size measured on an M4 Pro CPU as an edge proxy, not a Raspberry Pi.
- Models are faithful re-implementations from paper descriptions (no official
  code exists for any of the four).
