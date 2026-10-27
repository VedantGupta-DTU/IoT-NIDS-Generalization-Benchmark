# IoT Network Intrusion and Anomaly Detection Using a Bidirectional LSTM Autoencoder: Adversarial-Robustness Evaluation with Edge-Optimized Deployment

**Do published lightweight IoT intrusion-detection models keep working on a network they were not trained on?**

B.Tech Project (IT401), Department of Information Technology, Delhi Technological University.

| | |
|---|---|
| **Team** | Saksham Jain (23/IT/146), Vedant Gupta (23/IT/174), Abhinav Dubey (23/IT/007) |
| **Supervisor** | Prof. Shally Gupta |
| **Group number** | _add group number_ |

## Contents

- [What this project does](#what-this-project-does)
- [Repository contents](#repository-contents)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Evaluation method (important)](#evaluation-method-important)
- [Results so far](#results-so-far)
- [Status and limitations](#status-and-limitations)
- [References](#references)

## What this project does

Many recent journal papers report 96–99% accuracy for very small intrusion-detection models (about 2,000 to 15,000 parameters) on the CICIoT2023 dataset. None of them release code, and almost all of them test on the same dataset they trained on.

We rebuilt four of these published models from their papers, trained them all the same way on CICIoT2023, and then checked whether their scores hold up when

1. the test set is class-balanced instead of 73% DDoS (the papers' default);
2. the model is run, unchanged, on a different network: the CICIoMT2024 medical-IoT dataset;
3. the traffic is synthetic or drifted (noise, missing features, scale changes);
4. the model is quantised or pruned for an edge device.

Random Forest and LightGBM are included as classical baselines. Every result was repeated over three random seeds and two training budgets.

The short answer is that the published numbers come from an easy test setup, and the models that win the benchmark (tree ensembles) break down completely on a new network, while the small neural networks, especially the BiLSTM, hold up much better.

## Repository contents

| Path | Purpose |
|---|---|
| `src/data.py` | Downloads CICIoT2023 from the canonical Hugging Face mirror, builds the stratified subsample, standardises features, and aligns external datasets (CICIoMT2024) to the same feature space |
| `src/models.py` | The four rebuilt architectures (MLP14k, knowledge-distilled student CNN, DL-BiLSTM, TinyMLP) with the source paper noted in each docstring |
| `src/train.py` | Training loops: class-weighted cross-entropy and knowledge distillation |
| `src/lgbm_iso.py` | Runs LightGBM in a separate process (its OpenMP runtime crashes after PyTorch has used the Apple GPU in the same process) |
| `src/run_e1.py` | E1: train everything on CICIoT2023 and test on its natural and balanced test splits; saves checkpoints |
| `src/run_e1_unweighted.py` | E1 with plain (unweighted) cross-entropy, the papers' own training protocol |
| `src/run_e2.py` | E2: run the exact trained models on CICIoMT2024 |
| `src/run_e2_ablated.py` | E2 repeated on the 44 features both datasets share, with no imputation |
| `src/run_e3.py` | E3: retrain on CICIoT2023 + CICIoMT2024 together |
| `src/run_e4.py` | E4: synthetic data (Gaussian copula) and drift perturbations |
| `src/run_e5.py` | E5: model size, INT8 dynamic quantisation, 50% pruning, CPU latency |
| `src/run_sweep.py` | 36-run sweep over seeds {13, 21, 42} and epochs {4, 10} |
| `src/synthetic.py` | Per-class Gaussian-copula synthesiser and perturbation functions |
| `src/make_report.py` | Collects `results/*.json` into `results/RESULTS.md` |
| `src/fetch_models.py` | Downloads the six Random Forest models from the GitHub release (see [Trained models](#trained-models)) |
| `results/RESULTS.md` | All result tables |
| `results/FINDINGS.md` | What the results mean, written up |
| `results/SWEEP.md` | Mean ± std over seeds and training budgets |
| `results/checkpoints/` | Every model from the main experiments (seed 42): neural weights, LightGBM and Random Forest |
| `results/checkpoints_sweep/` | Every model from the 36-run sweep (seeds 13, 21, 42 × epochs 4, 10) |
| `research/lightweight-ciciot2023-models-and-datasets.md` | Literature survey: models, datasets, generalisation studies |
| `research/literature_review.xlsx` | Literature review workbook (24 papers and datasets, one row each) |

The datasets and the tree-model pickles are not committed (several GB). See Setup.

## Setup

Python 3.12. Tested on macOS (Apple M4 Pro, 24 GB RAM); training uses the Apple GPU when available and falls back to CPU.

```bash
git clone https://github.com/VedantGupta-DTU/IoT-NIDS-Generalization-Benchmark.git
cd IoT-NIDS-Generalization-Benchmark
pip install -r requirements.txt
```

**CICIoT2023** is downloaded automatically (1.6 GB of parquet) by `src/data.py` from the `lacg030175/CIC-IoT-2023-neto-full` mirror on Hugging Face, which preserves all 46,686,579 rows and 46 features of the original release (Neto et al., 2023).

**CICIoMT2024** is gated behind the Canadian Institute for Cybersecurity's registration form. Fill it at <https://www.unb.ca/cic/datasets/iomt-dataset-2024.html>, download `WiFI_and_MQTT/attacks/CSV/train` and `test`, and put the CSV files under `data/ciciomt2024/` (any sub-folder layout). Each file is one attack class; the label comes from the file name.

## How to run

Run E1 first; the other scripts reuse its checkpoints.

```bash
cd src
python3 run_e1.py            # ~15 min first time (download + preprocessing + training)
python3 run_e1_unweighted.py
python3 run_e4.py
python3 run_e5.py
python3 run_e2.py            # needs data/ciciomt2024/
python3 run_e2_ablated.py
python3 run_e3.py
python3 run_sweep.py         # ~2 hours
python3 make_report.py       # writes results/RESULTS.md
```

Everything is seeded (42 by default). With the trained models in place (see below), `run_e1.py` skips training and only evaluates, which takes about two minutes.

## Trained models

Every model whose numbers appear in the results is in the repository, so the tables can be re-checked without retraining:

| Files | What they are | Where |
|---|---|---|
| `results/checkpoints/{MLP14k,KDStudentCNN,KDTeacherCNN,DLBiLSTM,TinyMLP}_{8class,binary,binary44}.pt` | Neural weights, seed 42, for the 8-class, binary and 44-feature-ablation tasks | committed |
| `results/checkpoints/LightGBM_{8class,binary,binary44}.pkl` | LightGBM, seed 42 | committed |
| `results/checkpoints/RandomForest_{8class,binary,binary44}.pkl` | Random Forest, seed 42 (96–272 MB each) | release `models-v1`; run `python3 src/fetch_models.py` |
| `results/checkpoints_sweep/*_s{13,21,42}_e{4,10}.pt`, `LightGBM_binary_s*.pkl` | All sweep models, 3 seeds × 2 epoch budgets | committed |
| `results/checkpoints_sweep/RandomForest_binary_s{13,21,42}.pkl` | Sweep Random Forests (~96 MB each) | release `models-v1`; same script |

Loading them: the `.pt` files are PyTorch `state_dict`s for the classes in `src/models.py` (`build(name, in_dim, n_classes)` with `in_dim` 46, or 44 for the `binary44` ablation). The `.pkl` files are plain pickles of scikit-learn 1.5.1 / LightGBM 4.6.0 estimators saved under Python 3.12.

## Evaluation method (important)

- **Data.** From CICIoT2023's published 80/20 split we take a stratified subsample: at most 250,000 rows per attack class for training (1,529,980 rows) and 62,500 per class for a *balanced* test set (382,032 rows). We also keep a 499,112-row *natural* test sample with the original class mix (72.8% DDoS). Features are standardised with statistics from the training rows only. The 46 features contain no IP addresses, ports or timestamps.
- **Tasks.** Binary (benign vs attack) and 8-class (attack category).
- **Why two test sets.** On the natural mix, accuracy mostly measures how well a model spots DDoS. We report both so the published numbers can be reproduced *and* stress-tested.
- **Why MCC and false-positive rate.** CICIoMT2024 is 97.4% attacks, so a detector that calls everything an attack scores 97.4% accuracy. The Matthews correlation coefficient uses all four cells of the confusion matrix and exposes that; the benign false-positive rate shows the operational cost.
- **Model fidelity.** The four architectures were rebuilt from their papers' descriptions at the stated parameter budgets (none of the papers release code). Where a paper leaves a detail open, the choice is documented in `src/models.py`.
- **Cross-dataset.** CICIoMT2024 shares 45 of CICIoT2023's 46 features by name (it lacks `flow_duration` and `urg_count`). E2 zero-fills those two; the ablation retrains everything on the 44 shared features so no imputation is involved.

## Results so far

### E1 — same dataset (CICIoT2023), seed 42

Binary task:

| Model | Params | Natural-mix accuracy | Balanced accuracy | Balanced macro-F1 | MCC |
|---|---|---|---|---|---|
| MLP14k | 14,336 | 0.9934 | 0.8937 | 0.8399 | 0.7131 |
| KD student CNN | 1,940 | 0.9864 | 0.7840 | 0.7240 | 0.5531 |
| DL-BiLSTM | 11,432 | 0.9936 | 0.8971 | 0.8432 | 0.7161 |
| TinyMLP | 7,688 | 0.9935 | 0.8943 | 0.8404 | 0.7136 |
| Random Forest | – | 0.9973 | 0.9581 | 0.9289 | 0.8637 |
| LightGBM | – | 0.9974 | 0.9568 | 0.9271 | 0.8613 |

8-class task (balanced test): Random Forest 0.9222 accuracy / 0.8144 macro-F1, LightGBM 0.9339 / 0.8314; the four neural models 0.69–0.76 / 0.57–0.65. On the natural mix the trees reach 0.9926 and 0.9950 accuracy, matching the ~99% figures in the literature, while the neural models stay at 0.69–0.78 whether trained with weighted or plain cross-entropy — well below the 96–98% their papers report.

### E2 — different dataset (CICIoMT2024), models unchanged

| Model | Accuracy | MCC | Detection rate | Benign false-positive rate |
|---|---|---|---|---|
| MLP14k | 0.9763 | 0.3716 | 0.9968 | 0.7788 |
| KD student CNN | 0.9837 | 0.6275 | 0.9970 | 0.5080 |
| DL-BiLSTM | 0.9899 | **0.8157** | 0.9932 | **0.1324** |
| TinyMLP | 0.9726 | 0.1918 | 0.9963 | 0.9030 |
| Random Forest | 0.9737 | 0.0771 | 0.9999 | 0.9923 |
| LightGBM | 0.9743 | 0.1655 | 0.9997 | 0.9626 |

Accuracy is high for everyone because 97.4% of the data is attacks. The tree models are effectively saying "attack" for everything: Random Forest flags 99.2% of normal traffic. Retraining on the 44 shared features (no imputation) does not change this: Random Forest MCC 0.172, LightGBM 0.193, while the neural models reach 0.59–0.77.

### E3 — trained on both datasets together

Binary macro-F1 after training on a 50/50 mix: MLP14k 0.820 (CICIoT2023 test) / 0.970 (CICIoMT2024 test), DL-BiLSTM 0.810 / 0.968, TinyMLP 0.811 / 0.969, KD student 0.718 / 0.957. Transfer is restored at a cost of about 0.02–0.03 on the home dataset.

### E4 — synthetic and drifted traffic (8-class, macro-F1)

| Model | Real test | Copula-synthetic test | Noise σ=0.2 | 10% features dropped |
|---|---|---|---|---|
| MLP14k | 0.646 | 0.567 | 0.555 | 0.595 |
| DL-BiLSTM | 0.644 | 0.556 | 0.505 | 0.579 |
| TinyMLP | 0.640 | 0.564 | 0.538 | 0.593 |
| Random Forest | 0.814 | 0.805 | **0.459** | 0.791 |
| LightGBM | 0.831 | 0.797 | **0.292** | 0.755 |

Under feature noise the trees lose the most — the same brittleness that shows up cross-dataset.

### E5 — edge cost (CPU, batch size 1)

| Model | Params | FP32 size | INT8 size | Latency per flow | Macro-F1 FP32 → INT8 |
|---|---|---|---|---|---|
| MLP14k | 14,336 | 60 KB | 20 KB | 0.015 ms | 0.646 → 0.587 |
| KD student CNN | 1,940 | 11 KB | 8 KB | 0.088 ms | 0.569 → 0.509 |
| DL-BiLSTM | 11,432 | 50 KB | 22 KB | 0.173 ms | 0.644 → 0.538 |
| TinyMLP | 7,688 | 33 KB | 13 KB | 0.013 ms | 0.640 → 0.583 |

Size and speed are not a problem for any of these models. INT8 dynamic quantisation, however, costs 6–10 macro-F1 points at this scale. Pruning 50% of TinyMLP's weights costs only 0.013.

### Robustness sweep (seeds 13, 21, 42 × epochs 4, 10)

- Tree collapse on CICIoMT2024 holds on every seed: Random Forest MCC 0.069 ± 0.034, benign FPR 0.993 ± 0.005.
- DL-BiLSTM's cross-dataset result holds at 4 epochs: MCC 0.830 ± 0.013. The ordering of the other three neural models shifts with the seed.
- Training for 10 epochs instead of 4 improves in-distribution MCC by about 0.02 but destroys cross-dataset transfer for three of the four neural models (DL-BiLSTM 0.830 → 0.186). Only the knowledge-distilled student is unaffected.
- The 8-class gap is not closed by more epochs: MLP14k 0.754 ± 0.006 (4 epochs) and 0.767 ± 0.006 (10 epochs) balanced accuracy.

Full tables: [results/RESULTS.md](results/RESULTS.md) and [results/SWEEP.md](results/SWEEP.md). Interpretation: [results/FINDINGS.md](results/FINDINGS.md).

## Status and limitations

Done: literature survey, data pipeline, four rebuilt models plus two baselines, experiments E1–E5, the 44-feature ablation, and the 36-run seed/epoch sweep.

Not done yet:

- A third dataset (CIC-BCCC-NRC TabularIoTAttacks-2024, which unifies nine IoT datasets in one feature space).
- Adversarial-perturbation testing (purposeful small changes to attack traffic), which the synopsis promised.
- CTGAN as a second synthetic-data generator next to the copula.
- A real Raspberry Pi run. All latency and size numbers above come from an Apple M4 Pro CPU and should be read as relative, not absolute.
- The paper and the final report.

Other limits worth knowing:

- Cross-dataset tests are binary only, because the two datasets' attack taxonomies differ.
- The E1/E4/E5 tables are single-seed (42); only the headline claims were swept.
- The models are our re-implementations from the papers' descriptions, not the authors' code. Where a paper gives no layer widths we chose our own: the DL-BiLSTM paper reports 1,988 parameters on CICIoT2023 (and evaluates only its Recon and Mirai classes), while our version has 11,432 and is evaluated on all eight classes. Our 8-class numbers for the neural models are far below the papers' claims; the choices are documented in `src/models.py` so this can be checked.
- `lgbm_iso.py` exists because LightGBM segfaults in-process after PyTorch has used the Apple GPU. It is a workaround, not a feature.

## References

1. E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani, "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment," *Sensors* 23(13):5941, 2023.
2. S. Dadkhah, E. C. P. Neto, R. Ferreira, R. C. Molokwu, S. Sadeghi, A. A. Ghorbani, "CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT," *Internet of Things*, 2024.
3. J. Chandroth, J. Ali, "Lightweight MLP-Based Feature Extraction with Linear Classifier for Intrusion Detection System in Internet of Things," *Electronics* 15(8):1604, 2026. (MLP14k)
4. L.-H. Wang, Q. Dai, T. Du, L.-F. Chen, "Lightweight intrusion detection model based on CNN and knowledge distillation," *Applied Soft Computing*, 2024, doi:10.1016/j.asoc.2024.112118. (KD student CNN)
5. Z. Wang, H. Chen, S. Yang, X. Luo, D. Li, J. Wang, "A lightweight intrusion detection method for IoT based on deep learning and dynamic quantization," *PeerJ Computer Science* 9:e1569, 2023. (DL-BiLSTM)
6. F. Keskin, *reliable-edge-iot-ids* research artifact, GitHub, 2026. (TinyMLP)
7. M. Cantone, C. Marrocco, A. Bria, "On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection," arXiv:2402.10974, 2024.
8. M. Sarhan, S. Layeghy, M. Portmann, "Towards a Standard Feature Set for Network Intrusion Detection System Datasets," *Mobile Networks and Applications*, 2022.
9. G. AL Mukhaini, M. Anbar, S. Manickam, T. A. Al-Amiedy, A. Al Momani, "A systematic literature review of recent lightweight detection approaches leveraging machine and deep learning mechanisms in Internet of Things networks," *JKSU – Computer and Information Sciences* 36(1):101866, 2024.

The complete list, with dataset links and what we took from each paper, is in `research/literature_review.xlsx`.
