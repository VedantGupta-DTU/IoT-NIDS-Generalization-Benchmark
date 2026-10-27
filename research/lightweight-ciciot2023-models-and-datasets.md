# Lightweight CICIoT2023 Models & Datasets for a Cross-Dataset Generalization Study

**Research survey for IT401 B.Tech Project — compiled 20 Oct 2026**

**Project idea being supported:** take the best *existing* lightweight IoT-NIDS models trained on CICIoT2023 from recent journal papers, reproduce them exactly, then evaluate (a) on their own training dataset and (b) on new / combined / synthetic datasets, to measure whether the published performance holds on dynamic data.

---

## 1. Why this study is worth doing (the documented gap)

The literature already strongly suggests published IoT-IDS numbers do **not** transfer, which is exactly the gap your evaluation would quantify for *specific named lightweight models*:

- Cantone, Marrocco & Bria benchmarked ML-NIDS across four datasets: average MCC fell from **~94% within-dataset to ~29% cross-dataset** ([On the Cross-Dataset Generalization of ML for Network Intrusion Detection, arXiv 2402.10974](https://arxiv.org/abs/2402.10974)). Hossain et al. (2026) reach the same conclusion with a different protocol ([Assessing Generalisation Capability of ML Models for Intrusion Detection, arXiv 2605.04407](https://arxiv.org/html/2605.04407)).
- Random Forest trained on UNSW-NB15 or TON_IoT scores 95–99% in-dataset but **under 40% accuracy** when tested on the other dataset.
- A dataset-centric federated IDS study across Edge-IIoTset, CICIoT2023 and TII-SSRC-23 found **up to 30 percentage-point macro-F1 loss** on unseen environments — and that **combining multiple datasets in training substantially restores transfer** ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12824137/)). This directly supports your "combine datasets" arm.
- A 2026 paper specifically on **lightweight** IIoT models attributes failure to shortcut features (IPs, ports acting as dataset-specific identifiers) ([Cross-Domain Generalization Failure in Lightweight IDS Models for IIoT, arXiv](https://arxiv.org/pdf/2607.00553)); a 2026 MDPI Electronics paper blames distributional drift across BoT-IoT / ToN-IoT / UNSW-NB15 ([Distributional Drift in IoT IDS](https://doi.org/10.3390/electronics15112307)).
- Precedent for your exact dataset pairing: a Siamese-network zero-day IDS evaluates transfer **between CICIoT2023 and CIC-BCCC-NRC TabularIoTAttacks-2024** ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405959526000731)).

**Novelty angle for your paper:** nobody appears to have taken the *published, named, lightweight* CICIoT2023 models (the ones below), reproduced them faithfully, and subjected them all to the same cross-dataset / combined-dataset / synthetic-data protocol with edge metrics (latency, model size) reported alongside. That is a publishable survey + benchmark contribution.

---

## 2. Candidate lightweight models trained on CICIoT2023

Ranked roughly by how reproducible + citable they are for your study.

### Tier A — best candidates (simple architecture, stated sizes, strong venues)

| # | Model / paper | Venue, year | Reported on CICIoT2023 | Size / lightweight claim | Code? |
|---|---|---|---|---|---|
| 1 | **Lightweight MLP + linear classifier** ([MDPI Electronics](https://www.mdpi.com/2079-9292/15/8/1604)) | Electronics 15(8):1604, 2026 | **98.45% acc** (also 99.85% CICIDS2017, 99.21% NSL-KDD) | ~**14.6k params**, ~29k FLOPs | Not stated — but 2-layer MLP + softmax is trivially re-implementable from the paper |
| 2 | **CNN + knowledge distillation** (student model) ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494624008925), [ResearchGate](https://www.researchgate.net/publication/383360243_Lightweight_intrusion_detection_model_based_on_CNN_and_knowledge_distillation)) | Applied Soft Computing, 2024 | ~**96% acc** | student has **1,920 params** | Not stated; tiny CNN is re-implementable |
| 3 | **DL-BiLSTM + dynamic quantization** ([PMC / PeerJ CS](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10557502/)) | PeerJ Computer Science, 2023 | evaluated on CICIoT2023 (+ NSL-KDD) | INT8 dynamic quantization for edge | Architecture fully specified; closest to your original synopsis (BiLSTM), so your existing plan transfers |
| 4 | **TinyMLP + pruning + INT8** in `reliable-edge-iot-ids` ([GitHub](https://github.com/fesihkeskin/reliable-edge-iot-ids)) | companion artifact, manuscript in submission | CICIoT2023 primary + Edge-IIoTset validation; binary & multiclass, **zero-day leave-one-attack-out**, calibration, compression | TinyMLP + LR/RF/LightGBM baselines | **Yes — full MIT-licensed pipeline**: preprocessing, configs, seeds, notebooks. The only end-to-end reproducible codebase found |

### Tier B — strong results, heavier or less specified

| # | Model / paper | Venue, year | Reported on CICIoT2023 | Notes |
|---|---|---|---|---|
| 5 | **Two-stage lightweight edge framework** ([ScienceDirect](https://www.sciencedirect.com/org/science/article/pii/S1546221826003541)) | 2026 | **99.67% acc** on full CICIoT2023, ~64.5% train/test time reduction, Raspberry Pi validated | Highest headline number found; verify stage details |
| 6 | **BGL-RID hybrid + hybrid feature selection** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13444135/)) | 2025/26 | **98.88% acc**, optimized memory/time | Edge-focused |
| 7 | **CNN-LSTM energy-aware fog IDS** ([CMC 80(3)](https://www.techscience.com/cmc/v80n3/57889/html), [PDF](https://www.sciopen.com/article_pdf/1851191271469285377.pdf)) | CMC, 2024 | CICIoT2023 + 2 older datasets; latency & energy measured on Raspberry Pi 3 | Good template for your edge-metrics protocol |
| 8 | **GAT-BiGRU, TinyML + federated** ([Nature Scientific Reports](https://www.nature.com/articles/s41598-026-50690-0), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13269888/)) | Sci Reports, 2026 | **F1 0.89** on CICIoT2023 (honest number!) | Shows what realistic F1 looks like vs. 99%+ claims |
| 9 | **Hybrid CNN + ConvNeXt-Tiny** ([arXiv](https://arxiv.org/pdf/2509.06202)) | preprint | 99.58–99.99% across UNSW-NB15, CIC-IDS2018, CICIoT2023 | Multi-dataset already; preprint status |
| 10 | **Lightweight TCN (TFLite) on Raspberry Pi 4** ([arXiv](https://arxiv.org/pdf/2608.00118)) | preprint | 98.43% acc on CICIoT2023 DDoS-UDP subset, ~0.7 ms/sample | Subset-only evaluation |

Also useful as supporting citations: the [PeerJ CS dimensionality-reduction comparison on CICIoT2023 + Edge-IIoTset with Raspberry Pi benchmarks](https://peerj.com/articles/cs-3553/), the SLR on lightweight IoT detection approaches already in your synopsis (AL Mukhaini et al., JKSU-CIS 36(1):101866, 2024), a [quantization-aware training study (95 KB → 12.6 KB model)](https://journal.its.ac.id/index.php/ijcsam/article/download/8786/7987/26025), a [neuro-symbolic lightweight edge IDS](https://link.springer.com/article/10.1007/s44397-026-00047-z), and a [SHAP-guided pruning + Kronecker distillation IDS](https://arxiv.org/pdf/2512.19488).

**Practical reality check:** almost none of the journal papers release code. Your reproduction set should be: **#1 (MLP), #2 (KD-CNN student), #3 (DL-BiLSTM), #4 (TinyMLP repo)** — all four are small enough to re-implement exactly from the papers in Keras/PyTorch within days, span four model families (MLP / CNN / RNN / distilled), and #4 gives you a known-good preprocessing pipeline for CICIoT2023 to build on. Add RF/LightGBM from #4 as classical baselines.

---

## 3. Datasets for the generalization arm

### The key enabling resource: CIC-BCCC-NRC TabularIoTAttacks-2024

[UNB page](https://www.unb.ca/cic/datasets/tabular-iot-attack-2024.html) · [BCCC page](https://www.yorku.ca/research/bccc/tag/tabular-iot-attack-dataset-cic-bccc-nrc-tabulariotattack-2024/)

An aggregation of **nine** IoT attack datasets re-extracted into **one consistent tabular feature space** (CICFlowMeter-style flow features), 1M+ labeled records: IoT Network Intrusion (2019), MQTT-IoT-IDS-2020, TON_IoT (2021), UQ IoT 2022, CIC IoT 2022, Edge-IIoTset (2022), **CICIoT2023**, ACI-IoT-2023, CICIoMT2024.

This solves your biggest methodological problem (feature-space mismatch, see §5) **and** gives you the "combined multi-dataset" condition for free — train on the CICIoT2023 slice, test on the other eight slices, then train on combinations. It is downloadable from cicresearch.ca without special access.

### Individual datasets worth using directly

| Dataset | Year | Domain | Why it's useful for you |
|---|---|---|---|
| [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | 2023 | Consumer/smart-home IoT, 105 devices, 33 attacks in 7 classes | The home dataset all candidate models trained on |
| [CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) | 2024 | Medical IoT, 40 devices (25 real), 18 attacks, Wi-Fi/MQTT/Bluetooth | Same CIC testbed methodology as CICIoT2023 → the cleanest "same features, new domain" transfer test ([paper](https://www.researchgate.net/publication/383572709_CICIoMT2024_A_benchmark_dataset_for_multi-protocol_security_assessment_in_IoMT)) |
| [Edge-IIoTset](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) | 2022 | Industrial IoT, 61 features, ~14–17 attack types | Most-used second dataset in the lightweight-IDS literature (used by candidates #4, #8) |
| [ACI-IoT-2023](https://ieee-dataport.org/documents/aci-iot-network-traffic-dataset-2023) | 2023 | Army Cyber Institute home-IoT testbed; NetFlow + payload CSVs + 32 GB PCAP; ~743k flows, recon + DoS heavy | Independent (non-CIC) lab → stronger generalization claim; DOI 10.21227/qacj-3x32 |
| TON_IoT / Bot-IoT / [NF-series (NF-ToN-IoT-v2 etc.)](https://www.cyber.uq.edu.au/node/824) | 2019–21 | Mixed IoT/IIoT | The NetFlow v2 versions share a standard 43-feature schema across datasets ([Sarhan et al.](https://arxiv.org/pdf/2101.11315v1)) — an alternative unified feature space to TabularIoTAttacks |
| [BCCC-IoT-MQTT-IDS-2025](https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/bccc-mqtt-augmented-ids-2026-developed-and-designed-for-large-language-models-llm/) | 2025 | MQTT-focused IoT | Newest; good "never-seen protocol mix" test |
| CICIoV2024 | 2024 | In-vehicle (CAN) | Probably too far out-of-domain; mention in survey, skip in experiments |

### Synthetic-data arm

- **CTGAN** is the standard tabular generator in this space: used for rare-attack augmentation on CSE-CIC-IDS2018 (99.9% overall, [arXiv](https://arxiv.org/html/2502.06031v1)) and DoS/DDoS on Bot-IoT ([Sensors 2023](https://doi.org/10.3390/s23125644)).
- A conditional GAN + whale optimization (CGAN-AWOA) was tested **on CICIoT2023 itself** — 98.59% accuracy ([IJASCA](https://ijasca.zuj.edu.jo/index.php/IJASCA/article/download/71/25/232)).
- Cautionary citation: Conv-GAN augmentation *reduced* performance when synthetic quality was poor ([thesis](https://norma.ncirl.ie/8679/1/tejassandeepbafna.PDF)) — worth reporting both directions.
- Practical recipe: fit CTGAN (via the `sdv` Python library) per attack class on CICIoT2023, generate (i) augmented rare classes and (ii) a fully synthetic test set; also create **perturbed/drifted** variants (feature noise, time-shifted flows) as your "dynamic data" condition.

---

## 4. The feature-space problem (plan around this early)

CICIoT2023's native CSVs use **46 features from the CIC's own custom (DPKT-based) extractor — not CICFlowMeter**. Other datasets ship different schemas (Edge-IIoTset: 61 features; ACI-IoT: NetFlow). A model trained on CICIoT2023's 46 columns cannot be *directly* fed another dataset's CSV. No ready-made open-source re-implementation of the exact 46-feature extractor surfaced in the search ([comparison of extraction tools](https://arxiv.org/pdf/2501.13004)); community Python ports of CICFlowMeter exist but output a different set.

Three workable strategies, in order of practicality:

1. **Unified-schema route (recommended):** run everything on **TabularIoTAttacks-2024**, which contains CICIoT2023 re-extracted alongside eight other datasets in one schema. Train each candidate *architecture* on the CICIoT2023 slice (verifying you can reproduce ballpark published accuracy), then test cross-slice. This is the published precedent ([Siamese zero-day paper](https://www.sciencedirect.com/science/article/pii/S2405959526000731)).
2. **Common-subset route:** map the intersection of features (duration, byte/packet counts, flags, rates…) across native datasets, retrain on the subset. Keeps native data but weakens "exact model" claims.
3. **Re-extraction route:** run the CIC feature extraction code over the other datasets' PCAPs so every dataset gets the exact 46 CICIoT2023 features. Strongest claim, most work (32 GB+ PCAPs); only do this for one secondary dataset (e.g., ACI-IoT-2023) if time allows.

Also replicate the hygiene the generalization literature insists on: **drop IPs, ports, and timestamps** from features (shortcut/leakage), report **macro-F1 and MCC** (not just accuracy — CICIoT2023 is heavily imbalanced), and use the same splits/seeds across all models.

---

## 5. Suggested experimental matrix

Models (reproduced exactly): `MLP-14k` · `KD-CNN-1.9k` · `DL-BiLSTM` · `TinyMLP` (+ `RF`, `LightGBM` baselines)

| Condition | Data | Question answered |
|---|---|---|
| E1 In-distribution | CICIoT2023 (held-out test) | Do we reproduce the published numbers? |
| E2 Cross-dataset | → CICIoMT2024, Edge-IIoTset, ACI-IoT-2023 slices | Does performance hold on unseen environments? |
| E3 Combined training | CICIoT2023 + 1–2 others → test on a held-out dataset | Does multi-dataset training restore transfer? |
| E4 Synthetic/dynamic | CTGAN-generated + perturbed CICIoT2023 | Robustness to synthetic/drifted traffic |
| E5 Edge cost | all models, quantized | Latency / model size / memory on Raspberry Pi (keeps your synopsis's edge angle) |

Metrics everywhere: accuracy, precision, recall, macro-F1, MCC, FPR + (E5) model size, ms/sample.

---

## 6. Reference list (key sources)

**Models:** [MDPI Electronics MLP](https://www.mdpi.com/2079-9292/15/8/1604) · [ASOC CNN-KD](https://www.sciencedirect.com/science/article/abs/pii/S1568494624008925) · [PeerJ DL-BiLSTM + quantization](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10557502/) · [reliable-edge-iot-ids repo](https://github.com/fesihkeskin/reliable-edge-iot-ids) · [Two-stage edge framework](https://www.sciencedirect.com/org/science/article/pii/S1546221826003541) · [BGL-RID](https://pmc.ncbi.nlm.nih.gov/articles/PMC13444135/) · [CMC CNN-LSTM](https://www.techscience.com/cmc/v80n3/57889/html) · [Sci Reports GAT-BiGRU](https://www.nature.com/articles/s41598-026-50690-0) · [CNN+ConvNeXt-Tiny](https://arxiv.org/pdf/2509.06202) · [TCN on Pi 4](https://arxiv.org/pdf/2608.00118) · [PeerJ dim-reduction on Pi](https://peerj.com/articles/cs-3553/)

**Generalization:** [Cantone et al. 2024](https://arxiv.org/abs/2402.10974) · [Hossain et al. 2026](https://arxiv.org/html/2605.04407) · [Distributional drift, Electronics](https://doi.org/10.3390/electronics15112307) · [Lightweight IIoT cross-domain failure](https://arxiv.org/pdf/2607.00553) · [Federated dataset-centric eval](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12824137/) · [Siamese zero-day cross-dataset](https://www.sciencedirect.com/science/article/pii/S2405959526000731) · [Sarhan et al. standard feature sets](https://arxiv.org/pdf/2101.11315v1), [follow-up](https://arxiv.org/pdf/2104.07183)

**Datasets:** [TabularIoTAttacks-2024](https://www.unb.ca/cic/datasets/tabular-iot-attack-2024.html) · [CICIoMT2024](https://www.researchgate.net/publication/383572709_CICIoMT2024_A_benchmark_dataset_for_multi-protocol_security_assessment_in_IoMT) · [ACI-IoT-2023](https://ieee-dataport.org/documents/aci-iot-network-traffic-dataset-2023) · [UQ NetFlow datasets](https://www.cyber.uq.edu.au/node/824) · [CIC dataset index](https://www.unb.ca/cic/datasets/)

**Synthetic:** [CTGAN two-stage](https://arxiv.org/html/2502.06031v1) · [CGAN-AWOA on CICIoT2023](https://ijasca.zuj.edu.jo/index.php/IJASCA/article/download/71/25/232) · [CTGAN Bot-IoT, Sensors](https://doi.org/10.3390/s23125644) · [Conv-GAN counterexample](https://norma.ncirl.ie/8679/1/tejassandeepbafna.PDF)

---

*Caveats: several 2026-dated items are very recent — verify venue status and exact numbers against the full texts before citing in the paper. Accuracy figures above come from different setups (binary vs 34-class, subsets vs full data) and are not directly comparable to each other.*
