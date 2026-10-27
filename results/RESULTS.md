# Experimental Results — Lightweight IoT-NIDS Generalization Benchmark

Training data: CICIoT2023 stratified subsample (1,529,980 train / 382,032 test rows, classes: Benign, BruteForce, DDoS, DoS, Mirai, Recon, Spoofing, Web-based).


## E1 — In-distribution (8class)

`natural` = unmodified class mix (comparable to published numbers); `balanced` = per-class-capped stress test.

| model | nat acc | nat F1(macro) | bal acc | bal F1(macro) | bal MCC | bal FPR |
|---|---|---|---|---|---|---|
| MLP14k | 0.7651 | 0.5988 | 0.7602 | 0.6459 | 0.7262 | 0.0332 |
| KDStudentCNN | 0.6977 | 0.5042 | 0.6919 | 0.5691 | 0.6401 | 0.0442 |
| KDTeacherCNN | 0.6912 | 0.5330 | 0.6983 | 0.5867 | 0.6553 | 0.0422 |
| DLBiLSTM | 0.7763 | 0.5978 | 0.7584 | 0.6442 | 0.7238 | 0.0335 |
| TinyMLP | 0.7315 | 0.5861 | 0.7543 | 0.6397 | 0.7195 | 0.0341 |
| RandomForest | 0.9926 | 0.7764 | 0.9222 | 0.8144 | 0.9090 | 0.0107 |
| LightGBM | 0.9950 | 0.8029 | 0.9339 | 0.8314 | 0.9226 | 0.0090 |


## E1 — In-distribution (binary)

`natural` = unmodified class mix (comparable to published numbers); `balanced` = per-class-capped stress test.

| model | nat acc | nat F1(macro) | bal acc | bal F1(macro) | bal MCC | bal FPR |
|---|---|---|---|---|---|---|
| MLP14k | 0.9934 | 0.9352 | 0.8937 | 0.8399 | 0.7131 | 0.0348 |
| KDStudentCNN | 0.9864 | 0.8827 | 0.7840 | 0.7240 | 0.5531 | 0.0221 |
| KDTeacherCNN | 0.9888 | 0.9007 | 0.8065 | 0.7474 | 0.5892 | 0.0066 |
| DLBiLSTM | 0.9936 | 0.9364 | 0.8971 | 0.8432 | 0.7161 | 0.0440 |
| TinyMLP | 0.9935 | 0.9360 | 0.8943 | 0.8404 | 0.7136 | 0.0361 |
| RandomForest | 0.9973 | 0.9720 | 0.9581 | 0.9289 | 0.8637 | 0.0253 |
| LightGBM | 0.9974 | 0.9727 | 0.9568 | 0.9271 | 0.8613 | 0.0189 |


## E1 addendum — unweighted CE (papers' protocol), 8-class

| model | nat acc | nat F1(macro) | bal acc | bal F1(macro) |
|---|---|---|---|---|
| MLP14k | 0.7864 | 0.6318 | 0.8160 | 0.6676 |
| KDStudentCNN | 0.6894 | 0.5012 | 0.7248 | 0.5506 |
| DLBiLSTM | 0.7919 | 0.6200 | 0.8133 | 0.6591 |
| TinyMLP | 0.7936 | 0.6271 | 0.8126 | 0.6626 |


## E2 — Cross-dataset (exact models, unchanged weights)

External rows: 8,718,012, attack ratio 0.974, missing features: ['flow_duration', 'urg_count']

| model | acc | F1(macro) | MCC | DR | FPR |
|---|---|---|---|---|---|
| MLP14k | 0.9763 | 0.6593 | 0.3716 | 0.9968 | 0.7788 |
| KDStudentCNN | 0.9837 | 0.8032 | 0.6275 | 0.9970 | 0.5080 |
| DLBiLSTM | 0.9899 | 0.9072 | 0.8157 | 0.9932 | 0.1324 |
| TinyMLP | 0.9726 | 0.5718 | 0.1918 | 0.9963 | 0.9030 |
| RandomForest | 0.9737 | 0.5010 | 0.0771 | 0.9999 | 0.9923 |
| LightGBM | 0.9743 | 0.5291 | 0.1655 | 0.9997 | 0.9626 |


### E2 ablation — retrained on the 44 shared features (no imputation)

| model | CICIoT F1 | CICIoMT F1 | CICIoMT MCC | DR | FPR |
|---|---|---|---|---|---|
| MLP14k | 0.7884 | 0.8235 | 0.6654 | 0.9973 | 0.4658 |
| KDStudentCNN | 0.7172 | 0.7842 | 0.5850 | 0.9957 | 0.5295 |
| DLBiLSTM | 0.7946 | 0.7897 | 0.6102 | 0.9978 | 0.5494 |
| TinyMLP | 0.7889 | 0.8871 | 0.7748 | 0.9952 | 0.2480 |
| RandomForest | 0.9124 | 0.5343 | 0.1718 | 0.9995 | 0.9566 |
| LightGBM | 0.9074 | 0.5412 | 0.1928 | 0.9996 | 0.9491 |


## E3 — Combined-dataset training

| model | CICIoT2023 test F1 | external test F1 |
|---|---|---|
| MLP14k | 0.8203 | 0.9700 |
| KDStudentCNN | 0.7180 | 0.9567 |
| DLBiLSTM | 0.8102 | 0.9676 |
| TinyMLP | 0.8111 | 0.9686 |


## E4 — Synthetic & drifted data (8-class, macro-F1)

| model | synthetic test | noise_0.05 | noise_0.2 | dropout_10pct | scale_drift_0.2 |
|---|---|---|---|---|---|
| MLP14k | 0.5674 | 0.6285 | 0.5545 | 0.5948 | 0.6349 |
| KDStudentCNN | 0.5522 | 0.5624 | 0.4990 | 0.5244 | 0.5617 |
| KDTeacherCNN | 0.5633 | 0.5840 | 0.5182 | 0.5515 | 0.5750 |
| DLBiLSTM | 0.5562 | 0.6081 | 0.5052 | 0.5792 | 0.6009 |
| TinyMLP | 0.5637 | 0.6180 | 0.5375 | 0.5931 | 0.6254 |
| RandomForest | 0.8046 | 0.5119 | 0.4594 | 0.7914 | 0.5044 |
| LightGBM | 0.7965 | 0.3532 | 0.2924 | 0.7545 | 0.4224 |


**TSTR** (train on synthetic only, test on real):

| model | acc | F1(macro) | MCC |
|---|---|---|---|
| MLP14k | 0.6996 | 0.5940 | 0.6602 |
| TinyMLP | 0.6942 | 0.5858 | 0.6567 |


## E5 — Edge cost (CPU proxy; not a Raspberry Pi)

| model | params | FP32 MB | INT8 MB | lat b=1 (ms) | lat b=256 (ms/sample) | F1 | INT8 F1 |
|---|---|---|---|---|---|---|---|
| MLP14k | 14,336 | 0.060 | 0.020 | 0.015 | 0.0005 | 0.6459 | 0.5866 |
| KDStudentCNN | 1,940 | 0.011 | 0.008 | 0.088 | 0.0115 | 0.5691 | 0.5085 |
| DLBiLSTM | 11,432 | 0.050 | 0.022 | 0.173 | 0.0105 | 0.6442 | 0.5376 |
| TinyMLP | 7,688 | 0.033 | 0.013 | 0.013 | 0.0001 | 0.6397 | 0.5827 |
