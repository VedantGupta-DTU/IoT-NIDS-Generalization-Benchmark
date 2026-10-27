"""Reproductions of the four lightweight CICIoT2023 architectures (PyTorch).

Each model is re-implemented from its paper's description, sized to match the
stated parameter budget. Where a paper leaves a detail unspecified, the choice
is documented inline — these are faithful re-implementations, not the authors'
original code (none of the papers released code).

1. MLP14k    — "Lightweight MLP-based feature extraction with linear
               classifier" (Electronics 15(8):1604, 2026). Two-layer MLP
               embedding + linear softmax head, ~14.6k params target.
2. KDStudentCNN / KDTeacherCNN — "Lightweight intrusion detection model based
               on CNN and knowledge distillation" (Applied Soft Computing,
               2024). Student sized to ~1.9k params, trained with KD.
3. DLBiLSTM  — "A lightweight intrusion detection method for IoT based on deep
               learning and dynamic quantization" (PeerJ CS, 2023). DNN
               front-end feeding a BiLSTM, dynamic INT8 quantization for edge.
4. TinyMLP   — compact MLP in the style of the reliable-edge-iot-ids research
               artifact (TinyMLP + pruning + dynamic INT8).
"""
import torch
import torch.nn as nn


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class MLP14k(nn.Module):
    """46 -> 140 -> 52 embedding, linear head. ~14.3k params for 8 classes
    (paper states 14,662 for CICIoT2023; exact hidden sizes are not given, so
    widths were chosen to land on the stated budget)."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 140), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(140, 52), nn.ReLU(),
            nn.Linear(52, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class KDTeacherCNN(nn.Module):
    """1D-CNN teacher: two conv blocks + FC, trained first, then distilled."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))


class KDStudentCNN(nn.Module):
    """Tiny 1D-CNN student: 1,940 params for 8 classes (paper states 1,920;
    conv widths chosen to land on the budget)."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16, 60), nn.ReLU(),
            nn.Linear(60, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))


class DLBiLSTM(nn.Module):
    """DNN front-end (46 -> 96) whose embedding is read as a 12-step sequence
    of 8-dim vectors by a single BiLSTM layer (hidden 24 per direction).
    The paper stacks DNN layers before a BiLSTM; the exact reshape is our
    documented choice for flat flow records."""

    STEPS, DIMS = 12, 8

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.dnn = nn.Sequential(nn.Linear(in_dim, self.STEPS * self.DIMS), nn.ReLU())
        self.lstm = nn.LSTM(self.DIMS, 24, batch_first=True, bidirectional=True)
        self.head = nn.Linear(48, n_classes)

    def forward(self, x):
        h = self.dnn(x).view(-1, self.STEPS, self.DIMS)
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :])


class TinyMLP(nn.Module):
    """46 -> 64 -> 64 -> n, ~7.6k params; prune+quantize targets for E5."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


MODEL_REGISTRY = {
    "MLP14k": MLP14k,
    "KDStudentCNN": KDStudentCNN,
    "DLBiLSTM": DLBiLSTM,
    "TinyMLP": TinyMLP,
}


def build(name: str, in_dim: int, n_classes: int) -> nn.Module:
    return MODEL_REGISTRY[name](in_dim, n_classes)
