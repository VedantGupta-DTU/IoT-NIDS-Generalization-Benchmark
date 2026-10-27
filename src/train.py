"""Training loops: standard cross-entropy and knowledge distillation."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import SEED, set_seed


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(X, y, batch_size, shuffle, seed=SEED):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def train_model(model, Xtr, ytr, n_classes, epochs=4, batch_size=512, lr=1e-3,
                weighted=True, teacher=None, kd_alpha=0.7, kd_T=4.0, verbose=True):
    """Train with CE (optionally class-weighted). If `teacher` is given, use
    knowledge distillation: loss = alpha*KL(student_T || teacher_T) + (1-alpha)*CE."""
    set_seed(SEED)
    device = get_device()
    model = model.to(device)
    if teacher is not None:
        teacher = teacher.to(device).eval()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = class_weights(ytr, n_classes).to(device) if weighted else None
    ce = nn.CrossEntropyLoss(weight=w)
    loader = _loader(Xtr, ytr, batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(xb)
                kd = F.kl_div(
                    F.log_softmax(logits / kd_T, dim=1),
                    F.softmax(tl / kd_T, dim=1),
                    reduction="batchmean",
                ) * (kd_T ** 2)
                loss = kd_alpha * kd + (1 - kd_alpha) * loss
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        if verbose:
            print(f"  epoch {ep + 1}/{epochs} loss={total / n:.4f}")
    return model


@torch.no_grad()
def predict(model, X, batch_size=4096, device=None):
    device = device or get_device()
    model = model.to(device).eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)
