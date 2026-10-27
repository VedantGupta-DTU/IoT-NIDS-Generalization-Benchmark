"""E5 — Edge-cost proxy: parameters, serialized size (FP32 and dynamic INT8),
and CPU inference latency for every neural model.

Measured on this machine's CPU as a proxy for edge hardware (a Raspberry Pi
run would use the same artifacts via ONNX/ExecuTorch; absolute numbers differ,
ranking is indicative). Latency is the median of repeated timed batches after
warmup, reported per-sample for batch sizes 1 and 256.
"""
import copy
import io
import os
import time

import numpy as np
import torch
import torch.nn.utils.prune as prune

from common import print_table, save_results, set_seed
from data import load_meta, load_processed
from models import param_count
from run_e1 import NN_MODELS, train_all
from train import predict
from common import compute_metrics


def serialized_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def latency_ms(model: torch.nn.Module, X: np.ndarray, batch: int, reps: int = 30) -> float:
    model = model.cpu().eval()
    xb = torch.from_numpy(X[:batch])
    with torch.no_grad():
        for _ in range(5):
            model(xb)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(xb)
            times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000 / batch)


def main():
    set_seed()
    torch.backends.quantized.engine = "qnnpack"  # ARM build defaults to none
    meta = load_meta()
    Xtr, ytr8, _ = load_processed("train")
    Xte, yte8, _ = load_processed("test")
    n8 = len(meta["classes"])
    models = train_all("8class", Xtr, ytr8, n8)

    results, rows = {}, []
    for name in NN_MODELS:
        fp32 = copy.deepcopy(models[name]).cpu().eval()
        entry = {
            "params": param_count(fp32),
            "fp32_mb": serialized_mb(fp32),
            "lat_b1_ms": latency_ms(fp32, Xte, 1),
            "lat_b256_ms_per_sample": latency_ms(fp32, Xte, 256),
        }

        q = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(fp32), {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
        )
        entry["int8_mb"] = serialized_mb(q)
        entry["int8_lat_b256_ms_per_sample"] = latency_ms(q, Xte, 256)
        yp = predict(q, Xte, device=torch.device("cpu"))
        entry["int8_f1_macro"] = compute_metrics(yte8, yp, n8)["f1_macro"]
        yp32 = predict(fp32, Xte, device=torch.device("cpu"))
        entry["fp32_f1_macro"] = compute_metrics(yte8, yp32, n8)["f1_macro"]

        if name == "TinyMLP":  # pruning arm from the reliable-edge-iot-ids recipe
            pr = copy.deepcopy(fp32)
            for mod in pr.modules():
                if isinstance(mod, torch.nn.Linear):
                    prune.l1_unstructured(mod, "weight", amount=0.5)
                    prune.remove(mod, "weight")
            ypp = predict(pr, Xte, device=torch.device("cpu"))
            entry["pruned50_f1_macro"] = compute_metrics(yte8, ypp, n8)["f1_macro"]
            nz = sum(int((m.weight != 0).sum()) for m in pr.modules() if isinstance(m, torch.nn.Linear))
            entry["pruned50_nonzero_params"] = nz

        results[name] = entry
        rows.append([
            name, f"{entry['params']:,}", f"{entry['fp32_mb']:.3f}", f"{entry['int8_mb']:.3f}",
            f"{entry['lat_b1_ms']:.3f}", f"{entry['lat_b256_ms_per_sample']:.4f}",
            f"{entry['fp32_f1_macro']:.4f}", f"{entry['int8_f1_macro']:.4f}",
        ])

    save_results("e5_edge_cost", results)
    print_table(rows, ["model", "params", "fp32MB", "int8MB", "b1 ms", "b256 ms/smp", "F1", "int8F1"])


if __name__ == "__main__":
    main()
