import json
import numpy as np
import torch
from collections import deque
from model import PowerLSTM
import os
import matplotlib.pyplot as plt
from metrics import compute_metrics


# ----------------------------
# Config: exogenous features only
# ----------------------------
EXOG_KEYS = [
    "fps",
    "gs_fps",
    "num_pixels",
    "num_splats",
    "gpu_sm_util_percent",
    "gpu_mem_util_percent",
    "gpu_memory_used_MB",
]

# ----------------------------
# Helpers
# ----------------------------
def exog_vec(rec, keys=EXOG_KEYS) -> np.ndarray:
    return np.array([float(rec[k]) for k in keys], dtype=np.float32)

def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-9)

def denorm_power(p_norm: float, power_mean: float, power_std: float) -> float:
    return p_norm * (power_std + 1e-9) + power_mean

# ----------------------------
# Exog-only inference
# ----------------------------
@torch.no_grad()
def predict_power_exog_only(
    records,
    exog_model_ckpt,
    seq_len,
    exog_mean,
    exog_std,
    power_mean,
    power_std,
    device="cpu",
    hidden_dim=32,
    num_layers=1,
    dropout=0.7,
):
    """
    Exog-only inference for every record.
    Builds a progressive sequence buffer up to seq_len, then sliding window.
    Returns predicted power in WATTS, masked with None for first seq_len records.
    """
    exog_model = PowerLSTM(
        input_dim=len(EXOG_KEYS),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    exog_model.load_state_dict(torch.load(exog_model_ckpt, map_location="cpu"))
    exog_model = exog_model.to(device).eval()

    buf = deque(maxlen=seq_len)
    preds_watts = []

    for rec in records:
        x = normalize(exog_vec(rec), exog_mean, exog_std)
        buf.append(x)

        # progressive window: grows from (1, D) up to (seq_len, D)
        X_np = np.array(list(buf), dtype=np.float32)          # [T, D], T<=seq_len
        X = torch.tensor(X_np[None, :, :], dtype=torch.float32, device=device)  # [1, T, D]

        p_norm = exog_model(X).squeeze().item()
        p_watts = denorm_power(p_norm, power_mean, power_std)
        preds_watts.append(float(p_watts))

    # mask first seq_len predictions (no output during warm-up)
    masked = [None] * min(seq_len, len(preds_watts)) + preds_watts[seq_len:]
    return masked[:len(records)]

# ----------------------------
# Plot normalized predicted vs actual
# ----------------------------
def plot_pred_vs_actual_power_normalized(
    records,
    preds,
    power_mean,
    power_std,
    out_path="outputs/pred_vs_actual_norm.png",
    title="GPU Power (Normalized): Predicted vs Actual",
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    actual_norm = []
    pred_norm = []

    for r, p in zip(records, preds):
        if p is None:
            continue

        a = (float(r["gpu_power_watts"]) - power_mean) / (power_std + 1e-9)
        pn = (float(p) - power_mean) / (power_std + 1e-9)

        actual_norm.append(a)
        pred_norm.append(pn)

    actual_norm = np.array(actual_norm, dtype=np.float32)
    pred_norm = np.array(pred_norm, dtype=np.float32)

    x = np.arange(len(actual_norm))

    plt.figure()
    plt.plot(x, actual_norm, label="Actual (normalized)")
    plt.plot(x, pred_norm, label="Predicted (normalized)")
    plt.title(title)
    plt.xlabel("Timestep (after warm-up)")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved normalized plot to: {out_path}")
    return actual_norm, pred_norm

# ----------------------------
# (Optional) Metrics after warm-up
# ----------------------------
def mae_rmse_after_warmup(records, preds, power_mean, power_std):
    actual = []
    pred = []
    print(len(records), len(preds))
    for r, p in zip(records, preds):
        if p is None:
            continue
        actual.append(float(r["gpu_power_watts"]))
        pred.append(float(p))
    print(len(actual), len(pred))

    actual = np.array(actual, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)

    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    huber = compute_metrics(pred, actual, delta=2)

    # normalized MAE/RMSE (optional)
    actual_n = (actual - power_mean) / (power_std + 1e-9)
    pred_n = (pred - power_mean) / (power_std + 1e-9)
    mae_n = float(np.mean(np.abs(pred_n - actual_n)))
    rmse_n = float(np.sqrt(np.mean((pred_n - actual_n) ** 2)))
    huber_n = compute_metrics(pred_n, actual_n, delta=2)

    return {"mae_watts": mae, "rmse_watts": rmse, "mae_norm": mae_n, "rmse_norm": rmse_n, "huber_loss": huber, "huber_norm": huber_n}

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Load records
    records = []
    with open("../../dataset/validation_data_1/merged_log.json") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except:
                pass

    # Paths
    # exog_model_ckpt = "checkpoints_seq_len_120_no_power_lag/best_model_epoch_86_loss_0.027480796795745176.pt"
    exog_model_ckpt = "best_models/best_model_epoch_67_loss_0.45593543338278025.pt"
    norm_meta = json.load(open("../../dataset/normalization.json"))

    # IMPORTANT: exog stats must be length 7 (match EXOG_KEYS)
    exog_mean = np.array(norm_meta["feature_mean"][:len(EXOG_KEYS)], dtype=np.float32)
    exog_std  = np.array(norm_meta["feature_std"][:len(EXOG_KEYS)], dtype=np.float32)

    power_mean = float(norm_meta["power_mean"])
    power_std  = float(norm_meta["power_std"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Slice some window to test
    test_records = records

    # Predict using exog-only model
    preds = predict_power_exog_only(
        records=test_records,
        exog_model_ckpt=exog_model_ckpt,
        seq_len=120,
        exog_mean=exog_mean,
        exog_std=exog_std,
        power_mean=power_mean,
        power_std=power_std,
        device=device,
        hidden_dim=32,
        num_layers=1,
        dropout=0.7,
    )

    # Plot normalized
    plot_pred_vs_actual_power_normalized(
        records=test_records,
        preds=preds,
        power_mean=power_mean,
        power_std=power_std,
        out_path="outputs/pred_vs_actual_normalized_exog_only.png",
        title="GPU Power (Normalized): Exog-only Predicted vs Actual",
    )

    # Metrics
    metrics = mae_rmse_after_warmup(test_records, preds, power_mean, power_std)
    print("Metrics after warm-up:", metrics)
