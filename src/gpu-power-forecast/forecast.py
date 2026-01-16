import json
import numpy as np
import torch
from collections import deque
from model import PowerLSTM
import os
import matplotlib.pyplot as plt

EXOG_KEYS = [
    "fps",
    "gs_fps",
    "num_pixels",
    "num_splats",
    "gpu_sm_util_percent",
    "gpu_mem_util_percent",
    "gpu_memory_used_MB",
]

def exog_vec(rec, keys=EXOG_KEYS) -> np.ndarray:
    return np.array([float(rec[k]) for k in keys], dtype=np.float32)

def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-9)

def denorm_power(p_norm: float, power_mean: float, power_std: float) -> float:
    return p_norm * (power_std + 1e-9) + power_mean

def pad_window_to_seq_len(window_list, seq_len: int, pad_mode="repeat_first"):
    """
    window_list: list[np.ndarray] length T<=seq_len, each shape [D]
    returns np.ndarray shape [seq_len, D]
    pad_mode:
      - "repeat_first": pad missing rows by repeating first row
      - "repeat_last": pad missing rows by repeating last row
    """
    T = len(window_list)
    assert T >= 1, "Need at least 1 record to build a window"
    D = window_list[0].shape[0]

    out = np.empty((seq_len, D), dtype=np.float32)
    out[:T] = np.stack(window_list, axis=0)

    if T < seq_len:
        if pad_mode == "repeat_first":
            out[T:] = out[0]
        elif pad_mode == "repeat_last":
            out[T:] = out[T-1]
        else:
            raise ValueError("pad_mode must be 'repeat_first' or 'repeat_last'")

    return out


def power_lag_feats_from_hist(power_hist, power_lags, power_mean, power_std):
    max_lag = max(power_lags)
    if len(power_hist) < max_lag:
        return None
    feats = []
    for l in power_lags:
        p_lag = power_hist[-l]  # watts
        feats.append((p_lag - power_mean) / (power_std + 1e-9))
    return np.array(feats, dtype=np.float32)

@torch.no_grad()
def stage_a_exog_only_for_max_lag(
    records,
    exog_model,
    seq_len,
    max_lag,
    exog_mean,
    exog_std,
    power_mean,
    power_std,
    device="cpu",
    pad_mode="repeat_first",
):
    exog_model = exog_model.to(device).eval()

    buf = deque(maxlen=seq_len)
    power_hist = deque()
    preds = []

    steps = min(max_lag, len(records))
    for i in range(steps):
        x = normalize(exog_vec(records[i]), exog_mean, exog_std)
        buf.append(x)

        # X_np = pad_window_to_seq_len(list(buf), seq_len, pad_mode=pad_mode)
        X_np = np.array(list(buf), dtype=np.float32)
        X = torch.tensor(X_np[None, :, :], dtype=torch.float32, device=device)

        p_norm = exog_model(X).squeeze().item()
        p_watts = denorm_power(p_norm, power_mean, power_std)

        preds.append(float(p_watts))
        power_hist.append(float(p_watts))

    return preds, power_hist


@torch.no_grad()
def stage_b_lag_model_rest(
    records_rest,
    lag_model,
    seq_len,
    power_hist_seed,
    power_lags,
    exog_mean,
    exog_std,
    power_mean,
    power_std,
    device="cpu",
    pad_mode="repeat_first",
):
    lag_model = lag_model.to(device).eval()

    power_hist = deque(power_hist_seed)   # copy
    feat_buf = deque(maxlen=seq_len)
    preds = []
    for rec in records_rest:
        x = normalize(exog_vec(rec), exog_mean, exog_std)
        lagf = power_lag_feats_from_hist(power_hist, power_lags, power_mean, power_std)
        if lagf is None:
            # shouldn't happen if stage A ran for max_lag, but just in case:
            p_watts = power_hist[-1] if len(power_hist) else power_mean
            preds.append(float(p_watts))
            power_hist.append(float(p_watts))
            continue

        feat_t = np.concatenate([x, lagf], axis=0)
        feat_buf.append(feat_t)

        # X_np = pad_window_to_seq_len(list(feat_buf), seq_len, pad_mode=pad_mode)
        X_np = np.array(feat_buf, dtype=np.float32)
        X = torch.tensor(X_np[None, :, :], dtype=torch.float32, device=device)

        p_norm = lag_model(X).squeeze().item()
        p_watts = denorm_power(p_norm, power_mean, power_std)

        preds.append(float(p_watts))
        power_hist.append(float(p_watts))

    return preds


def predict_power_two_stage_only_till_max_lag(
    records,
    exog_model_ckpt,
    lag_model_ckpt,
    seq_len,
    power_lags=(1, 2),
    exog_mean=None,
    exog_std=None,
    power_mean=0.0,
    power_std=1.0,
    device="cpu",
    pad_mode="repeat_first",
):
    max_lag = max(power_lags)

    exog_model = PowerLSTM(
        input_dim=7,
        hidden_dim=32,
        num_layers=1,
        dropout=0.7
    )
    exog_model.load_state_dict(torch.load(exog_model_ckpt, map_location="cpu"))
    exog_model.eval()

    lag_model = PowerLSTM(
        input_dim=9,
        hidden_dim=32,
        num_layers=1,
        dropout=0.3
    )
    lag_model.load_state_dict(torch.load(lag_model_ckpt, map_location="cpu"))
    lag_model.eval()

    preds_a, power_hist = stage_a_exog_only_for_max_lag(
        records=records,
        exog_model=exog_model,
        seq_len=seq_len,
        max_lag=max_lag,
        exog_mean=exog_mean,
        exog_std=exog_std,
        power_mean=power_mean,
        power_std=power_std,
        device=device,
        pad_mode=pad_mode,
    )

    preds_b = []
    if len(records) > max_lag:
        preds_b = stage_b_lag_model_rest(
            records_rest=records[max_lag:],
            lag_model=lag_model,
            seq_len=seq_len,
            power_hist_seed=power_hist,
            power_lags=power_lags,
            exog_mean=exog_mean,
            exog_std=exog_std,
            power_mean=power_mean,
            power_std=power_std,
            device=device,
            pad_mode=pad_mode,
        )

    all_preds = preds_a + preds_b

    # 🔑 NEW: mask first seq_len predictions
    masked_preds = [None] * min(seq_len, len(all_preds)) + all_preds[seq_len:]

    return masked_preds[:len(records)]


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

        # normalize actual

        a = (float(r["gpu_power_watts"]) - power_mean) / (power_std + 1e-9)
        # print("unnormalized: ", r["gpu_power_watts"], "normalized: ", a)

        # normalize predicted (already in watts)
        pn = (float(p) - power_mean) / (power_std + 1e-9)
        print("unnormalized actual: ", r["gpu_power_watts"], "normalized actual: ", a, "unnormalized power: ", p, "normalized power: ", pn)

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



if __name__ == "__main__":
    my_records_without_power = []
    with open("../../dataset/merged_log.json") as f:
        for line in f:
            try:
                my_records_without_power.append(json.loads(line))
            except:
                pass

    exog_model_ckpt = "checkpoints_seq_len_120_no_power_lag/best_model_epoch_86_loss_0.027480796795745176.pt"
    lag_model_ckpt = "checkpoints_seq_len_120/best_model_epoch_50_loss_0.015646930248521165.pt"
    norm_meta = json.load(open("../../dataset/normalization.json"))
    
    
    exog_mean = np.array(norm_meta["feature_mean"])
    exog_std = np.array(norm_meta["feature_std"])
    
    power_mean = norm_meta["power_mean"]
    power_std = norm_meta["power_std"]


    preds = predict_power_two_stage_only_till_max_lag(
        records=my_records_without_power[-4000:],
        exog_model_ckpt=exog_model_ckpt,
        lag_model_ckpt=lag_model_ckpt,
        seq_len=120,
        power_lags=(1,2),
        exog_mean=exog_mean,
        exog_std=exog_std,
        power_mean=power_mean,
        power_std=power_std,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pad_mode="repeat_first",
    )
    plot_pred_vs_actual_power_normalized(
        records=my_records_without_power[-4000:],
        preds=preds,
        power_mean=power_mean,
        power_std=power_std,
        out_path="outputs/pred_vs_actual_normalized.png",
    )
    
