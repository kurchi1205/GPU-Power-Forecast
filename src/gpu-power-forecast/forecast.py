import json
import numpy as np
import torch
from collections import deque

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

        X_np = pad_window_to_seq_len(list(buf), seq_len, pad_mode=pad_mode)
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

        X_np = pad_window_to_seq_len(list(feat_buf), seq_len, pad_mode=pad_mode)
        X = torch.tensor(X_np[None, :, :], dtype=torch.float32, device=device)

        p_norm = lag_model(X).squeeze().item()
        p_watts = denorm_power(p_norm, power_mean, power_std)

        preds.append(float(p_watts))
        power_hist.append(float(p_watts))

    return preds


def predict_power_two_stage_only_till_max_lag(
    records,
    exog_model,
    lag_model,
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

    # Stage A: first max_lag records
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

    # Stage B: remaining records
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

    return preds_a + preds_b


my_records_without_power = json.load(open("/Users/prerana1298/computing/repo/distributed-gsplats/metrics_log/merged_log.json"))
exog_model_ckpt = "checkpoints_seq_len_120_no_power_lag/best_model_epoch_36_loss_0.5544316075943518.pt"
lag_model_ckpt = "checkpoints_seq_len_120_no_power_lag/best_model_epoch_50_loss_0.015646930248521165.pt"


preds = predict_power_two_stage_only_till_max_lag(
    records=my_records_without_power[-10:],
    exog_model=exog_model_ckpt,
    lag_model=lag_model_ckpt,
    seq_len=60,
    power_lags=(1,2),
    exog_mean=exog_mean,
    exog_std=exog_std,
    power_mean=power_mean,
    power_std=power_std,
    device="cuda" if torch.cuda.is_available() else "cpu",
    pad_mode="repeat_first",
)

print(len(preds), preds[:5])
