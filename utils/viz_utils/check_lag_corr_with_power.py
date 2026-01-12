import pandas as pd
import numpy as np
import json

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


TARGET = "gpu_power_watts"
FEATURES = [
    "gpu_sm_util_percent",
    "gpu_mem_util_percent",
    "gpu_utilization_percent",
    "num_pixels",
    "num_splats",
]

POWER_LAGS = 2        # power(t-1), power(t-2)
MAX_FEATURE_LAG = 120
TRAIN_FRAC = 0.7
RIDGE_ALPHA = 1.0


def load_ndjson(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    # Optional: parse timestamp and sort
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    return df.reset_index(drop=True)


def prepare_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def evaluate_feature_lags(df, feature):
    df = prepare_numeric(df, [TARGET, feature])

    # Power lags (baseline)
    for i in range(1, POWER_LAGS + 1):
        df[f"{TARGET}_lag{i}"] = df[TARGET].shift(i)

    df = df.dropna().reset_index(drop=True)
    n = len(df)

    split = int(TRAIN_FRAC * n)

    y = df[TARGET].values
    base_cols = [f"{TARGET}_lag{i}" for i in range(1, POWER_LAGS + 1)]

    # ---- baseline model ----
    X_base = df[base_cols].values
    base_model = Ridge(alpha=RIDGE_ALPHA)
    base_model.fit(X_base[:split], y[:split])

    base_pred = base_model.predict(X_base[split:])
    base_mae = mean_absolute_error(y[split:], base_pred)

    best = {
        "feature": feature,
        "best_lag": None,
        "best_mae": base_mae,
        "delta_mae": 0.0,
    }

    # ---- test lagged feature ----
    for lag in range(1, MAX_FEATURE_LAG + 1):
        df[f"{feature}_lag{lag}"] = df[feature].shift(lag)
        df = df.copy()
        tmp = df.dropna()

        if len(tmp) < split + 10:
            continue

        y2 = tmp[TARGET].values
        X = tmp[base_cols + [f"{feature}_lag{lag}"]].values

        split2 = int(TRAIN_FRAC * len(tmp))

        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X[:split2], y2[:split2])
        pred = model.predict(X[split2:])

        mae = mean_absolute_error(y2[split2:], pred)
        delta = base_mae - mae

        if delta > best["delta_mae"]:
            best.update({
                "best_lag": lag,
                "best_mae": mae,
                "delta_mae": delta,
            })

    return best


def run_all_features(df):
    results = []
    for feature in FEATURES:
        print(f"Testing feature: {feature}")
        res = evaluate_feature_lags(df.copy(), feature)
        results.append(res)
    return pd.DataFrame(results).sort_values("delta_mae", ascending=False)


# -------------------------
# USAGE
# -------------------------
# df must already be loaded and time-ordered
df = load_ndjson("../metrics_log/merged_log.json")
results = run_all_features(df)

print("\n=== Lag contribution results ===")
print(results)
