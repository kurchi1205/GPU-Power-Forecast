import json
import pandas as pd
import matplotlib.pyplot as plt
import os

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# -------- config --------
INPUT_FILE = "../metrics_log/merged_log.json"      # newline-delimited JSON
OUTPUT_DIR = "outputs/acf_pacf_plots"
POWER_COL = "gpu_power_watts"
MAX_LAGS_LIST = [1, 10, 20, 30, 60, 120]
# ------------------------


# Load NDJSON
rows = []
with open(INPUT_FILE, "r") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

df = pd.DataFrame(rows)

# Sort by time if available
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

# Power series
power = pd.to_numeric(df[POWER_COL], errors="coerce").dropna()

if len(power) < 20:
    raise RuntimeError("Not enough samples to plot ACF/PACF")

# Make output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- ACF / PACF for each max lag --------
for max_lag in MAX_LAGS_LIST:
    # ACF
    plt.figure()
    plot_acf(power, lags=max_lag)
    plt.title(f"ACF: GPU Power (max_lag={max_lag})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/acf_power_lag{max_lag}.png", dpi=150)
    plt.close()

    # PACF
    plt.figure()
    plot_pacf(power, lags=max_lag, method="ywm")
    plt.title(f"PACF: GPU Power (max_lag={max_lag})")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pacf_power_lag{max_lag}.png", dpi=150)
    plt.close()

print(f"Saved ACF/PACF plots to: {OUTPUT_DIR}/")
