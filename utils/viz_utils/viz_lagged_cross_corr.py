import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---- Load log ----
records = []
with open("merged_log.json") as f:
    for line in f:
        try:
            rec = json.loads(line)
            records.append(rec)
        except json.JSONDecodeError:
            pass

# ---- Extract arrays ----
power  = np.array([r["gpu_power_watts"] for r in records])
fps    = np.array([r["gs_fps"] for r in records])
splats = np.array([r["num_splats"] for r in records])

# num_pixels might not exist in very old logs, so guard it
if "num_pixels" in records[0]:
    pixels = np.array([r["num_pixels"] for r in records])
else:
    raise KeyError("num_pixels not found in merged_log.json entries")

times = [datetime.fromisoformat(r["timestamp"]) for r in records]
time_s = np.array([(t - times[0]).total_seconds() for t in times])

# ---- helper: z-score ----
def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)

power_z  = z(power)
fps_z    = z(fps)
splats_z = z(splats)
pixels_z = z(pixels)

# ---- cross-corr in frames ----
def cross_corr(a, b, max_lag):
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []
    for lag in lags:
        if lag < 0:
            c = np.corrcoef(a[-lag:], b[:lag])[0, 1]
        elif lag > 0:
            c = np.corrcoef(a[:-lag], b[lag:])[0, 1]
        else:
            c = np.corrcoef(a, b)[0, 1]
        corr.append(c)
    return np.array(lags), np.array(corr)

max_lag_frames = 200

lags_fps,    corr_fps    = cross_corr(fps_z,    power_z, max_lag_frames)
lags_splats, corr_splats = cross_corr(splats_z, power_z, max_lag_frames)
lags_pix,    corr_pix    = cross_corr(pixels_z, power_z, max_lag_frames)

# ---- keep only non-negative lags (feature leads power) ----
def positive_lags(lags, corr):
    mask = lags >= 0
    return lags[mask], corr[mask]

lags_fps_pos,    corr_fps_pos    = positive_lags(lags_fps,    corr_fps)
lags_splats_pos, corr_splats_pos = positive_lags(lags_splats, corr_splats)
lags_pix_pos,    corr_pix_pos    = positive_lags(lags_pix,    corr_pix)

# ---- convert frames → seconds using median timestep ----
dt_est = np.median(np.diff(time_s))   # seconds per sample

lagsec_fps    = lags_fps_pos    * dt_est
lagsec_splats = lags_splats_pos * dt_est
lagsec_pix    = lags_pix_pos    * dt_est

# ---- plot ----
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)

# 1) FPS → Power
axes[0].plot(lagsec_fps, corr_fps_pos, color="tab:blue")
axes[0].set_title("Effect of FPS on Power (positive lags only)")
axes[0].set_xlabel("Lag (seconds)  [how long after FPS change]")
axes[0].set_ylabel("Correlation")
axes[0].grid(True)

# 2) Num Splats → Power
axes[1].plot(lagsec_splats, corr_splats_pos, color="tab:purple")
axes[1].set_title("Effect of Num Splats on Power (positive lags only)")
axes[1].set_xlabel("Lag (seconds)  [how long after splat change]")
axes[1].set_ylabel("Correlation")
axes[1].grid(True)

# 3) Num Pixels → Power
axes[2].plot(lagsec_pix, corr_pix_pos, color="tab:green")
axes[2].set_title("Effect of Num Pixels on Power (positive lags only)")
axes[2].set_xlabel("Lag (seconds)  [how long after pixel-count change]")
axes[2].set_ylabel("Correlation")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("positive_lag_effects.jpeg")
plt.show()
