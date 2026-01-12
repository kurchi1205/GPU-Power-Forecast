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
power = np.array([r["gpu_power_watts"] for r in records])
mem_used = np.array([r["gpu_memory_used_MB"] for r in records])
sm_util  = np.array([r["gpu_sm_util_percent"] for r in records])
mem_util = np.array([r["gpu_mem_util_percent"] for r in records])

times = [datetime.fromisoformat(r["timestamp"]) for r in records]
time_s = np.array([(t - times[0]).total_seconds() for t in times])

# ---- helper: z-score ----
def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)

power_z   = z(power)
mem_used_z = z(mem_used)
sm_util_z  = z(sm_util)
mem_util_z = z(mem_util)

# ---- cross-corr in frames ----
def cross_corr(a, b, max_lag):
    lags = np.arange(0, max_lag + 1)
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

lags_mem_used, corr_mem_used = cross_corr(mem_used_z, power_z, max_lag_frames)
lags_sm,       corr_sm       = cross_corr(sm_util_z,  power_z, max_lag_frames)
lags_memu,     corr_memu     = cross_corr(mem_util_z, power_z, max_lag_frames)

# ---- keep only non-negative lags (feature leads power) ----
def positive_lags(lags, corr):
    mask = lags >= 0
    return lags[mask], corr[mask]

lags_mem_used_pos, corr_mem_used_pos = positive_lags(lags_mem_used, corr_mem_used)
lags_sm_pos,       corr_sm_pos       = positive_lags(lags_sm,       corr_sm)
lags_memu_pos,     corr_memu_pos     = positive_lags(lags_memu,     corr_memu)

# ---- convert frames → seconds using median timestep ----
dt_est = np.median(np.diff(time_s))  # seconds per sample

lagsec_mem_used = lags_mem_used_pos * dt_est
lagsec_sm       = lags_sm_pos       * dt_est
lagsec_memu     = lags_memu_pos     * dt_est

# ---- plot ----
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)

# 1) GPU memory used → Power
axes[0].plot(lagsec_mem_used, corr_mem_used_pos, color="tab:orange")
axes[0].set_title("Effect of GPU Memory Used on Power (positive lags only)")
axes[0].set_xlabel("Lag (seconds)  [how long after memory usage change]")
axes[0].set_ylabel("Correlation")
axes[0].grid(True)

# 2) SM utilization → Power
axes[1].plot(lagsec_sm, corr_sm_pos, color="tab:blue")
axes[1].set_title("Effect of SM Utilization on Power (positive lags only)")
axes[1].set_xlabel("Lag (seconds)  [how long after SM util change]")
axes[1].set_ylabel("Correlation")
axes[1].grid(True)

# 3) Memory controller utilization → Power
axes[2].plot(lagsec_memu, corr_memu_pos, color="tab:green")
axes[2].set_title("Effect of Memory Utilization on Power (positive lags only)")
axes[2].set_xlabel("Lag (seconds)  [how long after mem util change]")
axes[2].set_ylabel("Correlation")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("positive_lag_effects_gpu_mem_and_utils.jpeg")
plt.show()
