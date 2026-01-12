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

# ---- Filter: drop records with gs_fps > 500 ----
records = [r for r in records if r.get("gs_fps", 0) <= 500]

print(f"Filtered records count: {len(records)}")

# ---- Extract arrays ----
times = [datetime.fromisoformat(r["timestamp"]) for r in records]
gs_fps = np.array([r["gs_fps"] for r in records])
power = np.array([r["gpu_power_watts"] for r in records])
temp = np.array([r["gpu_temp_celsius"] for r in records])

num_splats = np.array([r["num_splats"] for r in records]) if "num_splats" in records[0] else None
num_pixels = np.array([r["num_pixels"] for r in records]) if "num_pixels" in records[0] else None
sm_util = np.array([r["gpu_sm_util_percent"] for r in records]) if "gpu_sm_util_percent" in records[0] else None
mem_util = np.array([r["gpu_mem_util_percent"] for r in records]) if "gpu_mem_util_percent" in records[0] else None

# ---- Normalize time ----
t0 = times[0]
time_s = np.array([(t - t0).total_seconds() for t in times])

# ---- Plot ----
fig, ax1 = plt.subplots(figsize=(10,6))

# GS FPS
ax1.plot(time_s, gs_fps, color="tab:blue", label="GS FPS", alpha=0.7)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("GS FPS", color="tab:blue")
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Power + Temp
ax2 = ax1.twinx()
ax2.plot(time_s, power, color="tab:red", label="GPU Power (W)", alpha=0.7)
ax2.plot(time_s, temp, color="tab:orange", label="GPU Temp (°C)", alpha=0.5, linestyle="--")
ax2.set_ylabel("Power / Temp", color="tab:red")
ax2.tick_params(axis='y', labelcolor='tab:red')

# Splats count
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
if num_splats is not None:
    ax3.plot(time_s, num_splats, color="tab:purple", linestyle="--", alpha=0.9, label="Num Splats")
ax3.set_ylabel("Num Splats", color="tab:purple")
ax3.tick_params(axis='y', labelcolor='tab:purple')

# ---- Pixel count as 4th axis ----
ax4 = ax1.twinx()
ax4.spines["right"].set_position(("outward", 120))  # further offset
if num_pixels is not None:
    ax4.plot(time_s, num_pixels, color="tab:green", linestyle=":", alpha=0.85, label="Num Pixels")
ax4.set_ylabel("Num Pixels", color="tab:green")
ax4.tick_params(axis='y', labelcolor='tab:green')

# SM / Mem util shaded
if sm_util is not None:
    ax1.fill_between(time_s, 0, sm_util, color="purple", alpha=0.15, label="SM Util (%)")
if mem_util is not None:
    ax1.fill_between(time_s, 0, mem_util, color="brown", alpha=0.15, label="Mem Util (%)")

# Legend merge
lines, labels = [], []
for ax in [ax1, ax2, ax3, ax4]:
    l, lab = ax.get_legend_handles_labels()
    lines += l
    labels += lab
ax1.legend(lines, labels, loc="upper left")

plt.title("GS FPS, GPU Power/Temp, Num Splats, Num Pixels, GPU Utilization Over Time")
plt.tight_layout()
plt.savefig("metrics_plot.jpeg")
plt.show()
