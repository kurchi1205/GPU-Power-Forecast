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

# ---- Filter: drop bad gs_fps ----
records = [r for r in records if r.get("gs_fps", 0) <= 500]

print(f"Filtered records count: {len(records)}")

# ---- Extract arrays ----
times = [datetime.fromisoformat(r["timestamp"]) for r in records]
time_s = np.array([(t - times[0]).total_seconds() for t in times])

power = np.array([r["gpu_power_watts"] for r in records])
gs_fps = np.array([r["gs_fps"] for r in records])

num_pixels = np.array([r["num_pixels"] for r in records]) if "num_pixels" in records[0] else None
num_splats = np.array([r["num_splats"] for r in records]) if "num_splats" in records[0] else None

# ---- Setup 3-row subplot ----
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# ===============================================
# 1) GS FPS vs Power
# ===============================================
axes[0].plot(time_s, power, color="tab:red", label="GPU Power (W)", alpha=0.7)
axes[0].set_ylabel("Power (W)", color="tab:red")
axes[0].tick_params(axis='y', labelcolor='tab:red')

ax0b = axes[0].twinx()
ax0b.plot(time_s, gs_fps, color="tab:blue", label="GS FPS", alpha=0.7)
ax0b.set_ylabel("GS FPS", color="tab:blue")
ax0b.tick_params(axis='y', labelcolor='tab:blue')

axes[0].set_title("Power vs GS FPS")

# ===============================================
# 2) Num Pixels vs Power
# ===============================================
axes[1].plot(time_s, power, color="tab:red", label="GPU Power (W)", alpha=0.7)
axes[1].set_ylabel("Power (W)", color="tab:red")
axes[1].tick_params(axis='y', labelcolor='tab:red')

if num_pixels is not None:
    ax1b = axes[1].twinx()
    ax1b.plot(time_s, num_pixels, color="tab:green", label="Num Pixels", alpha=0.7)
    ax1b.set_ylabel("Num Pixels", color="tab:green")
    ax1b.tick_params(axis='y', labelcolor='tab:green')
else:
    axes[1].text(0.5, 0.5, "num_pixels not logged", ha="center")

axes[1].set_title("Power vs Num Pixels")

# ===============================================
# 3) Num Splats vs Power
# ===============================================
axes[2].plot(time_s, power, color="tab:red", label="GPU Power (W)", alpha=0.7)
axes[2].set_ylabel("Power (W)", color="tab:red")
axes[2].tick_params(axis='y', labelcolor='tab:red')

if num_splats is not None:
    ax2b = axes[2].twinx()
    ax2b.plot(time_s, num_splats, color="tab:purple", alpha=0.7)
    ax2b.set_ylabel("Num Splats", color="tab:purple")
    ax2b.tick_params(axis='y', labelcolor='tab:purple')
else:
    axes[2].text(0.5, 0.5, "num_splats not logged", ha="center")

axes[2].set_title("Power vs Num Splats (Gaussians)")
axes[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("separate_vs_power.jpeg")
plt.show()
