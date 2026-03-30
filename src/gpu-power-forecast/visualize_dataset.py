"""
Dataset characterization plots for the GPU Power Forecast paper.
Generates vectorized PDF figures showing data distributions and variances
across the train, validation_1, and validation_2 splits.

Outputs (saved to dataset_plots/):
  1. timeseries_overview.pdf    — time-series of key signals per split
  2. power_variance.pdf         — box + violin plot of gpu_power_watts per split
  3. feature_distributions.pdf  — violin plots of all 7 input features per split
  4. correlation_heatmap.pdf    — Pearson correlation of features vs. power
  5. dataset_stats table is printed to stdout
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── output directory ────────────────────────────────────────────────────────
OUT_DIR = "dataset_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── matplotlib defaults for paper-quality vector output ─────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,   # embeds fonts as TrueType (required by most venues)
    "ps.fonttype": 42,
})

# ── dataset paths ────────────────────────────────────────────────────────────
SPLITS = {
    "Train":        "../../dataset/training_data/merged_log.json",
    "Validation 1": "../../dataset/validation_data_1/merged_log.json",
    "Validation 2": "../../dataset/validation_data_2/merged_log.json",
}

FEATURE_KEYS = [
    "fps",
    "gs_fps",
    "num_pixels",
    "num_splats",
    "gpu_sm_util_percent",
    "gpu_mem_util_percent",
    "gpu_memory_used_MB",
]

FEATURE_LABELS = [
    "FPS",
    "GS FPS",
    "Num Pixels",
    "Num Splats",
    "SM Util (%)",
    "Mem Util (%)",
    "Memory Used (MB)",
]

SPLIT_COLORS = {
    "Train":        "#4C72B0",
    "Validation 1": "#DD8452",
    "Validation 2": "#55A868",
}


# ── data loading ─────────────────────────────────────────────────────────────
def load_split(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def extract_arrays(records):
    power = np.array([r["gpu_power_watts"] for r in records], dtype=np.float32)
    features = {k: np.array([r[k] for r in records], dtype=np.float32) for k in FEATURE_KEYS}
    return power, features


# ── load all splits ──────────────────────────────────────────────────────────
data = {}
for name, path in SPLITS.items():
    records = load_split(path)
    power, features = extract_arrays(records)
    data[name] = {"power": power, "features": features, "n": len(records)}


# ── 1. Dataset statistics table ──────────────────────────────────────────────
print("\n=== Dataset Statistics: gpu_power_watts ===")
print(f"{'Split':<14} {'N':>6}  {'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
print("-" * 55)
for name, d in data.items():
    p = d["power"]
    print(f"{name:<14} {d['n']:>6}  {p.mean():>7.2f}  {p.std():>7.2f}  {p.min():>7.2f}  {p.max():>7.2f}")


# ── 2. Time-series overview ───────────────────────────────────────────────────
TIMESERIES_SIGNALS = [
    ("power",           "GPU Power (W)",      None),
    ("fps",             "FPS",               None),
    ("gpu_sm_util_percent", "SM Util (%)",   None),
    ("gpu_memory_used_MB",  "Memory (MB)",   None),
]

fig, axes = plt.subplots(len(TIMESERIES_SIGNALS), 1, figsize=(10, 8), sharex=False)
fig.suptitle("Time-Series Overview per Split", fontweight="bold")

for ax, (key, ylabel, _) in zip(axes, TIMESERIES_SIGNALS):
    for name, d in data.items():
        if key == "power":
            series = d["power"]
        else:
            series = d["features"][key]
        ax.plot(series, label=name, color=SPLIT_COLORS[name], linewidth=0.6, alpha=0.85)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.7)

axes[-1].set_xlabel("Timestep")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "timeseries_overview.pdf"), format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUT_DIR}/timeseries_overview.pdf")


# ── 3. Power variance — box + violin ─────────────────────────────────────────
fig, (ax_box, ax_vio) = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle("GPU Power Distribution per Split", fontweight="bold")

split_names = list(data.keys())
power_arrays = [data[n]["power"] for n in split_names]
colors = [SPLIT_COLORS[n] for n in split_names]

# box plot
bp = ax_box.boxplot(
    power_arrays,
    tick_labels=split_names,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=1),
    capprops=dict(linewidth=1),
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax_box.set_ylabel("GPU Power (W)")
ax_box.set_title("Box Plot")
ax_box.grid(axis="y", linewidth=0.4, alpha=0.5)

# violin plot
vp = ax_vio.violinplot(power_arrays, positions=range(len(split_names)), showmedians=True)
for body, color in zip(vp["bodies"], colors):
    body.set_facecolor(color)
    body.set_alpha(0.7)
ax_vio.set_xticks(range(len(split_names)))
ax_vio.set_xticklabels(split_names)
ax_vio.set_ylabel("GPU Power (W)")
ax_vio.set_title("Violin Plot")
ax_vio.grid(axis="y", linewidth=0.4, alpha=0.5)

# annotate mean ± std
for ax in (ax_box, ax_vio):
    for i, (name, arr) in enumerate(zip(split_names, power_arrays)):
        ax.text(
            i + (1 if ax is ax_box else 0),
            arr.max() + 3,
            f"μ={arr.mean():.1f}\nσ={arr.std():.1f}",
            ha="center", va="bottom", fontsize=8,
        )

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "power_variance.pdf"), format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR}/power_variance.pdf")


# ── 4. Feature distributions — violin per split ───────────────────────────────
n_feats = len(FEATURE_KEYS)
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Input Feature Distributions per Split", fontweight="bold")
axes = axes.flatten()

for ax, key, label in zip(axes, FEATURE_KEYS, FEATURE_LABELS):
    arrays = [data[n]["features"][key] for n in split_names]
    vp = ax.violinplot(arrays, positions=range(len(split_names)), showmedians=True)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    ax.set_xticks(range(len(split_names)))
    ax.set_xticklabels([n.replace("Validation ", "Val ") for n in split_names], fontsize=8)
    ax.set_title(label, fontsize=10)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

# hide unused subplot (2x4 grid, 7 features → 1 empty)
axes[n_feats].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "feature_distributions.pdf"), format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR}/feature_distributions.pdf")


# ── 5. Correlation heatmap (features vs. power, per split) ────────────────────
fig, axes = plt.subplots(1, len(data), figsize=(12, 4))
fig.suptitle("Pearson Correlation of Features with GPU Power (per Split)", fontweight="bold")

for ax, (name, d) in zip(axes, data.items()):
    matrix = np.stack([d["features"][k] for k in FEATURE_KEYS] + [d["power"]], axis=1)
    corr = np.corrcoef(matrix.T)

    # only show feature → power correlations (last column, exclude self)
    power_corr = corr[:-1, -1].reshape(-1, 1)

    im = ax.imshow(power_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["Power"], fontsize=9)
    ax.set_yticks(range(len(FEATURE_LABELS)))
    ax.set_yticklabels(FEATURE_LABELS, fontsize=9)
    ax.set_title(name, fontsize=10)

    for i, val in enumerate(power_corr[:, 0]):
        ax.text(0, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(val) > 0.5 else "black")

fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Pearson r")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "correlation_heatmap.pdf"), format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR}/correlation_heatmap.pdf")

print(f"\nAll plots saved to ./{OUT_DIR}/")
