"""
Camera position feature extraction and visualisation.

Provides:
  - extract_camera_features(records)   : returns position [x,y,z] + velocity [Δx,Δy,Δz]
  - CameraAwareDataset                 : PowerDataset extended with camera features (13-dim input)
  - plot_camera_features(out_dir)      : saves vectorised PDF plots to out_dir

Run standalone to generate plots:
    python camera_features.py
"""

import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)
from torch.utils.data import Dataset

# ── matplotlib defaults ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATASET_PATHS = {
    "Train":        "../../dataset/training_data/merged_log.json",
    "Validation 1": "../../dataset/validation_data_1/merged_log.json",
    "Validation 2": "../../dataset/validation_data_2/merged_log.json",
}

SPLIT_COLORS = {
    "Train":        "#4C72B0",
    "Validation 1": "#DD8452",
    "Validation 2": "#55A868",
}

FEATURE_KEYS = [
    "fps", "gs_fps", "num_pixels", "num_splats",
    "gpu_sm_util_percent", "gpu_mem_util_percent", "gpu_memory_used_MB",
]


# ── feature extraction ────────────────────────────────────────────────────────
def extract_camera_features(records):
    """
    Returns camera position and velocity as numpy arrays.

    Args:
        records: list of dicts, each with a 'camera_position' key ([x, y, z])

    Returns:
        cam_pos : np.ndarray [T, 3]  — raw (x, y, z)
        cam_vel : np.ndarray [T, 3]  — frame-to-frame delta; last row is zeros
        speed   : np.ndarray [T]     — L2 norm of velocity
    """
    cam_pos = np.array([r["camera_position"] for r in records], dtype=np.float32)
    delta   = np.diff(cam_pos, axis=0)                          # [T-1, 3]
    cam_vel = np.vstack([delta, np.zeros((1, 3), dtype=np.float32)])  # [T, 3]
    speed   = np.linalg.norm(cam_vel, axis=1)                   # [T]
    return cam_pos, cam_vel, speed


# ── dataset ───────────────────────────────────────────────────────────────────
class CameraAwareDataset(Dataset):
    """
    Extends the base feature set (7 exogenous features) with camera
    position [x, y, z] and velocity [Δx, Δy, Δz] → 13-dim input.

    Input  X : [seq_len, 13]
    Target y : scalar (normalised gpu_power_watts at t+seq_len)
    """

    def __init__(self, json_path, seq_len=120, train_ratio=0.8):
        self.seq_len = seq_len

        records = []
        with open(json_path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        self.power = np.array([r["gpu_power_watts"] for r in records], dtype=np.float32)

        base = np.array(
            [[r[k] for k in FEATURE_KEYS] for r in records], dtype=np.float32
        )
        cam_pos, cam_vel, _ = extract_camera_features(records)

        self.features = np.concatenate([base, cam_pos, cam_vel], axis=1)  # [T, 13]

        # normalise on train split only
        train_size = int(len(self.features) * train_ratio)
        self.feature_mean = self.features[:train_size].mean(axis=0)
        self.feature_std  = self.features[:train_size].std(axis=0) + 1e-9
        self.power_mean   = self.power[:train_size].mean()
        self.power_std    = self.power[:train_size].std() + 1e-9

        self.features = (self.features - self.feature_mean) / self.feature_std
        self.power    = (self.power    - self.power_mean)   / self.power_std

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = torch.tensor(self.features[idx: idx + self.seq_len], dtype=torch.float32)
        y = torch.tensor(self.power[idx + self.seq_len],         dtype=torch.float32)
        return X, y


# ── plotting ──────────────────────────────────────────────────────────────────
def _load_split(path):
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def plot_camera_features(out_dir="dataset_plots"):
    os.makedirs(out_dir, exist_ok=True)

    split_data = {}
    for name, path in DATASET_PATHS.items():
        records = _load_split(path)
        cam_pos, cam_vel, speed = extract_camera_features(records)
        power = np.array([r["gpu_power_watts"] for r in records], dtype=np.float32)
        split_data[name] = {
            "pos": cam_pos, "vel": cam_vel, "speed": speed, "power": power
        }

    # ── 1. 3-D trajectory ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle("Camera Trajectory per Split (3D)", fontweight="bold")

    for i, (name, d) in enumerate(split_data.items(), 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        pos = d["pos"]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color=SPLIT_COLORS[name], linewidth=0.6, alpha=0.8)
        ax.scatter(*pos[0],  color="green", s=30, zorder=5, label="start")
        ax.scatter(*pos[-1], color="red",   s=30, zorder=5, label="end")
        ax.set_title(name)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "camera_trajectory_3d.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/camera_trajectory_3d.pdf")

    # ── 2. 2-D projections (x-y, x-z, y-z) ──────────────────────────────────
    projections = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Camera Trajectory — 2D Projections", fontweight="bold")

    for ax, (xl, yl, xi, yi) in zip(axes, projections):
        for name, d in split_data.items():
            pos = d["pos"]
            ax.plot(pos[:, xi], pos[:, yi],
                    color=SPLIT_COLORS[name], linewidth=0.6, alpha=0.8, label=name)
            ax.scatter(pos[0, xi],  pos[0, yi],  color="green", s=20, zorder=5)
            ax.scatter(pos[-1, xi], pos[-1, yi], color="red",   s=20, zorder=5)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"{xl}–{yl} plane")
        ax.legend()
        ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "camera_trajectory_2d.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/camera_trajectory_2d.pdf")

    # ── 3. Position components over time ─────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
    fig.suptitle("Camera Position Components over Time", fontweight="bold")
    labels = ["x", "y", "z"]

    for ax, dim, lbl in zip(axes, range(3), labels):
        for name, d in split_data.items():
            ax.plot(d["pos"][:, dim],
                    color=SPLIT_COLORS[name], linewidth=0.6, alpha=0.85, label=name)
        ax.set_ylabel(lbl)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.7)

    axes[-1].set_xlabel("Timestep")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "camera_position_timeseries.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/camera_position_timeseries.pdf")

    # ── 4. Speed over time + power overlay ───────────────────────────────────
    fig, axes = plt.subplots(len(split_data), 1, figsize=(10, 8), sharex=False)
    fig.suptitle("Camera Speed vs. GPU Power over Time", fontweight="bold")

    for ax, (name, d) in zip(axes, split_data.items()):
        t = np.arange(len(d["speed"]))
        ax2 = ax.twinx()
        ax.plot(t, d["speed"],  color=SPLIT_COLORS[name], linewidth=0.7,
                alpha=0.8, label="Speed (m/step)")
        ax2.plot(t, d["power"], color="grey", linewidth=0.5,
                 alpha=0.5, label="GPU Power (W)")
        ax.set_ylabel("Camera Speed", color=SPLIT_COLORS[name])
        ax2.set_ylabel("GPU Power (W)", color="grey")
        ax.set_title(name)
        ax.grid(True, linewidth=0.3, alpha=0.4)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right",
                  fontsize=8, framealpha=0.7)

    axes[-1].set_xlabel("Timestep")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "camera_speed_vs_power.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/camera_speed_vs_power.pdf")

    # ── 5. Speed distribution per split ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Camera Speed Distribution per Split", fontweight="bold")

    split_names  = list(split_data.keys())
    speed_arrays = [split_data[n]["speed"] for n in split_names]
    colors       = [SPLIT_COLORS[n] for n in split_names]

    bp = axes[0].boxplot(
        speed_arrays, tick_labels=split_names, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    axes[0].set_ylabel("Speed (L2 norm per step)")
    axes[0].set_title("Box Plot")
    axes[0].grid(axis="y", linewidth=0.4, alpha=0.5)

    vp = axes[1].violinplot(speed_arrays, positions=range(len(split_names)),
                            showmedians=True)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.7)
    axes[1].set_xticks(range(len(split_names)))
    axes[1].set_xticklabels([n.replace("Validation ", "Val ") for n in split_names])
    axes[1].set_ylabel("Speed (L2 norm per step)")
    axes[1].set_title("Violin Plot")
    axes[1].grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "camera_speed_distribution.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/camera_speed_distribution.pdf")

    print(f"\nAll camera plots saved to ./{out_dir}/")


# ── standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_camera_features()

    # quick sanity check on the dataset
    ds = CameraAwareDataset("../../dataset/training_data/merged_log.json", seq_len=120)
    X, y = ds[0]
    print(f"\nCameraAwareDataset: len={len(ds)}, X shape={X.shape}, y shape={y.shape}")
    print(f"Feature dim: 7 base + 3 pos + 3 vel = {X.shape[1]}")
