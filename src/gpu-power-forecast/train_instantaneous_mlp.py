"""
Train and evaluate the InstantaneousMLP baseline.

This baseline uses only the 7 current-timestep exogenous features
(no window, no temporal context) to predict GPU power.

Run:
    python train_instantaneous_mlp.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import InstantaneousDataset
from model import InstantaneousMLP


# ── reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── evaluation helper ────────────────────────────────────────────────────────
def evaluate(model, loader, device, power_mean, power_std):
    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze()
            preds_all.append(pred.cpu().numpy())
            y_all.append(y.cpu().numpy())

    preds = np.concatenate(preds_all)
    actual = np.concatenate(y_all)

    # de-normalise for interpretable metrics
    preds_w  = preds  * (power_std + 1e-9) + power_mean
    actual_w = actual * (power_std + 1e-9) + power_mean

    mae  = float(np.mean(np.abs(preds_w - actual_w)))
    rmse = float(np.sqrt(np.mean((preds_w - actual_w) ** 2)))

    err = actual_w - preds_w
    delta = 2.0
    huber = float(np.mean(
        np.where(np.abs(err) <= delta,
                 0.5 * err ** 2,
                 delta * (np.abs(err) - 0.5 * delta))
    ))
    return mae, rmse, huber


# ── training loop ─────────────────────────────────────────────────────────────
def train():
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── dataset ──────────────────────────────────────────────────────────────
    dataset = InstantaneousDataset("../../dataset/training_data/merged_log.json")
    split   = int(0.80 * len(dataset))
    train_set = Subset(dataset, list(range(0, split)))
    val_set   = Subset(dataset, list(range(split, len(dataset))))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    input_dim = dataset.features.shape[1]   # 7
    model = InstantaneousMLP(input_dim=input_dim, hidden_dim=64, dropout=0.3)
    model.to(device)

    criterion = nn.HuberLoss(delta=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    ckpt_dir = "checkpoints_instantaneous_mlp"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    num_epochs = 100

    # ── training ──────────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)

        # validation loss (normalised, for scheduler + checkpointing)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).squeeze()
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(ckpt_dir, f"best_model_epoch_{epoch}_loss_{val_loss:.6f}.pt")
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print(f"\nBest val loss: {best_val_loss:.6f}")

    # ── final evaluation in watts ─────────────────────────────────────────────
    mae, rmse, huber = evaluate(
        model, val_loader, device,
        dataset.power_mean, dataset.power_std
    )
    print("\nInstantaneousMLP — Validation Results (de-normalised)")
    print("------------------------------------------------------")
    print(f"  MAE   : {mae:.3f} W")
    print(f"  RMSE  : {rmse:.3f} W")
    print(f"  Huber : {huber:.3f}")


if __name__ == "__main__":
    train()
