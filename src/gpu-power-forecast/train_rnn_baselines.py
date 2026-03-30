"""
Train and evaluate PowerGRU and PowerRNN baselines.

Both models use the same 120-step window, hidden_dim, dropout, and FC head
as the trained PowerLSTM — only the recurrent cell differs.
This isolates whether LSTM's gating mechanism is necessary.

Run:
    python train_rnn_baselines.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import PowerDataset
from model import PowerGRU, PowerRNN

SEQ_LEN   = 120
HIDDEN_DIM = 32
NUM_LAYERS = 1
DROPOUT    = 0.7


# ── reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, power_mean, power_std):
    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze()
            preds_all.append(pred.cpu().numpy())
            y_all.append(y.cpu().numpy())

    preds  = np.concatenate(preds_all)
    actual = np.concatenate(y_all)

    preds_w  = preds  * (power_std + 1e-9) + power_mean
    actual_w = actual * (power_std + 1e-9) + power_mean

    mae   = float(np.mean(np.abs(preds_w - actual_w)))
    rmse  = float(np.sqrt(np.mean((preds_w - actual_w) ** 2)))
    err   = actual_w - preds_w
    delta = 2.0
    huber = float(np.mean(
        np.where(np.abs(err) <= delta,
                 0.5 * err ** 2,
                 delta * (np.abs(err) - 0.5 * delta))
    ))
    return mae, rmse, huber


# ── generic training loop ─────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device, ckpt_dir, num_epochs=100):
    criterion = nn.HuberLoss(delta=2.0)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X).squeeze(), y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(
                ckpt_dir, f"best_model_epoch_{epoch}_loss_{val_loss:.6f}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    print(f"  Best val loss: {best_val_loss:.6f}")


# ── main ──────────────────────────────────────────────────────────────────────
def run():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    dataset = PowerDataset(
        "../../dataset/training_data/merged_log.json",
        seq_len=SEQ_LEN,
        add_power_as_lag=False,
    )
    split     = int(0.80 * len(dataset))
    train_set = Subset(dataset, list(range(0, split)))
    val_set   = Subset(dataset, list(range(split, len(dataset))))

    train_loader = DataLoader(train_set, batch_size=32,  shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False)

    input_dim = dataset.features.shape[1]  # 7

    models = {
        "PowerGRU": PowerGRU(input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                             num_layers=NUM_LAYERS, dropout=DROPOUT),
        "PowerRNN": PowerRNN(input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                             num_layers=NUM_LAYERS, dropout=DROPOUT),
    }

    results = {}
    for name, model in models.items():
        print(f"=== {name} ===")
        model.to(device)
        train_model(model, train_loader, val_loader, device,
                    ckpt_dir=f"checkpoints_{name.lower()}", num_epochs=100)
        mae, rmse, huber = evaluate(model, val_loader, device,
                                    dataset.power_mean, dataset.power_std)
        results[name] = {"MAE": mae, "RMSE": rmse, "Huber": huber}
        print(f"  → MAE: {mae:.3f} W | RMSE: {rmse:.3f} W | Huber: {huber:.3f}\n")

    print("=== Summary ===")
    print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'Huber':>8}")
    print("-" * 40)
    for name, m in results.items():
        print(f"{name:<12} {m['MAE']:>8.3f} {m['RMSE']:>8.3f} {m['Huber']:>8.3f}")


if __name__ == "__main__":
    run()
