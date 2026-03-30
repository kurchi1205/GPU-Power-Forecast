import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from data import PowerDataset
from model import PowerLSTM
from trainer_lstm import LSTMTrainer

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

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


def run_train():
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== PowerLSTM ===")
    print(f"Using device: {device}")

    dataset = PowerDataset("../../dataset/training_data/merged_log.json", seq_len=120, add_power_as_lag=False)
    split     = int(0.80 * len(dataset))
    train_idx = list(range(0, split))
    val_idx   = list(range(split, len(dataset)))

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    model = PowerLSTM(
        input_dim=dataset.features.shape[1],
        hidden_dim=32,
        num_layers=1,
        dropout=0.7
    )
    trainer = LSTMTrainer(
        model=model,
        train_dataset=train_set,
        val_dataset=val_set,
        batch_size=32,
        lr=5e-4,
        num_epochs=100,
        checkpoint_dir="checkpoints_seq_len_120_no_power_lag_80_train",
        use_wandb=False,
        log_every=10,
    )

    trainer.fit()
    trainer.save("checkpoints_seq_len_120_no_power_lag_80_train/gpu_power_lstm.pt")

    # final evaluation in watts
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    mae, rmse, huber = evaluate(model, val_loader, device,
                                dataset.power_mean, dataset.power_std)
    print("\nPowerLSTM — Validation Results (de-normalised)")
    print("-----------------------------------------------")
    print(f"  MAE   : {mae:.3f} W")
    print(f"  RMSE  : {rmse:.3f} W")
    print(f"  Huber : {huber:.3f}")

if __name__=="__main__":
    run_train()