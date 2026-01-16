import torch
import numpy as np
import random
from data import PowerDataset
from model import PowerLSTM
from trainer_lstm  import LSTMTrainer
from torch.utils.data import Subset

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

def run_train():
    set_seed()

    dataset = PowerDataset("../../dataset/merged_log.json", seq_len=120, add_power_as_lag=False)
    split = int(0.80 * len(dataset))
    train_idx = list(range(0, split))
    val_idx   = list(range(split, len(dataset)))

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    # model = PowerLSTM(input_dim=dataset.features.shape[1], hidden_dim=64, num_layers=2, dropout=0.8)
    model = PowerLSTM(
        input_dim=dataset.features.shape[1], 
        hidden_dim=32,      # Reduced hidden size
        num_layers=1,       # Fewer layers
        dropout=0.7
    )
    trainer = LSTMTrainer(
        model=model,
        train_dataset=train_set,
        val_dataset=val_set,
        batch_size=32,
        lr=5e-4,
        num_epochs=100,
        checkpoint_dir="checkpoints_seq_len_120_no_power_lag_80_train"
    )

    trainer.fit()
    trainer.save("checkpoints_seq_len_120_no_power_lag_80_train/gpu_power_lstm.pt")

if __name__=="__main__":
    run_train()