import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
import matplotlib.pyplot as plt
import gc

class LSTMTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=64,
        lr=1e-3,
        num_epochs=25,
        device=None,
        project="power_model_training",
        checkpoint_dir="checkpoints"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.project = project

        self.device = device or ("cuda" if torch.cuda.is_available() else "mps")

        self.model.to(self.device)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = (
            DataLoader(val_dataset, batch_size=1, shuffle=True)
            if val_dataset is not None else None
        )

        self.criterion = nn.HuberLoss(delta=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,  weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,      # Reduce LR by half
            patience=5,       # Wait 5 epochs
        )

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.checkpoint_dir = checkpoint_dir

        wandb.init(
            project=self.project,
        )

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0

        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.criterion(pred.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
        train_loss = epoch_loss / len(self.train_loader)
        wandb.log({"train_loss": train_loss}, step=epoch)
        return train_loss

    def validate(self, epoch):
        if self.val_loader is None:
            return None

        self.model.eval()
        val_loss = 0
        preds_all = []
        y_all = []
        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)

                loss = self.criterion(pred.squeeze(), y)
                val_loss += loss.item()

                preds_all.append(pred.squeeze().cpu())
                y_all.append(y.cpu())

        val_loss /= len(self.val_loader)
        wandb.log({"val_loss": val_loss}, step=epoch)

        return val_loss, preds_all, y_all

    def fit(self):
        print(f"Training on {self.device} for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss, preds, actual = self.validate(epoch)

            if val_loss is not None:
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    self.save(path=os.path.join(self.checkpoint_dir, f"best_model_epoch_{self.best_epoch}_loss_{val_loss}.pt"))
                    print(f"  → New best model saved! (Val Loss: {val_loss:.4f})")
                self.scheduler.step(val_loss)
                print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}")

            plt.figure(figsize=(10,4))
            plt.plot(actual[:5], label="Actual")
            plt.plot(preds[:5], label="Predicted")
            plt.legend()
            
            wandb.log({"pred_vs_actual": wandb.Image(plt)}, step=epoch)
            plt.close()
            gc.collect()

    def save(self, path="checkpoints/lstm_power.pt"):
        directory = os.path.dirname(path)
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path="checkpoints/lstm_power.pt"):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    