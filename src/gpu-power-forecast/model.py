import torch.nn as nn

class PowerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        # LSTM layer with dropout between layers (only if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Additional dropout layer after LSTM
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers with batch normalization
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        
        # Take last hidden state
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Fully connected layers
        return self.fc(out)