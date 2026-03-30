import torch.nn as nn


class InstantaneousMLP(nn.Module):
    """
    Feedforward MLP baseline — no temporal context.
    Input: current-timestep exogenous features, shape [D].
    Output: scalar power prediction.

    Used to test whether any temporal modeling is needed at all.
    Parameter budget is kept comparable to PowerLSTM (hidden_dim=32).
    """

    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class WindowedLinear(nn.Module):
    """
    Linear baseline on a flattened temporal window.
    Input: [seq_len, D]  →  flattened to [seq_len * D].
    Tests whether any non-linearity or recurrence is needed
    beyond a simple weighted sum of window features.
    """

    def __init__(self, seq_len, input_dim):
        super().__init__()
        self.fc = nn.Linear(seq_len * input_dim, 1)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class WindowedMLP(nn.Module):
    """
    MLP baseline on a flattened temporal window.
    Input: [seq_len, D]  →  flattened to [seq_len * D].
    Tests whether the ordering / recurrence of LSTM is needed
    vs. treating the window as a bag of unordered features.
    hidden_dim kept at 64 to give the model reasonable capacity
    without exploding parameter count.
    """

    def __init__(self, seq_len, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        flat_dim = seq_len * input_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


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


class PowerGRU(nn.Module):
    """
    GRU variant of PowerLSTM — identical architecture, GRU cell.
    GRU has fewer parameters than LSTM (no separate cell state),
    so comparing them isolates whether LSTM's extra gating is necessary.
    """

    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.7):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class PowerRNN(nn.Module):
    """
    Vanilla RNN variant of PowerLSTM — identical architecture, Elman RNN cell.
    No gating at all; isolates whether any gating mechanism is needed
    vs. a plain recurrent hidden state.
    """

    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.7):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)