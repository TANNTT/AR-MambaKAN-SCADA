# Copied from AR-MambaKAN-SCADA @ 8582ecc937b45081f5c4116694c73d685f6c8508
# Source file: src/models/rnn_baselines.py
# Local change: add readout={"last","full"} on BiLSTM (default "last" matches lines 18-19).
# MLP / unidirectional LSTM kept for G2b ablation codes A1 / A2 (no BiLSTM readout).

"""Classical RNN / MLP forecasting baselines."""

from torch import nn

from g_chung.readout import bilstm_state_readout


class BaselineBiLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, horizon: int, readout: str = "last"):
        super().__init__()
        if readout not in ("last", "full"):
            raise ValueError(f"readout must be 'last' or 'full', got {readout!r}")
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.readout = readout
        self.backbone = nn.LSTM(in_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),
        )

    def forward(self, x):
        out, _ = self.backbone(x)
        z = bilstm_state_readout(out, self.hidden_dim, self.readout)
        y = self.head(z).view(x.size(0), self.horizon, 2)
        return y


class BaselineLSTM(nn.Module):
    """Unidirectional LSTM (ablation A2). Readout flag ignored: only last step exists."""

    def __init__(self, in_dim: int, hidden_dim: int, horizon: int, readout: str = "last"):
        super().__init__()
        del readout
        self.horizon = horizon
        self.backbone = nn.LSTM(in_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),
        )

    def forward(self, x):
        out, _ = self.backbone(x)
        z = out[:, -1, :]
        y = self.head(z).view(x.size(0), self.horizon, 2)
        return y


class BaselineMLP(nn.Module):
    """Flattened MLP (ablation A1). No recurrent state / readout."""

    def __init__(self, in_dim: int, lookback: int, hidden_dim: int, horizon: int, readout: str = "last"):
        super().__init__()
        del readout
        self.lookback = lookback
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(in_dim * lookback, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),
        )

    def forward(self, x):
        z = x.reshape(x.size(0), -1)
        y = self.net(z).view(x.size(0), self.horizon, 2)
        return y
