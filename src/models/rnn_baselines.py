"""Classical RNN / MLP forecasting baselines."""

from torch import nn


class BaselineBiLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.backbone = nn.LSTM(in_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * 2),
        )

    def forward(self, x):
        out, _ = self.backbone(x)
        z = out[:, -1, :]
        y = self.head(z).view(x.size(0), self.horizon, 2)
        return y


class BaselineLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, horizon: int):
        super().__init__()
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
    def __init__(self, in_dim: int, lookback: int, hidden_dim: int, horizon: int):
        super().__init__()
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
