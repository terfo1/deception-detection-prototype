from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Configurable LSTM classifier for windowed gaze sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(inputs)
        features = self.dropout(hidden[-1])
        return self.classifier(features)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = out[..., : x.shape[-1]]
        residual = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + residual)


class TCNClassifier(nn.Module):
    """Simple TCN for temporal eye-tracking windows."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = input_dim
        for layer_idx in range(num_layers):
            dilation = 2**layer_idx
            blocks.append(TemporalBlock(in_channels, hidden_dim, kernel_size=3, dilation=dilation, dropout=dropout))
            in_channels = hidden_dim
        self.network = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(1, 2)
        features = self.network(x).mean(dim=-1)
        return self.classifier(features)


def build_sequence_model(config: dict) -> nn.Module:
    """Factory for sequence classifiers."""
    name = config["name"].lower()
    kwargs = {
        "input_dim": int(config["input_dim"]),
        "hidden_dim": int(config["hidden_dim"]),
        "num_layers": int(config["num_layers"]),
        "dropout": float(config["dropout"]),
        "num_classes": int(config["num_classes"]),
    }
    if name == "lstm":
        return LSTMClassifier(**kwargs)
    if name == "tcn":
        return TCNClassifier(**kwargs)
    raise ValueError(f"Unsupported sequence model: {config['name']}")
