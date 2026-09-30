from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TrainingHistory:
    best_epoch: int
    best_val_loss: float


def train_sequence_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    checkpoint_path: str | Path,
) -> tuple[nn.Module, TrainingHistory]:
    """Train a sequence classifier with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_losses.append(loss.item())
        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        LOGGER.info("Epoch %s | train_loss=%.4f | val_loss=%.4f", epoch, mean_train_loss, mean_val_loss)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            checkpoint = Path(checkpoint_path)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                LOGGER.info("Early stopping triggered at epoch %s.", epoch)
                break

    model.load_state_dict(best_state)
    return model, TrainingHistory(best_epoch=best_epoch, best_val_loss=best_val_loss)
