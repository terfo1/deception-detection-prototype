from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Torch dataset for sequence windows."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]
