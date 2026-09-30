from __future__ import annotations

from collections import deque

import numpy as np


class SlidingWindowInferenceAggregator:
    """Aggregate window predictions over time for streaming-style inference."""

    def __init__(self, history_size: int = 5, mode: str = "moving_average") -> None:
        self.history_size = history_size
        self.mode = mode
        self.history: deque[float] = deque(maxlen=history_size)

    def update(self, probability: float) -> float:
        """Update state and return the aggregated score."""
        self.history.append(float(probability))
        if self.mode == "majority_vote":
            binary = [int(value >= 0.5) for value in self.history]
            return float(np.mean(binary) >= 0.5)
        if self.mode == "moving_average":
            return float(np.mean(self.history))
        raise ValueError(f"Unsupported aggregation mode: {self.mode}")
