from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_eye_tracking_dataset(
    n_participants: int = 12,
    trials_per_participant: int = 8,
    samples_per_trial: int = 128,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a small synthetic dataset for smoke testing only."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for participant_idx in range(n_participants):
        participant_id = f"p{participant_idx:02d}"
        participant_label = int(participant_idx % 2 == 0)
        for trial_idx in range(trials_per_participant):
            trial_id = f"{participant_id}_t{trial_idx:02d}"
            base_x = rng.normal(loc=participant_label * 0.25, scale=0.1)
            base_y = rng.normal(loc=participant_label * -0.2, scale=0.1)
            pupil_base = 3.0 + participant_label * 0.2
            for t in range(samples_per_trial):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "trial_id": trial_id,
                        "timestamp": float(t * 16.67),
                        "gaze_x": base_x + rng.normal(scale=0.05),
                        "gaze_y": base_y + rng.normal(scale=0.05),
                        "pupil": pupil_base + rng.normal(scale=0.08),
                        "blink": int(rng.random() < (0.02 + participant_label * 0.01)),
                        "validity": float(rng.uniform(0.8, 1.0)),
                        "confidence": float(rng.uniform(0.7, 1.0)),
                        "label": participant_label,
                        "stimulus_id": f"s{trial_idx % 4}",
                        "source_file": "synthetic",
                    }
                )
    return pd.DataFrame(rows)
