"""Small explicitly synthetic gaze demonstration; no training or real participants."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.eye_tracking_analysis import AnalysisConfig, analyze, export_results, load_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or ROOT / "outputs" / f"assignment-3-demo-{uuid4().hex[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    # Analytic geometry: two stable plateaus and a 3-4-5 displacement over 20 ms.
    frame = pd.DataFrame({"participant_id": "synthetic-001", "session_id": "synthetic-session",
                          "trial_id": "synthetic-trial", "timestamp": np.arange(42) * 10.0,
                          "gaze_x": [0.0] * 20 + [1.5] + [3.0] * 21,
                          "gaze_y": [0.0] * 20 + [2.0] + [4.0] * 21})
    source = output / "synthetic_gaze.csv"
    frame.to_csv(source, index=False)
    loaded = load_csv(source)
    loaded.attrs.update(source_kind="synthetic", seed=None, generator="analytic-3-4-5-displacement-v1")
    config = AnalysisConfig("degrees", 30.0, "ms", max_gap_s=0.05)
    result = analyze(loaded, config)
    export_results(result, output / "analysis")
    print("SYNTHETIC DATA - software demonstration, no deception prediction")
    print(result.features.to_string(index=False))
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
