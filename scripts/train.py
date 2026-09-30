from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.pipelines import train_baseline_pipeline, train_sequence_pipeline
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the eye-tracking deception prototype.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--allow-synthetic-debug",
        action="store_true",
        help="Fallback to a synthetic dataset when raw data is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    family = config["model"]["family"].lower()
    if family == "baseline":
        metrics = train_baseline_pipeline(config, allow_synthetic_debug=args.allow_synthetic_debug)
    elif family == "deep":
        metrics = train_sequence_pipeline(config, allow_synthetic_debug=args.allow_synthetic_debug)
    else:
        raise ValueError(f"Unsupported model family: {family}")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
