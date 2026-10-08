from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import AnalysisConfig, analyze
from .io import export_results, load_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict offline gaze analysis; no deception classification")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing runs are protected")
    parser.add_argument("--coordinate-unit", required=True, choices=["pixels", "degrees", "normalized"])
    parser.add_argument("--timestamp-unit", choices=["s", "ms"])
    parser.add_argument("--velocity-threshold", required=True, type=float, help="In OUTPUT coordinate units per second")
    parser.add_argument("--min-fixation-s", type=float, default=0.1)
    parser.add_argument("--max-gap-s", type=float, default=0.1)
    parser.add_argument("--interpolate-max-s", type=float, default=0)
    parser.add_argument("--smoothing-ms", type=float, default=0)
    parser.add_argument("--screen-size", type=float, nargs=2, metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args(argv)
    try:
        config = AnalysisConfig(args.coordinate_unit, args.velocity_threshold, args.timestamp_unit,
                                args.min_fixation_s, args.max_gap_s, args.interpolate_max_s,
                                args.smoothing_ms, tuple(args.screen_size) if args.screen_size else None)
        result = analyze(load_csv(args.csv), config)
        export_results(result, args.output_dir)
    except (ValueError, OSError, UnicodeError) as error:
        parser.error(str(error))
    print(f"Offline analysis exported to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
