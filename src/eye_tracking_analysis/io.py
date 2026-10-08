from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import AnalysisResult
from .validation import IDS, validate_data


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    payload = path.read_bytes()
    text = payload.decode("utf-8-sig")
    header = next(csv.reader(io.StringIO(text)), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate column names")
    try:
        frame = pd.read_csv(io.StringIO(text), dtype={name: "string" for name in IDS}, keep_default_na=False,
                            na_values=["", "NA", "NaN", "nan"])
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError("Empty or malformed CSV") from error
    frame = validate_data(frame)
    frame.attrs.update(input_sha256=hashlib.sha256(payload).hexdigest(), input_name=path.name, source_kind="unspecified")
    return frame


def export_results(result: AnalysisResult, output_dir: str | Path, *, plot: bool = True) -> Path:
    """Create a new run directory; never overwrite prior research artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result.samples.to_csv(output / "samples.csv", index=False)
    result.features.to_csv(output / "features.csv", index=False)
    result.events.to_csv(output / "events.csv", index=False)
    metadata = {**result.metadata, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                              "pandas": pd.__version__},
                "module_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")}}
    (output / "analysis.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    if plot:
        from .visualization import plot_results
        plot_results(result, output / "gaze_analysis.png")
    return output
