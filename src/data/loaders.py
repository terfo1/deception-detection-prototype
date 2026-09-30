from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.adapters import BagOfLiesAdapter, CITAdapter, iter_loaded_frames
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def get_adapter(dataset_name: str, allow_missing_schema: bool = True):
    """Return the dataset adapter registered under a given name."""
    key = dataset_name.lower()
    if key == "cit":
        return CITAdapter(allow_missing_schema=allow_missing_schema)
    if key == "bag_of_lies":
        return BagOfLiesAdapter(allow_missing_schema=allow_missing_schema)
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def load_dataset(
    dataset_name: str,
    raw_dir: str | Path,
    recursive: bool = False,
    allow_missing_schema: bool = True,
) -> pd.DataFrame:
    """Load all supported raw files into a normalized dataframe."""
    raw_path = Path(raw_dir)
    adapter = get_adapter(dataset_name=dataset_name, allow_missing_schema=allow_missing_schema)
    files = adapter.discover_files(raw_path, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No raw files found in {raw_path} for dataset={dataset_name}.")

    frames = list(iter_loaded_frames(adapter, files))
    dataset = pd.concat(frames, ignore_index=True)
    LOGGER.info("Loaded %s rows from %s files.", len(dataset), len(files))
    return dataset
