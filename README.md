# Deception Detection Prototype

Research prototype for classification of deception-related or concealed-recognition patterns in eye-tracking temporal data under controlled experimental conditions.

This repository is intentionally framed as a controlled experimental classification pipeline, not a universal lie detector.

## Starting Dataset Choice

This initial version is built around the Zenodo dataset `Eye tracking as a Tool for deception detection with CIT` (DOI `10.5281/zenodo.18525952`).

Why this dataset first:

- It is directly about concealed knowledge / deception-related gaze behavior under a Concealed Information Test paradigm.
- It contains eye-tracking recordings from 39 participants and explicit experimental metadata.
- The published structure uses per-recording spreadsheet files, which is practical for an adapter-based ingestion layer.
- It is more immediately usable than `Bag-of-Lies`, which is multimodal and less convenient for fully reproducible first-pass setup.

`Bag-of-Lies` is still relevant and the codebase includes a placeholder adapter entry for future extension.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Dataset Placement

Automatic download is not implemented in this first version because the datasets are hosted externally and may change structure or access rules.

For the CIT dataset:

1. Download the `.xlsx` files from `https://zenodo.org/records/18525952`.
2. Place them under `data/raw/cit/`.

Expected example:

```text
data/raw/cit/CIT Recording1.xlsx
data/raw/cit/CIT Recording2.xlsx
```

For `Bag-of-Lies`:

- Place any future tabular eye-tracking exports under `data/raw/bag_of_lies/`.
- The current repository contains a placeholder adapter registration, but not a finalized parser because the available public materials do not fully specify a stable raw tabular export schema.

## Run Training

Baseline model on aggregated features:

```bash
python scripts/train.py --config configs/cit_baseline.yaml
```

LSTM on sliding windows:

```bash
python scripts/train.py --config configs/cit_lstm.yaml
```

If no external dataset is present, you can run a sanity-check pipeline with synthetic data:

```bash
python scripts/train.py --config configs/cit_baseline.yaml --allow-synthetic-debug
```

This synthetic mode is only for verifying code execution and should not be used for research claims.

## Current Scope

Implemented:

- adapter-based tabular dataset loading
- offline and online-safe preprocessing
- aggregated features and sliding windows
- subject-independent splitting
- baseline models and deep sequence models
- checkpointing and evaluation artifact generation
- streaming-style prediction aggregation

Limitations:

- fixation extraction is currently heuristic rather than vendor-specific
- `Bag-of-Lies` parsing is a TODO
- live deployment is not implemented
