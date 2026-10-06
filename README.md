# Eye-tracking Scientific Processing Prototype

[![CI](https://github.com/terfo1/deception-detection-prototype/actions/workflows/ci.yml/badge.svg)](https://github.com/terfo1/deception-detection-prototype/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small Python research tool for processing eye-tracking tables, extracting
trial features, and evaluating classifiers for a controlled Concealed Information
Test (CIT). It includes a reproducible synthetic demo, automated tests, and a
GitHub Actions pipeline that tests, builds, and publishes versioned packages.

The target is concealed knowledge in a specific experiment. The current module
does not establish accuracy for general deception detection.

## Quick start: no external dataset required

Python 3.10 or newer is required. Run from the repository root:

```bash
git clone https://github.com/terfo1/deception-detection-prototype.git
cd deception-detection-prototype
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -e '.[dev]'
python scripts/train.py --config configs/synthetic_baseline.yaml
python -m pytest -m 'not deep'
python -m ruff check .
```

The demo generates 24 synthetic participants with 8 trials each, processes them,
splits by participant, trains logistic regression, and writes these files:

```text
outputs/synthetic_demo/
  metrics/logistic_regression_metrics.json
  figures/logistic_regression_confusion_matrix.png
  figures/logistic_regression_roc_curve.png
  checkpoints/logistic_regression.pkl
```

Metrics describe synthetic debug data and must not be reported as scientific
evidence. Generated files are excluded from Git.

## What the module does

1. Maps tabular exports to a shared schema with participant, trial, timestamp,
   gaze coordinates, pupil diameter, validity, and label.
2. Filters invalid samples, interpolates within trials, and smooths signals.
3. Calculates pupil change relative to the initial trial baseline, then applies
   optional standardization while preserving participant and trial IDs.
4. Extracts aggregate trial features or sequence windows that stay within trials.
5. Splits participants into training, validation, and test groups.
6. Trains a classifier and exports metrics, figures, and a checkpoint.

| Component | Location |
|---|---|
| Raw-data schema and adapters | `src/data/` |
| Processing and feature extraction | `src/features/` |
| Baseline and sequence models | `src/models/` |
| Training and participant splitting | `src/training/` |
| Metrics and plots | `src/evaluation/` |
| Command-line training tool | `scripts/train.py` |
| Reproducible configs | `configs/` |
| Automated tests | `tests/` |
| CI and release workflow | `.github/workflows/ci.yml` |

## Multi-agent workflow

The existing pipeline can also be run through six specialized agents and a
bounded deterministic orchestrator. It provides typed state, signal quality,
saved-model inference, verification, JSON reports, execution logs, and per-agent
call statistics. No additional dependencies or web backend are required.

```bash
python examples/multi_agent_demo.py
```

This explicitly synthetic example trains outside the agents and runs inference
without labels on a held-out participant. See the
[multi-agent architecture and CLI guide](docs/MULTI_AGENT_ARCHITECTURE_RU.md)
for real-file/model contracts, configuration, limitations, and the diagram.
The current processing is offline; agent orchestration does not establish
real-time or scientifically validated deception detection.

For a portable visual demonstration with step-by-step playback and three actual
success/failure scenarios, run `python examples/multi_agent_demo.py --open`
(or double-click `examples/show_demo.cmd` on this Windows setup). The generated
HTML works offline and can be shared as one file. See the
[presentation walkthrough](docs/DEMO_GUIDE_RU.md).

## Chosen technologies

| Technology | Reason for choosing it |
|---|---|
| Python | A concise language with a mature scientific computing ecosystem. |
| pandas + NumPy | Table/schema handling and efficient numeric features/windows. |
| scikit-learn | Established baseline models, scaling pipelines, group splits, and metrics. |
| Matplotlib | Exports reproducible confusion matrices and ROC figures without a GUI. |
| openpyxl | Reads the source dataset's Excel format. |
| PyYAML | Stores experiment parameters separately from implementation. |
| PyTorch, optional | Supports gradient-based LSTM and TCN sequence models. |
| pytest + pytest-cov | Regression/integration tests and coverage reports. |
| Ruff | Fast checks for common Python coding errors. |
| Git + GitHub Actions | Versioned research code, issue tracking, and automated verification/releases. |

PyTorch is optional to keep the basic processing tool lightweight. XGBoost is
also optional: install `python -m pip install -e '.[xgboost]'` before selecting it.
No web framework or database is needed for this offline module.

## Running with CIT data

The source dataset is
[Eye tracking as a Tool for deception detection with CIT](https://zenodo.org/records/18525952),
DOI `10.5281/zenodo.18525952`. Its description reports 39 participants.
Download the spreadsheets independently and place them in `data/raw/cit/`:

```text
data/raw/cit/CIT Recording1.xlsx
data/raw/cit/CIT Recording2.xlsx
```

```bash
python scripts/train.py --config configs/cit_baseline.yaml
python -m pip install -e '.[deep]'
python scripts/train.py --config configs/cit_lstm.yaml
```

Available baseline names: `logistic_regression`, `random_forest`, `svm`,
`xgboost`. Available sequence names: `lstm`, `tcn`. Change `model.name` in a
copied config to select a compatible model family. Real-data parsing and
scientific validity limitations are tracked in
[GitHub Issues](https://github.com/terfo1/deception-detection-prototype/issues)
and [known issues](docs/KNOWN_ISSUES.md).

## Automated verification and release

```bash
python -m pip install -e '.[dev,deep]'
python -m pytest --cov=src --cov-report=term-missing
python -m ruff check .
python -m build
```

Tests check metadata preservation, trial-local pupil baselines, window boundaries,
participant separation, Excel column mapping, known metric values, CLI execution,
baseline artifacts, and tiny LSTM/TCN training runs. They assess software behavior;
they do not validate real-world deception detection.

GitHub Actions runs on pushes to `main`, pull requests, version tags, and manual
dispatch. Core tests run on Python 3.10/pandas 2 and Python 3.12/pandas 3. A separate
CPU job tests PyTorch models. Reports, demo outputs, and package distributions
are retained as workflow artifacts. A `v*` tag publishes a GitHub Release only
after the core, sequence, and package jobs succeed. No PyPI account is needed.

See [the project report](docs/PROJECT_REPORT.md) for the assignment requirements,
architecture, test strategy, and release process; see
[the Russian submission guide](docs/ASSIGNMENT_RU.md) for a concise checklist.

## Development context and dissertation plan

For continued development, start with [AGENTS.md](AGENTS.md), the
[project context](docs/PROJECT_CONTEXT.md), and
[current work status](docs/WORK_STATUS.md). Accepted decisions and open proposals
are tracked in [the decision log](docs/DECISIONS.md).
The [dissertation implementation plan](docs/THESIS_IMPLEMENTATION_PLAN_RU.md)
describes the work required for the intended streaming research tool;
planned capabilities are not claims about the current prototype.
Use the [experiment record template](docs/templates/EXPERIMENT_RECORD.md)
for reproducible research runs.

## License and contribution

The software is distributed under the [MIT License](LICENSE). External datasets
retain their own licenses and access conditions; MIT does not relicense them.
Raw participant files, local environments, and model outputs are excluded from
the repository. See [CONTRIBUTING.md](CONTRIBUTING.md) for the Git/issue workflow.
