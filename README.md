# Eye-tracking Scientific Processing Prototype

[![CI](https://github.com/terfo1/deception-detection-prototype/actions/workflows/ci.yml/badge.svg)](https://github.com/terfo1/deception-detection-prototype/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Python research prototype supporting the thesis topic **Neural Network Analysis
of Real-Time Eye-Tracking Data for Detecting Deceptive Statements**. It contains
offline gaze analysis, baseline and sequence-model training, saved-model inference,
synthetic demonstrations, automated tests, and a configured GitHub Actions workflow.

The intended main research target is truth/lie at statement level; a suitable raw
dataset and protocol remain open. The existing CIT adapter concerns concealed
recognition, a separate target. Model outputs are experimental probability estimates,
not proof that a person lied. Physical live acquisition and a fully causal pipeline
are not implemented. Local verification does not establish a successful GitHub run.

## Assignment 3: offline eye-tracking analysis

The reusable [analysis package](src/eye_tracking_analysis/) adds strict CSV loading,
validation, bounded interpolation, timestamp-aware smoothing, fixed screen geometry
normalization, estimated fixation/saccade events, dispersion, optional observed blink
onsets, headless visualization, and CSV/JSON export. It uses existing NumPy, pandas,
and Matplotlib dependencies. It does not change the legacy training algorithms.

After installation (`python -m pip install -e '.[dev]'`), run from the repository root:

```powershell
# Windows, existing project environment
.venv/Scripts/python.exe examples/eye_tracking_analysis_demo.py
.venv/Scripts/python.exe -m pytest tests/test_eye_tracking_analysis.py
```

```bash
# Linux/macOS, existing project environment
.venv/bin/python examples/eye_tracking_analysis_demo.py
.venv/bin/python -m pytest tests/test_eye_tracking_analysis.py
```

The example creates a **new unique** `outputs/assignment-3-demo-<id>/` directory.
Its input is explicitly synthetic: 42 samples at 100 Hz, two stable plateaus and
one known 3-4-5 displacement. Expected results are two estimated fixations
(mean duration 0.195 s), one estimated saccade (amplitude 5 degrees,
peak velocity 250 degrees/s), and dispersion approximately 2.469341 degrees.
No blink rate is produced because no blink observations are supplied.

### Data format and running the module

UTF-8 CSV: `participant_id`, `session_id`, `trial_id`, `gaze_x`, `gaze_y` are
required. IDs must be explicit and nonempty; leading zeroes are preserved.
Optional `timestamp` is strictly increasing within each participant/session/trial;
declare seconds or milliseconds explicitly. Optional `blink` must contain complete
0/1 observations. Empty gaze cells are missing, not zero and not evidence of a blink.
Labels are not required or used. Additional columns, including `statement_id`, are
preserved; the caller must assign a distinct `trial_id` to each independent statement.

```csv
participant_id,session_id,trial_id,timestamp,gaze_x,gaze_y
synthetic-001,synthetic-session,synthetic-trial,0,0,0
synthetic-001,synthetic-session,synthetic-trial,10,0,0
```

```bash
python -m src.eye_tracking_analysis.cli path/to/gaze.csv --output-dir outputs/my-new-run --coordinate-unit degrees --timestamp-unit ms --velocity-threshold 30
```

The same CLI is installed as `eye-tracking-analysis`. For pixel-to-screen
normalization, use `--coordinate-unit pixels --screen-size 1920 1080` and specify
the threshold in **normalized units/s**. No data-fitted scaling is performed.
Optional `--interpolate-max-s 0.04` fills only complete, short, internally bounded
gaze gaps; `--smoothing-ms 20` enables a trailing time-window mean using recorded
timestamps. Acquisition gaps exceeding `--max-gap-s` (default 0.1 s) reset events
and smoothing. Preprocessing never crosses participant/session/trial boundaries.

Output: `samples.csv`, `features.csv`, `events.csv`, `analysis.json`,
`gaze_analysis.png`. JSON records input hash, configuration, units, versions,
module hashes, and limitations. Existing output directories are rejected.
Without timestamps, only sample quality and spatial dispersion are available;
temporal features are blank, and temporal preprocessing is rejected.

### Research limitations

Event detection groups consecutive low/high-velocity sample intervals. Thresholds
and minimum durations are explicit engineering choices, not validated physiology.
Durations use first-to-last timestamps without extrapolating a final sample.
Blink rate counts observed 0-to-1 transitions per minute of recorded intervals,
excluding acquisition gaps; initial/after-gap blink states are left-censored.
Interpolation uses future samples, so the complete module is **offline**.
There is no truth/lie accuracy, device test, or measured streaming latency.

Detailed methods, technology alternatives, verification, report locations, and
publication checklist: [Assignment 3 guide](docs/assignment_3/GUIDE_RU.md).
The editable English report source is
[ASSIGNMENT_3_REPORT_EN.md](docs/assignment_3/ASSIGNMENT_3_REPORT_EN.md);
[Issue drafts](docs/assignment_3/ISSUES.md) are prepared locally, not published.

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
| Strict offline event analysis, CSV/JSON export and plots | `src/eye_tracking_analysis/` |
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

GitHub Actions is configured for pushes to `main` and `feature/assignment-3`, pull requests to `main`, version tags, and manual
dispatch. Core tests run on Python 3.10/pandas 2 and Python 3.12/pandas 3. A separate
CPU job tests PyTorch models on manual dispatch or version tags; ordinary pushes
and pull requests use the core environment without PyTorch/model downloads.
Both synthetic demos execute in the core job. Reports, demo outputs, and package
distributions are configured as artifacts. A `v*` tag can publish a GitHub Release
only after checks succeed and repository variable `ENABLE_RELEASE` equals `true`.
No release was published by this assignment; the updated remote workflow has not
been run or verified. No PyPI account is needed.

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
