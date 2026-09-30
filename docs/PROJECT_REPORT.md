# Scientific software module: project report

## Purpose and scope

The module converts eye-tracking tables into trial features or sequence windows,
trains a classifier, and produces reproducible evaluation files. This supports
scientific work on concealed knowledge under a controlled CIT protocol. A small
synthetic experiment makes installation and software behavior independently
checkable without distributing participant data.

The synthetic example includes a deliberately generated class signal. Its
accuracy verifies execution and cannot establish experimental detection accuracy.
Real-data limitations and planned improvements are documented in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Assignment requirements and evidence

| Requirement | Implementation | Evidence |
|---|---|---|
| Use Git version control | `main` branch, descriptive commits, branch/PR guidance, version tags, exclusions for generated/raw files | `git log --oneline`; `.gitignore`; `CONTRIBUTING.md` |
| Host on GitHub with README, license, issues | Public source repository, installation and usage documentation, MIT software license, issue forms, research backlog | [Repository](https://github.com/terfo1/deception-detection-prototype), [issues](https://github.com/terfo1/deception-detection-prototype/issues) |
| Basic CI/CD | Matrix lint/tests, optional sequence training tests, package build, release gated on successful jobs | `.github/workflows/ci.yml`; [Actions](https://github.com/terfo1/deception-detection-prototype/actions) |
| Justify technologies | Explain language, numerical libraries, modeling tools, configs, tests, and hosting choices | README technology table and discussion below |
| Apply automated testing | pytest regression and integration tests, coverage/JUnit reports, Ruff checks | `tests/`; downloadable workflow artifacts |

## Design and technology choices

Python fits this module because its scientific ecosystem allows tabular
processing, statistics, and model evaluation in one language. NumPy handles
arrays and windows; pandas handles heterogeneous tables and grouped transforms.
openpyxl supports the dataset's existing XLSX exports, avoiding a manual format
conversion step. YAML configurations record experiment choices separately from
the processing implementation.

scikit-learn supplies established baseline classifiers and metrics, so the module
does not need custom implementations of logistic regression, random forests, or
SVM. PyTorch supplies sequence models and automatic differentiation, but remains
an optional extra because the basic processing workflow does not require it.
Matplotlib produces headless figures usable in reports and automated runners.

pytest supports data fixtures, parametrized checks, temporary output directories,
and subprocess integration tests. pytest-cov produces coverage reports; Ruff
checks common source errors. GitHub Actions runs these checks directly beside
the source repository and stores reports and distributable package files.
An offline module needs no application server, web framework, or database.

## Data flow and outputs

`table or synthetic generator -> canonical schema -> per-trial preprocessing ->
aggregate features or windows -> participant split -> training -> metrics,
figures, checkpoint`

Participant and trial IDs identify grouping boundaries. Baseline correction uses
physical pupil diameters before optional z-scoring. Window construction stays
inside each trial; the default participant split keeps all examples from a given
participant in the same subset.

The corrected metadata handling uses a Series transform, which keeps the original
columns on both pandas 2 and 3. A regression test requires all six synthetic
trial groups to survive preprocessing and produce six aggregate examples.

The baseline outputs a JSON metrics file, confusion matrix figure, optional ROC
figure, and a pickle containing the model and ordered feature columns. The
sequence pipeline produces metrics/figures and a weight checkpoint. Complete
sequence inference packaging remains a documented improvement.

## Verification strategy

Tests use small synthetic data and temporary directories so the real dataset is
not needed, and existing experimental outputs are not overwritten.

| Check | What failure it detects |
|---|---|
| Metadata and feature-count regression | Participant/trial loss or collapsed segments |
| Trial-local baseline and transform order | Cross-trial normalization or division after z-scoring |
| Validity filtering/interpolation | Unexpected sample retention or missing values |
| Window boundaries and invalid parameters | Windows crossing trials or invalid iteration settings |
| Participant split | Leakage between train, validation, and test participants |
| Small XLSX adapter fixture | Broken column, group, validity, or stimulus mapping |
| Known metric example | Incorrect confusion matrix or accuracy calculation |
| Baseline integration tests | Failed training/export or an unusable saved checkpoint |
| CLI subprocess test | Broken documented invocation |
| LSTM/TCN integration tests | Failed forward/backward training or missing checkpoint |

Test success demonstrates software behavior. Scientific performance still needs
the full dataset, protocol-based segmentation, informative AOI features, and a
validation design that estimates participant-level generalization.

## CI/CD process

Continuous integration runs on `main` pushes, pull requests, manual dispatch,
and `v*` tags. Two core environments exercise Python 3.10 with pandas 2 and
Python 3.12 with pandas 3. Another job installs CPU PyTorch and runs tiny sequence
training tests. A packaging job builds a wheel and source distribution.

Core jobs upload JUnit and coverage reports and synthetic demo outputs. The
package job uploads distributions. On version tags, the release job waits for
all three prerequisite jobs to succeed and then publishes the tested wheel and
source archive to GitHub Releases using the repository's built-in token.
The token has read-only contents permission by default; only the release job
receives contents write permission.

This is continuous delivery of a scientific Python package through GitHub
Releases. Installing the package or deploying it into a separate service is
outside this offline tool's scope.

The workflow follows the official
[GitHub Python build/testing guide](https://docs.github.com/actions/automating-builds-and-tests/building-and-testing-python)
and [workflow artifact documentation](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts).
Test invocation follows [pytest documentation](https://docs.pytest.org/en/stable/how-to/usage.html).

## Data provenance and licensing

Real eye-tracking exports originate from
[Zenodo DOI 10.5281/zenodo.18525952](https://zenodo.org/records/18525952).
Their distribution conditions are separate from the MIT software license.
The repository excludes raw participant data, environments, generated metrics,
figures, and checkpoints. Readers can run the synthetic experiment immediately
and obtain real data independently from the source.

## Local acceptance results

Verified on Windows with Python 3.12.4 and pandas 3.0.2 on 2026-09-30:

- `python -m pytest --cov=src --cov-report=term`: 22 tests passed;
  source coverage was 84%, including tiny LSTM and TCN training runs.
- `python -m ruff check src scripts tests main.py`: passed.
- `python -m build`: produced a wheel and a source distribution.
- The documented synthetic CLI processed 24,576 samples into 192 trial examples,
  with 128 training, 24 validation, and 40 test examples. Both classes were
  present in all three subsets.

Coverage is measured software execution, not a claim that all scientific
assumptions are tested. The hosted run results are available in
[GitHub Actions](https://github.com/terfo1/deception-detection-prototype/actions).
