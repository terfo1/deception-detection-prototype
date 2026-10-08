# Prepared GitHub Issues — Assignment 3

These are local drafts. No issue numbers or publication are claimed.
After approval, open them in the existing repository; do not attach participant data.

## 1. Validate offline velocity events against annotated eye-tracking events

**Type:** research validation. **Scope:** `src/eye_tracking_analysis/analysis.py`.

The assignment estimates fixation/saccade intervals using an explicit velocity
threshold. Smooth pursuit, device noise, sampling frequency and preprocessing
can affect event identity. Engineering tests verify geometry, not physiological validity.

**Acceptance criteria:** choose a permitted annotated dataset and document rights,
units and event definition; compare estimated boundaries/durations with reference
events; select thresholds without held-out participants; report event errors and
missingness, including negative results. Preserve the existing synthetic oracles.

## 2. Build and test a fully causal gaze preprocessing contract

**Type:** enhancement. **Scope:** future PIPE-01; not implemented by Assignment 3.

Bounded offline interpolation uses a future endpoint. Legacy `online_safe` only
makes smoothing causal. Neither path establishes stream readiness.

**Acceptance criteria:** define available calibration and gap policy; implement
bounded state and resets by participant/session/trial; verify prefix independence
from future samples and identical outputs for different chunk sizes; measure
window accumulation separately from compute latency. Replay is not a physical SDK test.

## 3. Reject unknown training labels and document experimental targets

**Type:** scientific correctness. **Scope:** legacy adapters/training.

Legacy loading/aggregation can default labels to zero or infer CIT labels from
filenames. The new offline analysis does not use labels and does not fix this path.

**Acceptance criteria:** separate truth/lie and CIT schemas; fail training/evaluation
on missing or unsupported labels; preserve label provenance and all boundaries;
retain label-free prediction; test unknown labels and filename independence;
record participant-group splits before learned transforms.

## Suggested pull request description

Add strict offline CSV gaze analysis for Assignment 3, with explicit units,
session/trial boundaries, bounded interpolation, temporal smoothing, estimated
events, quality, headless plots and protected run exports. Add 35 numerical and
boundary tests, execute the synthetic analysis in existing CI, and document the
implementation and research limits in the English report.

Validation: 97 tests passed locally, Ruff and pip check passed. Updated GitHub
workflow execution remains unverified until a permitted push/PR. No participant
data, checkpoints or generated output files belong in the PR.
