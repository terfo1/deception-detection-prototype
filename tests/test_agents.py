from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from src.agents.cli import main as cli_main
from src.agents.orchestrator import Orchestrator
from src.agents.schemas import (
    AgentResult, AgentTask, EyeSample, EyeTrackingWindow, FeatureVector, PredictionResult, QualityStatus,
    SessionRequest, Step, VerificationStatus, WorkflowConfig, WorkflowStatus,
)
from src.agents.services import FileEyeTrackingSource, ModelSpec, SignalQualityService
from src.agents.telemetry import ExecutionJournal, StageFailure, ToolContext
from src.data.adapters import CITAdapter
from src.data.synthetic import make_synthetic_eye_tracking_dataset
from src.features.aggregated import create_aggregated_features
from src.features.preprocessing import preprocess_eye_tracking
from src.training.pipelines import make_split_indices, train_baseline_pipeline
from src.utils.config import load_config


@pytest.fixture(scope="module")
def resources(tmp_path_factory):
    root = tmp_path_factory.mktemp("agent-integration")
    config = load_config("configs/synthetic_baseline.yaml")
    config["dataset"].update(trials_per_participant=2, samples_per_trial=32)
    config["training"]["output_dir"] = str(root / "training")
    train_baseline_pipeline(config)
    raw = make_synthetic_eye_tracking_dataset(24, 2, 32, seed=42)
    features = create_aggregated_features(preprocess_eye_tracking(raw, config["preprocessing"]))
    train, val, test = make_split_indices(config, features["participant_id"].to_numpy())
    selected = features.iloc[test[0]]
    assert selected["participant_id"] not in set(features.iloc[np.r_[train, val]]["participant_id"])
    frame = raw[(raw["participant_id"] == selected["participant_id"]) & (raw["trial_id"] == selected["trial_id"])].copy()
    frame["session_id"] = "test-session-private"
    path = root / "window.csv"
    frame.drop(columns="label").to_csv(path, index=False)
    checkpoint = root / "training/checkpoints/logistic_regression.pkl"
    spec = ModelSpec("baseline", str(checkpoint), hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     "synthetic-test", "synthetic_debug", ((0, "debug class 0"), (1, "debug class 1")),
                     "ms", config["preprocessing"])
    request = SessionRequest("test-session-private", selected["participant_id"], selected["trial_id"], "ms")
    return path, spec, request, frame, config


def run(resources, *, config=None, spec=None, source=None):
    path, original_spec, request, *_ = resources
    workflow = Orchestrator(source or FileEyeTrackingSource(path), spec or original_spec,
                            config or WorkflowConfig(min_model_confidence=0.0))
    return workflow, workflow.run(request)


def test_success_reuses_pipeline_and_model_without_ground_truth(resources):
    workflow, state = run(resources)
    assert state.status == WorkflowStatus.COMPLETED
    assert state.quality_report.status == QualityStatus.ACCEPTED
    assert state.verification.status == VerificationStatus.ACCEPTED
    assert state.report.prediction == state.prediction
    assert state.step_count == 6
    assert "label" not in state.features.names
    _, spec, _, frame, _ = resources
    expected = create_aggregated_features(preprocess_eye_tracking(frame, spec.preprocessing), include_labels=False)
    labeled = create_aggregated_features(preprocess_eye_tracking(frame, spec.preprocessing))
    assert labeled.columns[:4].tolist() == ["participant_id", "trial_id", "label", "stimulus_id"]
    np.testing.assert_allclose(state.features.values, expected[list(state.features.names)].to_numpy())
    with Path(spec.checkpoint).open("rb") as handle:
        model = pickle.load(handle)["model"]
    np.testing.assert_allclose([v for _, v in state.prediction.probabilities], model.predict_proba(state.features.values)[0])
    assert len(workflow.journal.events) == 14


def test_model_class_order_must_match_checkpoint(resources):
    spec = replace(resources[1], class_meanings=((1, "debug 1"), (0, "debug 0")))
    _, state = run(resources, spec=spec)
    assert state.status == WorkflowStatus.FAILED
    assert state.report.errors[-1].code == "MODEL_CLASS_CONTRACT_MISMATCH"


def test_statistics_and_privacy_safe_logs(resources, caplog):
    workflow, state = run(resources)
    rows = {row.agent_name: row for row in workflow.journal.statistics()}
    assert rows["Orchestrator"].total_calls == 1
    assert rows["FeatureExtractionAgent"].total_calls == 3
    assert len(rows) == 7
    assert max(row.percentage for row in rows.values()) < 40
    assert sum(row.percentage for row in rows.values()) == pytest.approx(100)
    for event in workflow.journal.events:
        assert event.duration_ms >= 0
        assert event.finish_time >= event.start_time
        assert event.status == "SUCCESS"
    agent_events = [event for event in workflow.journal.events if event.kind == "agent"]
    assert len(agent_events) == 7
    assert next(e for e in agent_events if e.agent_name == "FeatureExtractionAgent").tools_called == (
        "features.preprocess", "features.extract"
    )
    assert state.request.session_id not in caplog.text
    assert "gaze_x" not in caplog.text


def rejected_source(resources, tmp_path, refreshable=False):
    _, _, _, frame, _ = resources
    frame = frame.copy()
    frame.loc[:, ["gaze_x", "gaze_y", "pupil"]] = np.nan
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    source = FileEyeTrackingSource(path)
    source.can_refresh = refreshable
    return source


def test_static_invalid_data_fails_without_pointless_retry(resources, tmp_path):
    _, state = run(resources, source=rejected_source(resources, tmp_path))
    assert state.status == WorkflowStatus.FAILED
    assert state.quality_report.status == QualityStatus.REJECTED
    assert state.retry_count == 0
    assert state.prediction is None
    assert state.report.status == WorkflowStatus.FAILED


def test_refreshable_source_stops_after_three_rejections(resources, tmp_path):
    workflow, state = run(resources, source=rejected_source(resources, tmp_path, True),
                          config=WorkflowConfig(max_retries=2))
    assert state.retry_count == 2
    assert state.step_count == 7  # Three acquisitions/checks and one terminal report.
    assert state.status == WorkflowStatus.FAILED
    assert len([e for e in workflow.journal.events if e.tool == "source.acquire"]) == 3
    assert {e.retry_attempt for e in workflow.journal.events if e.tool == "source.acquire"} == {0, 1, 2}


def test_recoverable_source_error_can_retry_but_is_visible(resources):
    path, *_ = resources

    class RecoveringSource(FileEyeTrackingSource):
        can_refresh = True
        calls = 0

        def acquire(self, request):
            self.calls += 1
            if self.calls == 1:
                raise StageFailure("TRANSIENT_SOURCE", recoverable=True)
            return super().acquire(request)

    _, state = run(resources, source=RecoveringSource(path))
    assert state.status == WorkflowStatus.COMPLETED
    assert state.retry_count == 1
    assert state.verification.status == VerificationStatus.INCONCLUSIVE
    assert state.report.errors[0].code == "TRANSIENT_SOURCE"


def test_low_model_confidence_and_unset_policy_are_inconclusive(resources):
    _, successful = run(resources)
    threshold = (1 + successful.prediction.confidence) / 2
    assert successful.prediction.confidence < threshold
    _, state = run(resources, config=WorkflowConfig(min_model_confidence=threshold))
    assert state.verification.status == VerificationStatus.INCONCLUSIVE
    assert "below_configured_model_confidence" in state.verification.reasons
    _, unset = run(resources, config=WorkflowConfig())
    assert unset.verification.status == VerificationStatus.INCONCLUSIVE


def test_model_failure_is_controlled_and_logged(resources):
    workflow, state = run(resources, spec=replace(resources[1], checkpoint_sha256="0" * 64))
    assert state.status == WorkflowStatus.FAILED
    assert state.retry_count == 0
    assert state.prediction is None
    assert state.report.errors[-1].code == "MODEL_CHECKSUM_MISMATCH"
    failures = [e for e in workflow.journal.events if e.status == "FAILED"]
    assert {e.agent_name for e in failures} == {"PredictionAgent", "Orchestrator"}
    assert any(e.tool == "model.predict" for e in failures)


def test_feature_failure_is_controlled_without_retry(resources):
    workflow, state = run(resources, spec=replace(resources[1], preprocessing={"mode": "invalid"}))
    assert state.status == WorkflowStatus.FAILED
    assert state.retry_count == 0
    assert state.report.errors[-1].step == Step.FEATURE_EXTRACTION
    assert any(e.agent_name == "FeatureExtractionAgent" and e.status == "FAILED" for e in workflow.journal.events)


@pytest.mark.parametrize("limit", [1, 3, 5])
def test_max_steps_reserves_terminal_report(resources, limit):
    _, state = run(resources, config=WorkflowConfig(max_workflow_steps=limit))
    assert state.status == WorkflowStatus.FAILED
    assert state.step_count == limit
    assert state.report.errors[-1].code == "MAX_WORKFLOW_STEPS"


def test_deadline_expiry_still_produces_failure_report(resources):
    _, state = run(resources, config=WorkflowConfig(deadline_seconds=1e-12))
    assert state.status == WorkflowStatus.FAILED
    assert state.report.errors[-1].code == "WORKFLOW_DEADLINE"


def test_deadline_overrun_discards_late_source_result(resources):
    path, *_ = resources

    class SlowSource(FileEyeTrackingSource):
        def acquire(self, request):
            result = super().acquire(request)
            time.sleep(0.03)
            return result

    _, state = run(resources, source=SlowSource(path), config=WorkflowConfig(deadline_seconds=0.01))
    assert state.raw_data is None
    assert state.report.errors[-1].code == "WORKFLOW_DEADLINE"


@pytest.mark.parametrize("change", ["session", "statement", "disjoint", "stimulus"])
def test_source_never_combines_boundaries(resources, tmp_path, change):
    _, _, request, frame, _ = resources
    frame = frame.copy()
    if change == "session":
        frame.loc[16:, "session_id"] = "other-session"
        # No matching segment should be combined with an identically named trial in another session.
        request = replace(request, session_id="absent")
    elif change == "statement":
        frame["statement_id"] = ["a"] * 16 + ["b"] * 16
    elif change == "stimulus":
        frame.loc[frame.index[16:], "stimulus_id"] = "other-stimulus"
    else:
        frame.loc[frame.index[16], "trial_id"] = "another-trial"
    path = tmp_path / "boundaries.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(StageFailure):
        FileEyeTrackingSource(path).acquire(request)


def test_labels_do_not_affect_predictions_and_ids_are_not_guessed(resources, tmp_path):
    path, spec, request, frame, _ = resources
    _, base = run(resources)
    for value in [0, 1, "unknown", None]:
        altered = frame.copy()
        altered["label"] = value
        source = tmp_path / f"guilty_target_{value}.csv"
        altered.to_csv(source, index=False)
        _, state = run(resources, source=FileEyeTrackingSource(source))
        assert state.prediction.probabilities == base.prediction.probabilities
    missing = tmp_path / "missing-boundaries.csv"
    frame.drop(columns=["participant_id", "trial_id"]).to_csv(missing, index=False)
    _, state = run(resources, source=FileEyeTrackingSource(missing))
    assert state.status == WorkflowStatus.FAILED
    # CIT's training hook must not derive a class or inference boundaries from its filename.
    normalized = CITAdapter().normalize_schema(pd.DataFrame({"gaze_x": [1]}),
                                               Path("guilty.xlsx"), for_inference=True)
    assert "label" not in normalized
    assert normalized["participant_id"].isna().all()


def test_quality_reports_measured_checks_and_missing_parameters(resources):
    path, _, request, *_ = resources
    window = FileEyeTrackingSource(path).acquire(request)
    service = SignalQualityService(WorkflowConfig(gaze_bounds=(-1, 1, -1, 1)), ("timestamp", "gaze_x", "gaze_y"))
    samples = list(window.samples)
    samples[0] = replace(samples[0], gaze_x=1000, pupil=-1)
    quality = service.assess(replace(window, samples=tuple(samples)))
    assert quality.valid_samples == 31
    assert quality.status == QualityStatus.WARNING
    assert quality.sampling_hz == pytest.approx(1000 / 16.67)
    assert quality.accumulation_time_ms == pytest.approx(31 * 16.67)
    samples[1] = replace(samples[1], timestamp=samples[0].timestamp)
    assert service.assess(replace(window, samples=tuple(samples))).status == QualityStatus.REJECTED
    assert "pupil_upper_bound_unspecified" in quality.unavailable_checks


@pytest.mark.parametrize("factory", [
    lambda: WorkflowConfig(max_retries=-1),
    lambda: WorkflowConfig(max_workflow_steps=0),
    lambda: WorkflowConfig(min_model_confidence=1.1),
    lambda: WorkflowConfig(deadline_seconds=float("nan")),
    lambda: WorkflowConfig(gaze_bounds=(1, 0, 0, 1)),
    lambda: SessionRequest("", "p", "t", "ms"),
    lambda: SessionRequest("s", "p", "t", "unknown"),
    lambda: EyeSample(0, float("nan"), 1),
    lambda: FeatureVector("aggregated", ("label",), ((1.0,),), 1),
    lambda: FeatureVector("sequence", ("x", "y"), ((1.0,),), 1),
    lambda: PredictionResult(3, "v", "cit", ((0, "control"), (1, "target")), 0),
    lambda: PredictionResult(0, "v", "cit", ((0, "control"),), 0, ((0, 0.7),), 0.7,
                             "predict_proba_uncalibrated"),
])
def test_typed_contracts_reject_invalid_values(factory):
    with pytest.raises(ValueError):
        factory()


def test_task_payload_and_tool_allowlist_are_enforced():
    request = SessionRequest("s", "p", "t", "ms")
    task = AgentTask("s", Step.PREDICTION, 0)
    with pytest.raises(ValueError):
        AgentResult(task, EyeTrackingWindow(request, (), ()))
    journal = ExecutionJournal("run")
    context = ToolContext(journal, "test", task, float("inf"), ("allowed",))
    with pytest.raises(StageFailure, match="UNREGISTERED_TOOL"):
        context.call("other", lambda: None)
    with pytest.raises(ValueError):
        ToolContext(journal, "test", task, float("inf"), tuple(str(i) for i in range(6)))


def test_unexpected_exception_does_not_expose_raw_data(resources, caplog):
    path, *_ = resources

    class BrokenSource(FileEyeTrackingSource):
        def acquire(self, request):
            raise RuntimeError("sensitive-raw-user-data")

    _, state = run(resources, source=BrokenSource(path))
    assert state.status == WorkflowStatus.FAILED
    assert "sensitive-raw-user-data" not in caplog.text
    assert "sensitive-raw-user-data" not in json.dumps(asdict(state.report))


def test_predict_only_model_does_not_fabricate_confidence(resources, tmp_path):
    from sklearn.svm import SVC

    _, base = run(resources)
    n = len(base.features.names)
    # A genuine estimator with no predict_proba; a tiny synthetic unit fixture.
    model = SVC(probability=False).fit(np.array([np.zeros(n), np.ones(n)]), [0, 1])
    path = tmp_path / "predict-only.pkl"
    path.write_bytes(pickle.dumps({"model": model, "feature_columns": list(base.features.names)}))
    spec = replace(resources[1], checkpoint=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    _, state = run(resources, spec=spec)
    assert state.status == WorkflowStatus.COMPLETED
    assert state.prediction.probabilities is None
    assert state.prediction.confidence is None
    assert state.verification.status == VerificationStatus.INCONCLUSIVE


def test_cli_new_process_exports_report_stats_and_no_samples(resources, tmp_path):
    path, spec, request, *_ = resources
    manifest = asdict(spec)
    manifest["class_meanings"] = dict(spec.class_meanings)
    spec_path = tmp_path / "model.json"
    spec_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "run"
    args = ["--source", str(path), "--model-spec", str(spec_path), "--trusted-model",
            "--session-id", request.session_id, "--participant-id", request.participant_id,
            "--trial-id", request.trial_id, "--timestamp-unit", "ms", "--output-dir", str(out)]
    result = subprocess.run([sys.executable, "-m", "src.agents.cli", *args],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "COMPLETED"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["verification"]["status"] == "INCONCLUSIVE"
    assert "raw_data" not in json.loads((out / "state.json").read_text(encoding="utf-8"))
    assert len(json.loads((out / "logs.json").read_text(encoding="utf-8"))) == 14
    assert json.loads((out / "provenance.json").read_text())["source_sha256"]
    assert cli_main(args) == 2  # Existing run must never be overwritten.
    failure_out = tmp_path / "failed-run"
    args[1] = str(tmp_path / "missing-source.csv")
    args[-1] = str(failure_out)
    assert cli_main(args) == 1
    failure_report = json.loads((failure_out / "report.json").read_text(encoding="utf-8"))
    assert failure_report["status"] == "FAILED"
    assert failure_report["prediction"] is None
    assert failure_report["errors"][-1]["code"] == "SOURCE_IO_ERROR"
