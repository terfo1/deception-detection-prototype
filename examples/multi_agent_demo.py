"""Explicit synthetic example. Production workflow never fabricates a source/model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.cli import main as workflow_main
from src.agents.presentation import PresentationScenario, render_presentation
from src.data.synthetic import make_synthetic_eye_tracking_dataset
from src.features.aggregated import create_aggregated_features
from src.features.preprocessing import preprocess_eye_tracking
from src.training.pipelines import make_split_indices, train_baseline_pipeline
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic engineering demonstration, not deception research.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / f"agent-demo-{uuid4().hex}")
    parser.add_argument("--present", action="store_true", help="Generate a portable HTML demo with three real workflows.")
    parser.add_argument("--open", action="store_true", help="Open the HTML demo in the default browser (implies --present).")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = load_config("configs/synthetic_baseline.yaml")
    config["dataset"].update(trials_per_participant=2, samples_per_trial=32)
    config["training"]["output_dir"] = str(root / "training")
    train_baseline_pipeline(config)
    raw = make_synthetic_eye_tracking_dataset(24, 2, 32, seed=config["seed"])
    features = create_aggregated_features(preprocess_eye_tracking(raw, config["preprocessing"]))
    train, val, test = make_split_indices(config, features["participant_id"].to_numpy())
    split = {name: sorted(set(features.iloc[indices]["participant_id"]))
             for name, indices in (("train", train), ("validation", val), ("test", test))}
    (root / "split-manifest.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    selected = features.iloc[test[0]]
    trial = raw[(raw["participant_id"] == selected["participant_id"]) & (raw["trial_id"] == selected["trial_id"])]
    trial.drop(columns=["label"]).to_csv(root / "synthetic-window.csv", index=False)
    checkpoint = root / "training/checkpoints/logistic_regression.pkl"
    manifest = {
        "family": "baseline", "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_version": "synthetic-example-logistic-regression", "target": "synthetic_debug",
        "class_meanings": {"0": "synthetic pattern 0", "1": "synthetic pattern 1"},
        "timestamp_unit": "ms", "preprocessing": config["preprocessing"],
    }
    (root / "model-spec.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = workflow_main([
        "--source", str(root / "synthetic-window.csv"), "--model-spec", str(root / "model-spec.json"),
        "--trusted-model", "--session-id", "synthetic-example", "--participant-id", selected["participant_id"],
        "--trial-id", selected["trial_id"], "--timestamp-unit", "ms", "--output-dir", str(root / "workflow"),
    ])
    if result:
        return result
    if args.present or args.open:
        # Failure fixtures are confined to this explicitly synthetic example.
        bad_signal = trial.drop(columns=["label"]).copy()
        bad_signal.loc[:, ["gaze_x", "gaze_y", "pupil"]] = float("nan")
        bad_signal.to_csv(root / "synthetic-invalid-window.csv", index=False)
        bad_contract = {**manifest, "checkpoint_sha256": "0" * 64}
        (root / "invalid-model-spec.json").write_text(json.dumps(bad_contract, indent=2), encoding="utf-8")
        for source_name, contract_name, scenario_name in (
            ("synthetic-invalid-window.csv", "model-spec.json", "rejected-signal"),
            ("synthetic-window.csv", "invalid-model-spec.json", "model-error"),
        ):
            exit_code = workflow_main([
                "--source", str(root / source_name), "--model-spec", str(root / contract_name), "--trusted-model",
                "--session-id", "synthetic-example", "--participant-id", selected["participant_id"],
                "--trial-id", selected["trial_id"], "--timestamp-unit", "ms",
                "--output-dir", str(root / scenario_name),
            ])
            if exit_code != 1:
                raise RuntimeError("The demonstration failure scenario did not produce a controlled FAILED run.")
        presentation = render_presentation((
            PresentationScenario("1. Нормальный сигнал", "Полный pipeline завершается; проверка остаётся INCONCLUSIVE без научного порога уверенности.", root / "workflow"),
            PresentationScenario("2. Отказ: плохой сигнал", "Пропуски в синтетическом окне блокируют признаки и inference. Report сохраняет причину отказа.", root / "rejected-signal"),
            PresentationScenario("3. Отказ: ошибка модели", "Несовпадение SHA256 checkpoint вызывает контролируемый отказ и запись в журнале.", root / "model-error"),
        ), root / "presentation.html")
        print(json.dumps({"presentation": str(presentation)}, ensure_ascii=False))
    # Example provenance is separate from the inference-only production record.
    (root / "example-provenance.json").write_text(json.dumps({
        "kind": "synthetic_engineering_example", "seed": config["seed"], "training_config": config,
        "split_manifest": "split-manifest.json",
    }, indent=2) + "\n", encoding="utf-8")
    from src.agents.schemas import WorkflowConfig
    print(json.dumps({"engineering_config": asdict(WorkflowConfig()), "root": str(root)}))
    if args.open:
        import webbrowser
        webbrowser.open(presentation.as_uri())
    return result


if __name__ == "__main__":
    raise SystemExit(main())
