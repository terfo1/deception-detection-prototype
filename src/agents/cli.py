from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from uuid import uuid4

import yaml

from src.agents.orchestrator import Orchestrator
from src.agents.schemas import SessionRequest, WorkflowConfig, WorkflowStatus
from src.agents.services import FileEyeTrackingSource, ModelSpec, cit_file_source


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded offline multi-agent eye-tracking workflow.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-format", choices=("csv", "cit-xlsx"), default="csv")
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--trusted-model", action="store_true", required=True,
                        help="Confirm that the model checkpoint comes from a trusted source (pickle can execute code).")
    parser.add_argument("--config", type=Path, default=Path("configs/multi_agent.yaml"))
    parser.add_argument("--session-id", required=True, help="Research session ID; declared if CSV has no session_id.")
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--timestamp-unit", choices=("s", "ms", "us"), required=True)
    parser.add_argument("--question-id")
    parser.add_argument("--statement-id")
    parser.add_argument("--output-dir", type=Path, help="New directory only; defaults to outputs/agents-<UUID>.")
    args = parser.parse_args(argv)
    try:
        cfg = WorkflowConfig(**yaml.safe_load(args.config.read_text(encoding="utf-8")))
        spec = ModelSpec.load(args.model_spec)
        request = SessionRequest(args.session_id, args.participant_id, args.trial_id, args.timestamp_unit,
                                 args.question_id, args.statement_id)
        source = cit_file_source(args.source) if args.source_format == "cit-xlsx" else FileEyeTrackingSource(args.source)
        workflow = Orchestrator(source, spec, cfg)
        output = args.output_dir or Path("outputs") / f"agents-{uuid4().hex}"
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "FAILED", "error": f"SETUP_{type(exc).__name__}"}), file=sys.stderr)
        return 2
    state = workflow.run(request)
    snapshot = asdict(state)
    snapshot.pop("raw_data")  # Raw samples never become ordinary logs or default artifacts.
    write_json(output / "state.json", snapshot)
    write_json(output / "report.json", asdict(state.report) if state.report else None)
    write_json(output / "logs.json", [asdict(event) for event in workflow.journal.events])
    write_json(output / "agent-stats.json", [asdict(row) for row in workflow.journal.statistics()])
    write_json(output / "provenance.json", {
        "run_id": workflow.journal.run_id, "config": asdict(cfg), "model_contract": asdict(spec),
        "source_format": args.source_format,
        "source_sha256": source.source_sha256,
        "python": platform.python_version(),
        "versions": {name: __import__(name).__version__ for name in ("numpy", "pandas", "sklearn", "yaml")},
        "seed": None, "split_manifest": None, "mode": "inference_only_no_training_no_evaluation",
    })
    print(json.dumps({"status": state.status.value, "output_dir": str(output), "run_id": workflow.journal.run_id}))
    return 0 if state.status == WorkflowStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
