"""Portable offline viewer of actual workflow artifacts; does not run inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PresentationScenario:
    title: str
    description: str
    directory: Path


def render_presentation(scenarios: tuple[PresentationScenario, ...], destination: Path) -> Path:
    """Embed reports/logs only, without raw source samples or remote dependencies."""
    if not scenarios:
        raise ValueError("At least one completed workflow artifact directory is required.")
    payload = []
    for scenario in scenarios:
        record = {"title": scenario.title, "description": scenario.description}
        for key, filename in (("report", "report.json"), ("events", "logs.json"),
                              ("statistics", "agent-stats.json"), ("provenance", "provenance.json")):
            record[key] = json.loads((scenario.directory / filename).read_text(encoding="utf-8"))
        if record["report"] is None:
            raise ValueError("Presentation requires an available terminal report.")
        payload.append(record)
    # Prevent HTML/script termination even if artifact labels contain untrusted text.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    for char, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                          ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        encoded = encoded.replace(char, escaped)
    template = Path(__file__).with_name("presentation.html").read_text(encoding="utf-8")
    html = template.replace("__WORKFLOW_PAYLOAD__", encoded)
    # A report is an artifact: do not overwrite a previous exported presentation.
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(html)
    return destination
