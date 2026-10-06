import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from src.agents.presentation import PresentationScenario, render_presentation


def test_artifact_text_cannot_escape_embedded_json(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    malicious = '</script><script>alert("raw")</script>&\u2028'
    for name, value in (("report.json", {"warning": malicious}), ("logs.json", []),
                        ("agent-stats.json", []), ("provenance.json", {})):
        (source / name).write_text(json.dumps(value), encoding="utf-8")
    destination = tmp_path / "demo.html"
    render_presentation((PresentationScenario(malicious, "description", source),), destination)
    html = destination.read_text(encoding="utf-8")
    encoded = re.search(r'<script type="application/json" id="demo-data">(.*?)</script>', html, re.S).group(1)
    assert "</script>" not in encoded
    assert json.loads(encoded)[0]["report"]["warning"] == malicious
    with pytest.raises(FileExistsError):
        render_presentation((PresentationScenario("title", "description", source),), destination)


def test_presentation_example_runs_three_real_scenarios_and_is_portable(tmp_path):
    output = tmp_path / "demo"
    result = subprocess.run([sys.executable, "examples/multi_agent_demo.py", "--present", "--output-dir", str(output)],
                            capture_output=True, text=True, timeout=90, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    html = (output / "presentation.html").read_text(encoding="utf-8")
    encoded = re.search(r'<script type="application/json" id="demo-data">(.*?)</script>', html, re.S).group(1)
    scenarios = json.loads(encoded)
    assert len(scenarios) == 3
    assert [scenario["report"]["status"] for scenario in scenarios] == ["COMPLETED", "FAILED", "FAILED"]
    assert scenarios[0]["report"]["verification"]["status"] == "INCONCLUSIVE"
    assert scenarios[1]["report"]["quality"]["status"] == "REJECTED"
    assert scenarios[1]["report"]["prediction"] is None
    assert scenarios[2]["report"]["errors"][-1]["code"] == "MODEL_CHECKSUM_MISMATCH"
    assert scenarios[2]["report"]["prediction"] is None
    assert "raw_data" not in encoded
    assert "<script src=" not in html and 'href="https://' not in html
    for scenario, directory in zip(scenarios, ("workflow", "rejected-signal", "model-error")):
        report = json.loads((output / directory / "report.json").read_text(encoding="utf-8"))
        assert scenario["report"] == report
    assert max(row["percentage"] for row in scenarios[0]["statistics"]) < 40
    assert (Path("src/agents") / "presentation.html").exists()
