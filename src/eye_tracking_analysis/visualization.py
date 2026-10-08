from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .analysis import AnalysisResult
from .validation import IDS


def plot_results(result: AnalysisResult, path: str | Path) -> None:
    """Render headlessly; every trial gets an independent trace."""
    figure = Figure(figsize=(10, 6), layout="constrained")
    FigureCanvasAgg(figure)
    trajectory, timeline = figure.subplots(1, 2)
    for key, group in result.samples.groupby(IDS, sort=False):
        label = "/".join(map(str, key))
        # Break trajectories at acquisition gaps as well as missing samples.
        xy = group[["gaze_x", "gaze_y"]].copy()
        if "time_s" in group:
            gaps = group["time_s"].diff() > result.metadata["config"]["max_gap_s"]
            xy.loc[gaps] = float("nan")
        trajectory.plot(xy["gaze_x"], xy["gaze_y"], marker=".", label=label)
        if "time_s" in group:
            timeline.plot(group["time_s"], xy["gaze_x"], label=label)
    unit = result.metadata["coordinate_unit"]
    trajectory.set(xlabel=f"Gaze x ({unit})", ylabel=f"Gaze y ({unit})", title="Trial trajectories")
    timeline.set(xlabel="Time (s)", ylabel=f"Gaze x ({unit})", title="Recorded timestamps only")
    trajectory.legend(fontsize=7)
    if "time_s" not in result.samples:
        timeline.text(0.5, 0.5, "No timestamps: temporal features unavailable", ha="center", wrap=True,
                      transform=timeline.transAxes)
    figure.suptitle("Offline eye-tracking analysis - descriptive engineering estimates")
    figure.savefig(path, dpi=160)
