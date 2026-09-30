from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_baseline_model(name: str):
    """Construct a baseline classifier for aggregated features."""
    key = name.lower()
    if key == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )
    if key == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, class_weight="balanced")),
            ]
        )
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError("xgboost is not installed.") from exc
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
        )
    raise ValueError(f"Unsupported baseline model: {name}")
