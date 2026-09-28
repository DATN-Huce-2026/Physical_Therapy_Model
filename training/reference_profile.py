from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferenceProfile:
    """Khoảng chuẩn cho các feature start/turning/rom của repetition đúng."""

    exercise: str
    feature_names: tuple[str, ...]
    lower_percentile: float
    upper_percentile: float
    margin_degrees: float
    correct_repetitions: int
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray

    def compare(self, feature_values: np.ndarray) -> dict[str, object]:
        values = np.asarray(feature_values, dtype=float)
        expected = (len(self.feature_names),)
        if values.shape != expected:
            raise ValueError(f"Feature có shape {values.shape}, cần {expected}")

        result: dict[str, dict[str, object]] = {}
        for index, feature_name in enumerate(self.feature_names):
            value = values[index]
            lower = self.lower[index]
            upper = self.upper[index]
            median = self.median[index]
            if not np.isfinite(value):
                result[feature_name] = {"status": "unobserved"}
                continue
            if not np.isfinite(lower) or not np.isfinite(upper):
                result[feature_name] = {"status": "profile_unavailable"}
                continue

            if value < lower:
                direction = "lower"
                deviation = value - lower
            elif value > upper:
                direction = "higher"
                deviation = value - upper
            else:
                direction = "within_range"
                deviation = 0.0

            result[feature_name] = {
                "status": "observed",
                "value": round(float(value), 3),
                "direction": direction,
                "signed_deviation_deg": round(float(deviation), 3),
                "reference_lower": round(float(lower), 3),
                "reference_median": round(float(median), 3),
                "reference_upper": round(float(upper), 3),
            }
        return {"exercise": self.exercise, "features": result}

    def to_dict(self) -> dict[str, object]:
        return {
            "exercise": self.exercise,
            "feature_names": list(self.feature_names),
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "margin_degrees": self.margin_degrees,
            "correct_repetitions": self.correct_repetitions,
            "lower": self.lower.tolist(),
            "median": self.median.tolist(),
            "upper": self.upper.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> ReferenceProfile:
        with Path(path).open(encoding="utf-8") as file:
            payload = json.load(file)
        return cls(
            exercise=str(payload["exercise"]),
            feature_names=tuple(payload["feature_names"]),
            lower_percentile=float(payload["lower_percentile"]),
            upper_percentile=float(payload["upper_percentile"]),
            margin_degrees=float(payload["margin_degrees"]),
            correct_repetitions=int(payload["correct_repetitions"]),
            lower=np.asarray(payload["lower"], dtype=np.float32),
            median=np.asarray(payload["median"], dtype=np.float32),
            upper=np.asarray(payload["upper"], dtype=np.float32),
        )


def fit_reference_profile(
    features: np.ndarray,
    labels: np.ndarray,
    exercise: str,
    feature_names: tuple[str, ...],
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    margin_degrees: float = 3.0,
) -> ReferenceProfile:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError("features phải có shape [repetition, feature]")
    correct_features = features[labels == 1]
    if len(correct_features) == 0:
        raise ValueError("Không có repetition đúng (label=1) để xây reference profile")

    lower = np.full(features.shape[1], np.nan, dtype=np.float32)
    median = np.full(features.shape[1], np.nan, dtype=np.float32)
    upper = np.full(features.shape[1], np.nan, dtype=np.float32)
    for index in range(features.shape[1]):
        observed = correct_features[:, index]
        observed = observed[np.isfinite(observed)]
        if len(observed):
            lower[index] = np.percentile(observed, lower_percentile) - margin_degrees
            median[index] = np.median(observed)
            upper[index] = np.percentile(observed, upper_percentile) + margin_degrees

    return ReferenceProfile(
        exercise=exercise,
        feature_names=feature_names,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        margin_degrees=margin_degrees,
        correct_repetitions=len(correct_features),
        lower=lower,
        median=median,
        upper=upper,
    )
