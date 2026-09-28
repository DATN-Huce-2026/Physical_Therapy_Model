"""Suy luận trực tiếp từ feature của một repetition đang nằm trong bộ nhớ."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExerciseConfig, canonical_name, load_exercise_config
from .model import correct_probabilities, load_classifier
from .reference_profile import ReferenceProfile


class InsufficientObservationError(ValueError):
    """Không đủ feature quan sát được để đưa ra kết quả có ý nghĩa."""


class LiveRepetitionClassifier:
    """Load model một lần và dự đoán trực tiếp, không cần tạo CSV tạm."""

    def __init__(
        self,
        *,
        exercise: str,
        config_path: str | Path,
        artifact_dir: str | Path,
        min_observed_ratio: float = 0.5,
    ) -> None:
        if not 0 < min_observed_ratio <= 1:
            raise ValueError("min_observed_ratio phải nằm trong (0, 1]")

        self.config: ExerciseConfig = load_exercise_config(config_path, exercise)
        model_dir = Path(artifact_dir) / self.config.name
        self.artifact = load_classifier(model_dir / "classifier.joblib")
        self.profile = ReferenceProfile.load(model_dir / "reference_profile.json")
        self.min_observed_ratio = float(min_observed_ratio)

        artifact_features = tuple(self.artifact["feature_names"])
        if artifact_features != self.config.feature_columns:
            raise ValueError(
                f"Feature model {artifact_features} khác config "
                f"{self.config.feature_columns}"
            )
        if tuple(self.profile.feature_names) != self.config.feature_columns:
            raise ValueError("Feature reference_profile khác feature của model")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.config.feature_columns

    def predict(self, feature_values: dict[str, float | None]) -> dict[str, object]:
        """Dự đoán một repetition từ mapping tên feature -> giá trị."""
        canonical_values = {
            canonical_name(name): value for name, value in feature_values.items()
        }
        ordered_values = [
            self._numeric_or_nan(canonical_values.get(name))
            for name in self.feature_names
        ]
        observed_count = int(np.isfinite(ordered_values).sum())
        observed_ratio = observed_count / len(ordered_values)
        if observed_ratio < self.min_observed_ratio:
            raise InsufficientObservationError(
                f"Chỉ quan sát được {observed_count}/{len(ordered_values)} feature "
                f"({observed_ratio:.0%}); cần ít nhất {self.min_observed_ratio:.0%}"
            )

        features = pd.DataFrame([ordered_values], columns=self.feature_names)
        probability = float(correct_probabilities(self.artifact["model"], features)[0])
        threshold = float(self.artifact["decision_threshold"])
        correct = probability >= threshold
        raw_values = np.asarray(ordered_values, dtype=float)

        return {
            "exercise": self.config.name,
            "label": int(correct),
            "correct": bool(correct),
            "correct_probability": round(probability, 6),
            "decision_threshold": threshold,
            "observed_features": observed_count,
            "total_features": len(ordered_values),
            "observed_feature_ratio": round(observed_ratio, 6),
            "reference_comparison": self.profile.compare(raw_values),
        }

    @staticmethod
    def _numeric_or_nan(value: object) -> float:
        if value is None:
            return float("nan")
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return number if np.isfinite(number) else float("nan")
