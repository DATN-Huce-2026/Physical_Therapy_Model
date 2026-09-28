from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def canonical_name(value: object) -> str:
    """Đưa tên bài tập/cột về snake_case ổn định."""
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


@dataclass(frozen=True)
class ExerciseConfig:
    """Cấu hình một bài tập ở chế độ một dòng = một repetition."""

    name: str
    feature_columns: tuple[str, ...]
    lower_percentile: float = 5.0
    upper_percentile: float = 95.0
    profile_margin_deg: float = 3.0
    decision_threshold: float = 0.5

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> ExerciseConfig:
        features = tuple(
            canonical_name(column) for column in payload.get("feature_columns", [])
        )
        config = cls(
            name=canonical_name(name),
            feature_columns=features,
            lower_percentile=float(payload.get("lower_percentile", 5.0)),
            upper_percentile=float(payload.get("upper_percentile", 95.0)),
            profile_margin_deg=float(payload.get("profile_margin_deg", 3.0)),
            decision_threshold=float(payload.get("decision_threshold", 0.5)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Tên bài tập không được để trống")
        if not self.feature_columns:
            raise ValueError(f"Bài tập {self.name!r} chưa khai báo feature_columns")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError(f"Bài tập {self.name!r} có feature bị lặp")
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("Percentile phải thỏa 0 <= lower < upper <= 100")
        if self.profile_margin_deg < 0:
            raise ValueError("profile_margin_deg phải >= 0")
        if not 0 < self.decision_threshold < 1:
            raise ValueError("decision_threshold phải nằm trong (0, 1)")


def load_exercise_config(path: str | Path, exercise: str) -> ExerciseConfig:
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)

    normalized = {canonical_name(name): value for name, value in payload.items()}
    exercise_name = canonical_name(exercise)
    if exercise_name not in normalized:
        available = ", ".join(sorted(normalized))
        raise KeyError(
            f"Không có cấu hình cho {exercise!r}. Các bài tập hiện có: {available}"
        )
    return ExerciseConfig.from_dict(exercise_name, normalized[exercise_name])
