from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExerciseConfig, canonical_name


class DatasetError(ValueError):
    """Dữ liệu CSV không đúng schema huấn luyện."""


@dataclass(frozen=True)
class PreparedFeatures:
    features: pd.DataFrame
    metadata: tuple[dict[str, object], ...]
    feature_names: tuple[str, ...]
    numeric_feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class PreparedDataset:
    features: pd.DataFrame
    labels: np.ndarray
    metadata: tuple[dict[str, object], ...]
    feature_names: tuple[str, ...]
    numeric_feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]

    @property
    def numeric_features(self) -> pd.DataFrame:
        return self.features.loc[:, self.numeric_feature_names]

    @property
    def missing_cell_ratio(self) -> float:
        return float(self.numeric_features.isna().to_numpy().mean())

    @property
    def rows_with_missing_ratio(self) -> float:
        return float(self.numeric_features.isna().any(axis=1).mean())


def _read_csv(path: str | Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [canonical_name(column) for column in frame.columns]
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise DatasetError(f"{source} có cột bị lặp: {duplicated}")
    return frame


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DatasetError(f"{source} thiếu các cột bắt buộc: {missing}")


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["exercise"] = result["exercise"].map(canonical_name)
    result["rep_id"] = result["rep_id"].astype(str).str.strip()
    if (result["rep_id"] == "").any():
        raise DatasetError("rep_id không được để trống")
    return result


def _check_unique_keys(frame: pd.DataFrame, source: str) -> None:
    duplicated = frame.duplicated(["exercise", "rep_id"], keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, ["exercise", "rep_id"]].head(5)
        raise DatasetError(
            f"{source} có repetition bị lặp: {examples.to_dict('records')}"
        )


def _prepare_feature_frame(
    angles_path: str | Path,
    config: ExerciseConfig,
) -> tuple[pd.DataFrame, tuple[dict[str, object], ...]]:
    source = f"angles ({angles_path})"
    angles = _normalize_keys(_read_csv(angles_path, source))
    required = {"exercise", "rep_id", *config.model_columns}
    _require_columns(angles, required, source)
    _check_unique_keys(angles, source)

    angles = angles.loc[angles["exercise"] == config.name].copy()
    if angles.empty:
        raise DatasetError(
            f"Không có dữ liệu cho bài tập {config.name!r} trong {source}"
        )

    features = angles.loc[:, config.model_columns].copy()
    for column in config.feature_columns:
        original = features[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = (
            original.notna() & original.astype(str).str.strip().ne("") & numeric.isna()
        )
        if invalid.any():
            examples = original.loc[invalid].head(5).tolist()
            raise DatasetError(f"Cột {column!r} chứa giá trị không phải số: {examples}")
        features[column] = numeric.astype(float)

    values = features.loc[:, config.feature_columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise DatasetError("Feature chứa giá trị vô cực")
    observed = np.isfinite(values)
    if ((values[observed] < 0) | (values[observed] > 180)).any():
        raise DatasetError("Giá trị góc/ROM phải nằm trong khoảng 0–180 độ")

    for column in config.categorical_columns:
        normalized: list[str | float] = []
        for value in features[column]:
            if pd.isna(value) or not str(value).strip():
                normalized.append(np.nan)
                continue
            category = canonical_name(value)
            if not category:
                raise DatasetError(
                    f"Cột {column!r} chứa category không hợp lệ: {value!r}"
                )
            normalized.append(category)
        features[column] = normalized

    metadata_items: list[dict[str, object]] = []
    metadata_columns = ["exercise", "rep_id", *config.categorical_columns]
    for row in angles.loc[:, metadata_columns].to_dict("records"):
        item: dict[str, object] = {
            "exercise": row["exercise"],
            "rep_id": str(row["rep_id"]),
        }
        for column in config.categorical_columns:
            value = row[column]
            item[column] = (
                canonical_name(value)
                if not pd.isna(value) and str(value).strip()
                else None
            )
        metadata_items.append(item)
    metadata = tuple(metadata_items)
    return features.reset_index(drop=True), metadata


def _parse_label(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        number = int(value)
        if number in (0, 1) and float(value) == number:
            return number
    text = str(value).strip().lower()
    mapping = {
        "1": 1,
        "true": 1,
        "correct": 1,
        "đúng": 1,
        "dung": 1,
        "0": 0,
        "false": 0,
        "incorrect": 0,
        "sai": 0,
    }
    if text not in mapping:
        raise DatasetError(f"Nhãn {value!r} không hợp lệ; dùng 1/0 hoặc true/false")
    return mapping[text]


def load_features(
    angles_path: str | Path,
    config: ExerciseConfig,
) -> PreparedFeatures:
    features, metadata = _prepare_feature_frame(angles_path, config)
    return PreparedFeatures(
        features=features,
        metadata=metadata,
        feature_names=config.model_columns,
        numeric_feature_names=config.feature_columns,
        categorical_feature_names=config.categorical_columns,
    )


def load_dataset(
    angles_path: str | Path,
    labels_path: str | Path,
    config: ExerciseConfig,
) -> PreparedDataset:
    features, metadata = _prepare_feature_frame(angles_path, config)
    source = f"labels ({labels_path})"
    labels = _normalize_keys(_read_csv(labels_path, source))
    label_column = "label" if "label" in labels.columns else "correct"
    _require_columns(labels, {"exercise", "rep_id", label_column}, source)
    _check_unique_keys(labels, source)
    labels = labels.loc[labels["exercise"] == config.name].copy()

    label_map = {
        (row.exercise, str(row.rep_id)): _parse_label(getattr(row, label_column))
        for row in labels.itertuples(index=False)
    }
    y: list[int] = []
    for item in metadata:
        key = (item["exercise"], item["rep_id"])
        if key not in label_map:
            raise DatasetError(f"Không tìm thấy label cho {key}")
        y.append(label_map[key])

    angle_keys = {(item["exercise"], item["rep_id"]) for item in metadata}
    extra_labels = set(label_map) - angle_keys
    if extra_labels:
        raise DatasetError(
            f"Có label không có dòng angles tương ứng: {list(extra_labels)[:5]}"
        )

    return PreparedDataset(
        features=features,
        labels=np.asarray(y, dtype=np.int64),
        metadata=metadata,
        feature_names=config.model_columns,
        numeric_feature_names=config.feature_columns,
        categorical_feature_names=config.categorical_columns,
    )
