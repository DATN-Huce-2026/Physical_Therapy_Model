from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_exercise_config
from .dataset import PreparedDataset, load_dataset
from .model import (
    evaluate_classifier,
    evaluate_missingness_baseline,
    fit_classifier,
    fit_missingness_baseline,
    save_classifier,
    select_classifier,
)
from .reference_profile import fit_reference_profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "exercises.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Random Forest với một dòng dữ liệu cho mỗi repetition"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Thư mục chứa angles/labels train, val và test",
    )
    parser.add_argument("--exercise", required=True, help="Tên bài tập")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument(
        "--angles-suffix",
        default="",
        help="Hậu tố file angles, ví dụ _updated -> angles_train_updated.csv",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _load_split(
    data_dir: Path,
    split: str,
    config,
    angles_suffix: str = "",
) -> PreparedDataset:
    return load_dataset(
        data_dir / f"angles_{split}{angles_suffix}.csv",
        data_dir / f"labels_{split}.csv",
        config,
    )


def _split_summary(dataset: PreparedDataset) -> dict[str, object]:
    summary: dict[str, object] = {
        "repetitions": len(dataset.labels),
        "incorrect_label_0": int((dataset.labels == 0).sum()),
        "correct_label_1": int((dataset.labels == 1).sum()),
        "missing_cell_ratio": round(dataset.missing_cell_ratio, 6),
        "rows_with_missing_ratio": round(dataset.rows_with_missing_ratio, 6),
    }
    if dataset.categorical_feature_names:
        summary["categorical_distribution"] = {}
        for column in dataset.categorical_feature_names:
            distribution: dict[str, object] = {}
            values = dataset.features[column].fillna("unknown")
            for category in sorted(values.unique()):
                mask = values == category
                labels = dataset.labels[mask.to_numpy()]
                distribution[str(category)] = {
                    "repetitions": int(mask.sum()),
                    "incorrect_label_0": int((labels == 0).sum()),
                    "correct_label_1": int((labels == 1).sum()),
                }
            summary["categorical_distribution"][column] = distribution
    return summary


def _metrics_by_category(
    model,
    dataset: PreparedDataset,
    column: str,
    threshold: float,
) -> dict[str, object]:
    result: dict[str, object] = {}
    values = dataset.features[column].fillna("unknown")
    for category in sorted(values.unique()):
        mask = values == category
        result[str(category)] = evaluate_classifier(
            model,
            dataset.features.loc[mask].reset_index(drop=True),
            dataset.labels[mask.to_numpy()],
            threshold,
        )
    return result


def train(args: argparse.Namespace) -> dict[str, object]:
    config = load_exercise_config(args.config, args.exercise)
    data_dir = Path(args.data_dir)
    train_set = _load_split(data_dir, "train", config, args.angles_suffix)
    val_set = _load_split(data_dir, "val", config, args.angles_suffix)
    test_set = _load_split(data_dir, "test", config, args.angles_suffix)

    best_parameters, search_results = select_classifier(
        train_set.features,
        train_set.labels,
        val_set.features,
        val_set.labels,
        numeric_feature_names=config.feature_columns,
        categorical_feature_names=config.categorical_columns,
        n_estimators=args.n_estimators,
        threshold=config.decision_threshold,
        random_state=args.random_state,
    )

    selection_model = fit_classifier(
        train_set.features,
        train_set.labels,
        best_parameters,
        args.random_state,
        numeric_feature_names=config.feature_columns,
        categorical_feature_names=config.categorical_columns,
    )
    validation_metrics = evaluate_classifier(
        selection_model,
        val_set.features,
        val_set.labels,
        config.decision_threshold,
    )

    combined_features = pd.concat(
        [train_set.features, val_set.features], ignore_index=True
    )
    combined_labels = np.concatenate([train_set.labels, val_set.labels])
    final_model = fit_classifier(
        combined_features,
        combined_labels,
        best_parameters,
        args.random_state,
        numeric_feature_names=config.feature_columns,
        categorical_feature_names=config.categorical_columns,
    )
    test_metrics = evaluate_classifier(
        final_model,
        test_set.features,
        test_set.labels,
        config.decision_threshold,
    )

    profile = fit_reference_profile(
        combined_features.loc[:, config.feature_columns].to_numpy(dtype=float),
        combined_labels,
        exercise=config.name,
        feature_names=config.feature_columns,
        lower_percentile=config.lower_percentile,
        upper_percentile=config.upper_percentile,
        margin_degrees=config.profile_margin_deg,
    )

    missing_model = fit_missingness_baseline(
        train_set.numeric_features, train_set.labels, args.random_state
    )
    missing_validation = evaluate_missingness_baseline(
        missing_model, val_set.numeric_features, val_set.labels
    )
    final_missing_model = fit_missingness_baseline(
        combined_features.loc[:, config.feature_columns],
        combined_labels,
        args.random_state,
    )
    missing_test = evaluate_missingness_baseline(
        final_missing_model, test_set.numeric_features, test_set.labels
    )

    categorical_values = {
        column: sorted(combined_features[column].dropna().astype(str).unique().tolist())
        for column in config.categorical_columns
    }

    artifact_dir = Path(args.output_dir) / config.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = save_classifier(
        artifact_dir / "classifier.joblib",
        final_model,
        {
            "exercise": config.name,
            "feature_names": config.model_columns,
            "numeric_feature_names": config.feature_columns,
            "categorical_feature_names": config.categorical_columns,
            "categorical_values": categorical_values,
            "decision_threshold": config.decision_threshold,
            "label_meaning": {0: "incorrect", 1: "correct"},
            "parameters": best_parameters,
        },
    )
    profile_path = profile.save(artifact_dir / "reference_profile.json")

    report: dict[str, object] = {
        "exercise": config.name,
        "label_meaning": {"0": "incorrect/false", "1": "correct/true"},
        "feature_count": len(config.model_columns),
        "features": list(config.model_columns),
        "numeric_features": list(config.feature_columns),
        "categorical_features": list(config.categorical_columns),
        "categorical_values": categorical_values,
        "splits": {
            "train": _split_summary(train_set),
            "validation": _split_summary(val_set),
            "test": _split_summary(test_set),
        },
        "selected_parameters": best_parameters,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_category": {
            column: _metrics_by_category(
                final_model,
                test_set,
                column,
                config.decision_threshold,
            )
            for column in config.categorical_columns
        },
        "missingness_only_baseline": {
            "validation": missing_validation,
            "test": missing_test,
        },
        "hyperparameter_search": search_results,
        "classifier": str(classifier_path),
        "reference_profile": str(profile_path),
    }
    if missing_test["macro_f1"] >= test_metrics["macro_f1"] - 0.05:
        report["data_warning"] = (
            "Missingness-only baseline gần bằng hoặc tốt hơn model góc; "
            "cần kiểm tra tương quan giữa góc quay, dữ liệu thiếu và label."
        )

    report_path = artifact_dir / "training_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    report["report"] = str(report_path)
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = train(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
