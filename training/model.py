from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


class TrainingError(ValueError):
    pass


def build_classifier(
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: str | float,
    random_state: int,
) -> Pipeline:
    """Median imputation + missing mask + Random Forest."""
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def candidate_parameters(n_estimators: int) -> list[dict[str, object]]:
    return [
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        for max_depth in (6, 10, None)
        for min_samples_leaf in (2, 5, 10)
        for max_features in ("sqrt", 0.5)
    ]


def correct_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise TrainingError("Classifier không có lớp correct=1")
    return probabilities[:, classes.index(1)]


def evaluate_classifier(
    model: Pipeline,
    features: pd.DataFrame,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    probabilities = correct_probabilities(model, features)
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    report: dict[str, object] = {
        "samples": len(labels),
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(labels, predictions)), 6
        ),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 6),
        "incorrect": {
            "label": 0,
            "precision": round(float(precision[0]), 6),
            "recall": round(float(recall[0]), 6),
            "f1": round(float(f1[0]), 6),
            "support": int(support[0]),
        },
        "correct": {
            "label": 1,
            "precision": round(float(precision[1]), 6),
            "recall": round(float(recall[1]), 6),
            "f1": round(float(f1[1]), 6),
            "support": int(support[1]),
        },
        "confusion_matrix_labels_0_1": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }
    if len(np.unique(labels)) == 2:
        report["roc_auc_correct"] = round(
            float(roc_auc_score(labels, probabilities)), 6
        )
        report["pr_auc_correct"] = round(
            float(average_precision_score(labels, probabilities)), 6
        )
        report["pr_auc_incorrect"] = round(
            float(average_precision_score(1 - labels, 1 - probabilities)), 6
        )
    return report


def select_classifier(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    val_features: pd.DataFrame,
    val_labels: np.ndarray,
    *,
    n_estimators: int,
    threshold: float,
    random_state: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if len(np.unique(train_labels)) < 2:
        raise TrainingError("Tập train phải có cả label 0 và label 1")

    results: list[dict[str, object]] = []
    for params in candidate_parameters(n_estimators):
        model = build_classifier(**params, random_state=random_state)
        model.fit(train_features, train_labels)
        metrics = evaluate_classifier(model, val_features, val_labels, threshold)
        results.append({"parameters": params, "validation_metrics": metrics})

    results.sort(
        key=lambda item: (
            item["validation_metrics"]["macro_f1"],
            item["validation_metrics"]["balanced_accuracy"],
        ),
        reverse=True,
    )
    return dict(results[0]["parameters"]), results


def fit_classifier(
    features: pd.DataFrame,
    labels: np.ndarray,
    parameters: dict[str, object],
    random_state: int,
) -> Pipeline:
    if len(np.unique(labels)) < 2:
        raise TrainingError("Cần cả label 0 và label 1 để train Random Forest")
    model = build_classifier(**parameters, random_state=random_state)
    model.fit(features, labels)
    return model


def fit_missingness_baseline(
    features: pd.DataFrame,
    labels: np.ndarray,
    random_state: int,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features.isna().astype(np.uint8), labels)
    return model


def evaluate_missingness_baseline(
    model: RandomForestClassifier,
    features: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, object]:
    predictions = model.predict(features.isna().astype(np.uint8))
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(labels, predictions)), 6
        ),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 6),
        "confusion_matrix_labels_0_1": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def save_classifier(
    path: str | Path, model: Pipeline, metadata: dict[str, object]
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, **metadata}, output_path)
    return output_path


def load_classifier(path: str | Path) -> dict[str, object]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise TrainingError("File classifier không đúng định dạng")
    return artifact
