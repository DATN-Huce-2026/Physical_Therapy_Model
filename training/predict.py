from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_exercise_config
from .dataset import load_features
from .model import correct_probabilities, load_classifier
from .reference_profile import ReferenceProfile
from .train import DEFAULT_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dự đoán từ CSV một dòng cho mỗi repetition"
    )
    parser.add_argument("--angles", required=True)
    parser.add_argument("--exercise", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--output", help="File JSON kết quả")
    return parser.parse_args()


def predict(args: argparse.Namespace) -> list[dict[str, object]]:
    config = load_exercise_config(args.config, args.exercise)
    prepared = load_features(args.angles, config)
    artifact_dir = Path(args.artifact_dir) / config.name
    artifact = load_classifier(artifact_dir / "classifier.joblib")
    profile = ReferenceProfile.load(artifact_dir / "reference_profile.json")

    artifact_features = tuple(artifact["feature_names"])
    if artifact_features != prepared.feature_names:
        raise ValueError(
            f"Feature model {artifact_features} khác dữ liệu {prepared.feature_names}"
        )

    probabilities = correct_probabilities(artifact["model"], prepared.features)
    threshold = float(artifact["decision_threshold"])
    raw_values = prepared.features.to_numpy(dtype=float)
    results: list[dict[str, object]] = []
    for index, metadata in enumerate(prepared.metadata):
        probability = float(probabilities[index])
        results.append(
            {
                **metadata,
                "label": int(probability >= threshold),
                "correct": bool(probability >= threshold),
                "correct_probability": round(probability, 6),
                "decision_threshold": threshold,
                "reference_comparison": profile.compare(raw_values[index]),
            }
        )
    return results


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    results = predict(args)
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
