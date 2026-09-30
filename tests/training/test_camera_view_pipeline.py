import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from training.config import load_exercise_config
from training.dataset import load_dataset
from training.live_inference import LiveRepetitionClassifier
from training.model import load_classifier
from training.predict import predict
from training.train import train


class TestCameraViewPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.config_path = self.root / "exercises.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "abduction": {
                        "feature_columns": ["SHOULDER_start"],
                        "categorical_columns": ["camera_view"],
                        "decision_threshold": 0.5,
                    }
                }
            ),
            encoding="utf-8",
        )

        for split, size in (("train", 60), ("val", 30), ("test", 30)):
            angle_rows = []
            label_rows = []
            views = ("Front", "Left", "Right")
            for index in range(size):
                label = index % 2
                value = 80.0 + index % 3 if label else 20.0 + index % 3
                rep_id = f"{split}_{index}"
                angle_rows.append(
                    ["Abduction", rep_id, views[index % len(views)], value]
                )
                label_rows.append(["Abduction", rep_id, label])

            pd.DataFrame(
                angle_rows,
                columns=["exercise", "rep_id", "camera_view", "SHOULDER_start"],
            ).to_csv(
                self.data_dir / f"angles_{split}_updated.csv",
                index=False,
            )
            pd.DataFrame(
                label_rows,
                columns=["exercise", "rep_id", "label"],
            ).to_csv(self.data_dir / f"labels_{split}.csv", index=False)

        self.artifact_root = self.root / "artifacts"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_loader_chuan_hoa_camera_view_va_giu_feature_numeric(self):
        config = load_exercise_config(self.config_path, "Abduction")
        dataset = load_dataset(
            self.data_dir / "angles_train_updated.csv",
            self.data_dir / "labels_train.csv",
            config,
        )

        self.assertEqual(config.model_columns, ("shoulder_start", "camera_view"))
        self.assertEqual(
            set(dataset.features["camera_view"]),
            {"front", "left", "right"},
        )
        self.assertEqual(dataset.feature_names, config.model_columns)
        self.assertEqual(dataset.numeric_feature_names, ("shoulder_start",))
        self.assertEqual(dataset.missing_cell_ratio, 0.0)

    def test_train_predict_va_live_inference_co_camera_view(self):
        report = train(
            Namespace(
                data_dir=str(self.data_dir),
                exercise="Abduction",
                config=str(self.config_path),
                output_dir=str(self.artifact_root),
                angles_suffix="_updated",
                n_estimators=20,
                random_state=7,
            )
        )

        self.assertEqual(report["categorical_features"], ["camera_view"])
        self.assertEqual(
            report["categorical_values"]["camera_view"],
            ["front", "left", "right"],
        )
        self.assertIn("camera_view", report["test_metrics_by_category"])

        artifact = load_classifier(
            self.artifact_root / "abduction" / "classifier.joblib"
        )
        transformed_names = (
            artifact["model"].named_steps["preprocessor"].get_feature_names_out()
        )
        self.assertIn("camera_view_front", transformed_names)
        self.assertIn("camera_view_left", transformed_names)
        self.assertIn("camera_view_right", transformed_names)

        batch_results = predict(
            Namespace(
                angles=str(self.data_dir / "angles_test_updated.csv"),
                exercise="Abduction",
                config=str(self.config_path),
                artifact_dir=str(self.artifact_root),
                output=None,
            )
        )
        self.assertEqual(batch_results[0]["camera_view"], "front")

        live = LiveRepetitionClassifier(
            exercise="Abduction",
            config_path=self.config_path,
            artifact_dir=self.artifact_root,
        )
        live_result = live.predict({"shoulder_start": 81.0, "camera_view": "Front"})
        self.assertEqual(
            live_result["categorical_features"],
            {"camera_view": "front"},
        )
        with self.assertRaises(ValueError):
            live.predict({"shoulder_start": 81.0, "camera_view": "top"})


if __name__ == "__main__":
    unittest.main()
