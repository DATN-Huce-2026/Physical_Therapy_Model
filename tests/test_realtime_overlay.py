import unittest

import numpy as np

from realtime_predict import _draw_overlay
from src.repetition_tracker import RepetitionTracker


class TestRealtimeOverlay(unittest.TestCase):
    def test_panel_tu_co_va_khong_che_den_camera(self):
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        tracker = RepetitionTracker(
            (
                "LEFT_ELBOW",
                "RIGHT_ELBOW",
                "LEFT_SHOULDER",
                "RIGHT_SHOULDER",
            )
        )
        prediction = {
            "correct": False,
            "correct_probability": 0.277,
            "reference_comparison": {
                "features": {
                    "left_elbow_turning": {
                        "status": "observed",
                        "direction": "lower",
                        "signed_deviation_deg": -7.1,
                    }
                }
            },
        }

        x, y, width, height = _draw_overlay(
            frame,
            tracker,
            repetition_count=2,
            prediction=prediction,
            prediction_error=None,
            visible_joint_count=2,
            total_joint_count=4,
        )

        self.assertLess(width, frame.shape[1] * 0.8)
        self.assertLess(height, frame.shape[0] * 0.4)
        self.assertTrue(np.all(frame[y + 2, x + 2] > 0))
        self.assertTrue(np.all(frame[y + 2, x + 2] < 200))
        self.assertTrue(np.all(frame[-1, -1] == 200))


if __name__ == "__main__":
    unittest.main()
