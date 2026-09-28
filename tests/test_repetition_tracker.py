import unittest

from src.repetition_tracker import RepetitionTracker

JOINTS = (
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
)


def angles(
    right_shoulder,
    *,
    left_shoulder=15.0,
    left_elbow=170.0,
    right_elbow=168.0,
):
    values = {
        "LEFT_ELBOW": left_elbow,
        "RIGHT_ELBOW": right_elbow,
        "LEFT_SHOULDER": left_shoulder,
        "RIGHT_SHOULDER": right_shoulder,
    }
    return {
        joint: ({"angle_2d_smooth": value} if value is not None else None)
        for joint, value in values.items()
    }


class TestRepetitionTracker(unittest.TestCase):
    def make_tracker(self):
        return RepetitionTracker(
            JOINTS,
            baseline_window=5,
            feature_window=1,
            start_threshold=5,
            start_hold_frames=2,
            min_rom=20,
            reversal_threshold=5,
            reversal_hold_frames=2,
            end_tolerance=3,
            end_hold_frames=2,
        )

    def test_tao_start_turning_rom_khi_hoan_thanh_mot_rep(self):
        tracker = self.make_tracker()
        sequence = [10] * 5 + [20, 30, 50, 80, 100, 90, 70, 40, 20, 11, 10]

        completed = None
        for frame_id, value in enumerate(sequence):
            result = tracker.update(frame_id, angles(value))
            if result is not None:
                completed = result

        self.assertIsNotNone(completed)
        self.assertEqual(completed.driver_joint, "RIGHT_SHOULDER")
        self.assertEqual(completed.start_frame, 5)
        self.assertEqual(completed.turning_frame, 9)
        self.assertEqual(completed.end_frame, 15)
        self.assertEqual(completed.feature_values["right_shoulder_start"], 10.0)
        self.assertEqual(completed.feature_values["right_shoulder_turning"], 100.0)
        self.assertEqual(completed.feature_values["right_shoulder_rom"], 90.0)
        self.assertEqual(completed.feature_values["left_shoulder_rom"], 0.0)
        self.assertEqual(len(completed.feature_values), 12)

    def test_chua_du_doan_khi_chua_quay_ve_diem_bat_dau(self):
        tracker = self.make_tracker()
        sequence = [10] * 5 + [20, 30, 50, 80, 100, 90, 70]

        results = [
            tracker.update(frame_id, angles(value))
            for frame_id, value in enumerate(sequence)
        ]

        self.assertTrue(all(result is None for result in results))
        self.assertEqual(tracker.state, "returning")

    def test_khop_khong_nhin_thay_duoc_giu_la_none(self):
        tracker = self.make_tracker()
        sequence = [10] * 5 + [20, 30, 50, 80, 100, 90, 70, 40, 20, 11, 10]

        completed = None
        for frame_id, value in enumerate(sequence):
            result = tracker.update(
                frame_id,
                angles(value, left_shoulder=None, left_elbow=None),
            )
            if result is not None:
                completed = result

        self.assertIsNotNone(completed)
        self.assertIsNone(completed.feature_values["left_shoulder_start"])
        self.assertIsNone(completed.feature_values["left_shoulder_turning"])
        self.assertIsNone(completed.feature_values["left_shoulder_rom"])
        self.assertIsNone(completed.feature_values["left_elbow_rom"])
        self.assertEqual(completed.observed_feature_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
