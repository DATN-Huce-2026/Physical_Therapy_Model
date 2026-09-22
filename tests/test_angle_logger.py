import csv
import json
import os
import tempfile
import unittest

from src.angle_logger import AngleLogger, ExerciseAngleLogger

JOINTS = ["LEFT_ELBOW", "RIGHT_KNEE"]


def angle_entry(v2d=None, v3d=None, s2d=None, s3d=None):
    return {
        "angle_2d": v2d,
        "angle_3d": v3d,
        "angle_2d_smooth": s2d,
        "angle_3d_smooth": s3d,
    }


class TestColumns(unittest.TestCase):

    def test_du_cot_cho_moi_khop(self):
        logger = AngleLogger(JOINTS)
        self.assertEqual(
            logger.columns,
            ["frame_id", "timestamp_sec", "has_pose",
             "LEFT_ELBOW_2d", "LEFT_ELBOW_2d_smooth",
             "LEFT_ELBOW_3d", "LEFT_ELBOW_3d_smooth",
             "RIGHT_KNEE_2d", "RIGHT_KNEE_2d_smooth",
             "RIGHT_KNEE_3d", "RIGHT_KNEE_3d_smooth"]
        )

    def test_tat_3d_va_smooth(self):
        logger = AngleLogger(JOINTS, include_3d=False, include_smooth=False)
        self.assertEqual(
            logger.columns,
            ["frame_id", "timestamp_sec", "has_pose", "LEFT_ELBOW_2d", "RIGHT_KNEE_2d"]
        )


class TestTrucThoiGian(unittest.TestCase):
    """Đây là điểm dễ hỏng nhất: log phải khớp 1-1 với frame của video gốc."""

    def test_ghi_ca_frame_khong_co_nguoi(self):
        logger = AngleLogger(JOINTS)

        logger.add(0, 0.0, {"LEFT_ELBOW": angle_entry(90.0)}, has_pose=True)
        logger.add(1, 0.033, {}, has_pose=False)
        logger.add(2, 0.066, {"LEFT_ELBOW": angle_entry(95.0)}, has_pose=True)

        self.assertEqual(len(logger.records), 3)
        self.assertEqual([r["frame_id"] for r in logger.records], [0, 1, 2])
        self.assertFalse(logger.records[1]["has_pose"])
        self.assertIsNone(logger.records[1]["LEFT_ELBOW_2d"])

    def test_chuoi_giu_nguyen_do_dai_ke_ca_cho_trong(self):
        logger = AngleLogger(JOINTS)
        for frame in range(5):
            has_pose = frame != 2
            angles = {"LEFT_ELBOW": angle_entry(90.0)} if has_pose else {}
            logger.add(frame, frame / 30.0, angles, has_pose=has_pose)

        series = logger.series("LEFT_ELBOW", "2d")
        self.assertEqual(len(series), 5)
        self.assertIsNone(series[2])

    def test_khop_thieu_trong_dict_van_co_cot(self):
        logger = AngleLogger(JOINTS)
        logger.add(0, 0.0, {"LEFT_ELBOW": angle_entry(90.0)}, has_pose=True)
        self.assertIn("RIGHT_KNEE_2d", logger.records[0])
        self.assertIsNone(logger.records[0]["RIGHT_KNEE_2d"])


class TestSummary(unittest.TestCase):

    def test_thong_ke_tren_chuoi_da_loc_theo_mac_dinh(self):
        """Tầm vận động tính từ chuỗi thô sẽ bị gai nhiễu thổi phồng."""
        logger = AngleLogger(["LEFT_ELBOW"])
        for frame, (raw, smooth) in enumerate([(100, 100), (240, 101), (100, 100)]):
            logger.add(frame, frame / 30.0,
                       {"LEFT_ELBOW": angle_entry(raw, None, smooth, None)},
                       has_pose=True)

        self.assertEqual(logger.summary()["LEFT_ELBOW"]["rom"], 1.0)
        self.assertEqual(logger.summary("2d")["LEFT_ELBOW"]["rom"], 140.0)

    def test_khong_co_du_lieu_thi_bao_count_0(self):
        logger = AngleLogger(JOINTS)
        logger.add(0, 0.0, {}, has_pose=False)

        stat = logger.summary()["LEFT_ELBOW"]
        self.assertEqual(stat["count"], 0)
        self.assertIsNone(stat["rom"])

    def test_gia_tri_thong_ke(self):
        logger = AngleLogger(["LEFT_ELBOW"])
        for frame, v in enumerate([90.0, 100.0, 110.0]):
            logger.add(frame, 0.0, {"LEFT_ELBOW": angle_entry(v, None, v, None)},
                       has_pose=True)

        stat = logger.summary()["LEFT_ELBOW"]
        self.assertEqual(stat["count"], 3)
        self.assertEqual(stat["min"], 90.0)
        self.assertEqual(stat["max"], 110.0)
        self.assertEqual(stat["mean"], 100.0)
        self.assertEqual(stat["rom"], 20.0)


class TestXuatFile(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.logger = AngleLogger(JOINTS)
        self.logger.add(0, 0.0, {"LEFT_ELBOW": angle_entry(90.0, 80.0, 90.0, 80.0)},
                        has_pose=True)
        self.logger.add(1, 0.033, {}, has_pose=False)

    def test_csv_doc_lai_dung_so_dong_va_so_cot(self):
        path = self.logger.to_csv(os.path.join(self.tmp, "a", "angles.csv"))
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0].keys()), self.logger.columns)
        self.assertEqual(rows[0]["LEFT_ELBOW_2d"], "90.0")
        self.assertEqual(rows[1]["LEFT_ELBOW_2d"], "")

    def test_json_co_summary_va_metadata(self):
        path = self.logger.to_json(os.path.join(self.tmp, "b", "angles.json"))
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        self.assertEqual(payload["total_frames"], 2)
        self.assertEqual(payload["joints"], JOINTS)
        self.assertIn("summary", payload)
        self.assertEqual(len(payload["frames"]), 2)

    def test_tu_tao_thu_muc_chua_ton_tai(self):
        path = os.path.join(self.tmp, "chua", "co", "angles.csv")
        self.assertTrue(os.path.exists(self.logger.to_csv(path)))

    def test_lam_tron_2_chu_so(self):
        logger = AngleLogger(["LEFT_ELBOW"])
        logger.add(0, 0.123456,
                   {"LEFT_ELBOW": angle_entry(90.123456, None, 90.987654, None)},
                   has_pose=True)

        record = logger.records[0]
        self.assertEqual(record["LEFT_ELBOW_2d"], 90.12)
        self.assertEqual(record["LEFT_ELBOW_2d_smooth"], 90.99)
        self.assertEqual(record["timestamp_sec"], 0.1235)


class TestExerciseAngleLogger(unittest.TestCase):

    def test_schema_batch_chi_co_bai_tap_rep_frame_va_goc(self):
        logger = ExerciseAngleLogger()
        logger.add("squat", "rep_01", 0,
                   {"LEFT_KNEE": angle_entry(90.0, s2d=91.25)},
                   ["LEFT_KNEE"], "angle_2d_smooth")

        self.assertEqual(logger.columns,
                         ["exercise", "rep_id", "frame", "LEFT_KNEE"])
        self.assertEqual(logger.records[0],
                         {"exercise": "squat", "rep_id": "rep_01",
                          "frame": 0, "LEFT_KNEE": 91.25})

    def test_chi_tao_cot_khi_goc_co_gia_tri(self):
        logger = ExerciseAngleLogger()
        logger.add("elbow", "rep_01", 0, {}, ["LEFT_ELBOW"], "angle_2d")
        logger.add("knee", "rep_02", 0,
                   {"RIGHT_KNEE": angle_entry(100.0)}, ["RIGHT_KNEE"], "angle_2d")

        self.assertEqual(logger.columns,
                         ["exercise", "rep_id", "frame", "RIGHT_KNEE"])
        self.assertNotIn("LEFT_ELBOW", logger.records[0])
        self.assertNotIn("LEFT_ELBOW", logger.records[1])


if __name__ == "__main__":
    unittest.main()
