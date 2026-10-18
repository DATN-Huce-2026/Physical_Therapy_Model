import unittest

import numpy as np

from keraal_mtl.assessment import assess


def probabilities(**overrides):
    base = {
        "exercise": [0.9, 0.05, 0.05],
        "exercise_error_class": [0.7, 0.1, 0.1, 0.1],
        "global_evaluation": [0.7, 0.2, 0.05, 0.05],
        "error_type": [0.6, 0.1, 0.1, 0.1, 0.1],
        "severity": [0.8, 0.2],
        "body_part": [0.1, 0.7, 0.1, 0.1],
    }
    base.update(overrides)
    return {task: np.asarray(values) for task, values in base.items()}


class TestAssess(unittest.TestCase):
    def test_cac_head_nghieng_ve_dung_thi_ket_luan_dung(self):
        result = assess(probabilities(), expected_exercise="CTK")

        # (0.7/0.9 + 0.7/0.8 + 0.6/0.7) / 3
        self.assertAlmostEqual(result["score"], 83.7, places=1)
        self.assertEqual(result["verdict"], "ĐÚNG")
        self.assertIsNone(result["error"])
        self.assertEqual(result["warnings"], [])

    def test_sai_thi_chi_ra_loai_loi_bo_phan_va_muc_do(self):
        result = assess(
            probabilities(
                exercise_error_class=[0.1, 0.1, 0.1, 0.7],
                global_evaluation=[0.1, 0.2, 0.6, 0.1],
                error_type=[0.1, 0.1, 0.1, 0.6, 0.1],
                severity=[0.3, 0.7],
                body_part=[0.1, 0.1, 0.1, 0.7],
            )
        )
        self.assertEqual(result["verdict"], "SAI")
        error = result["error"]
        self.assertEqual(error["error_type"], "Error3")
        self.assertEqual(error["reason"], "Incomplete")
        self.assertEqual(error["body_part"], "LeftArm")
        self.assertEqual(error["severity"], "LargeError")
        self.assertIn("tay trái", error["message"])

    def test_canh_bao_khi_tap_nham_bai(self):
        result = assess(probabilities(exercise=[0.1, 0.8, 0.1]), expected_exercise="CTK")
        self.assertEqual(result["exercise"], "ELK")
        self.assertTrue(any("CTK" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
