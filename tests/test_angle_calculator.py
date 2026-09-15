import math
import unittest

from src.angle_calculator import AngleCalculator


def make_landmark(pixel, visibility=1.0, world=None):
    return {
        "pixel": pixel,
        "coor_2d": pixel,
        "coor_3d": world if world is not None else (pixel[0], pixel[1], 0.0),
        "visibility": visibility,
    }


class TestAngleFormula(unittest.TestCase):
    """Kiểm tra công thức tích vô hướng trên các ca có đáp án biết trước."""

    def assertAngle(self, a, b, c, expected):
        got = AngleCalculator.angle_between(a, b, c)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, expected, places=3)

    def test_goc_vuong_2d(self):
        self.assertAngle((0, 1), (0, 0), (1, 0), 90.0)

    def test_duoi_thang_2d(self):
        self.assertAngle((-1, 0), (0, 0), (1, 0), 180.0)

    def test_gap_sat_2d(self):
        self.assertAngle((1, 0), (0, 0), (1, 0), 0.0)

    def test_goc_45_2d(self):
        self.assertAngle((1, 0), (0, 0), (1, 1), 45.0)

    def test_goc_120_2d(self):
        self.assertAngle((1, 0), (0, 0), (-0.5, math.sin(math.radians(120))), 120.0)

    def test_goc_vuong_3d(self):
        self.assertAngle((0, 0, 1), (0, 0, 0), (1, 0, 0), 90.0)

    def test_goc_60_3d(self):
        c = (math.cos(math.radians(60)), math.sin(math.radians(60)), 0)
        self.assertAngle((1, 0, 0), (0, 0, 0), c, 60.0)

    def test_khong_doi_khi_doi_cho_hai_canh(self):
        """Góc A-B-C phải bằng góc C-B-A."""
        a, b, c = (3, 7), (1, 2), (9, 4)
        self.assertAlmostEqual(
            AngleCalculator.angle_between(a, b, c),
            AngleCalculator.angle_between(c, b, a),
            places=9
        )

    def test_khong_doi_khi_phong_to_canh(self):
        """Góc chỉ phụ thuộc hướng, không phụ thuộc độ dài chi."""
        base = AngleCalculator.angle_between((2, 0), (0, 0), (0, 3))
        scaled = AngleCalculator.angle_between((500, 0), (0, 0), (0, 0.001))
        self.assertAlmostEqual(base, scaled, places=9)

    def test_vector_suy_bien_tra_ve_none(self):
        """Hai điểm trùng nhau: không xác định được góc, và không được ném lỗi."""
        self.assertIsNone(AngleCalculator.angle_between((0, 0), (0, 0), (1, 0)))
        self.assertIsNone(AngleCalculator.angle_between((1, 0), (0, 0), (0, 0)))

    def test_chan_bien_arccos_khong_tra_nan(self):
        """Ba điểm thẳng hàng cách nhau rất xa dễ làm cos vượt ngoài [-1, 1]
        do sai số dấu phẩy động, khiến arccos trả về NaN."""
        for point in [(1e8, 1e8), (1e-7, 1e-7), (12345.6789, 98765.4321)]:
            with self.subTest(point=point):
                same = AngleCalculator.angle_between(point, (0, 0), point)
                opposite = AngleCalculator.angle_between(
                    point, (0, 0), (-point[0], -point[1])
                )
                self.assertFalse(math.isnan(same))
                self.assertFalse(math.isnan(opposite))
                self.assertAlmostEqual(same, 0.0, places=3)
                self.assertAlmostEqual(opposite, 180.0, places=3)


class TestCompute(unittest.TestCase):

    def setUp(self):
        self.calc = AngleCalculator(min_visibility=0.5)
        # Khuỷu trái gập vuông: vai(11) -> khuỷu(13) -> cổ tay(15)
        self.landmarks = {
            11: make_landmark((100, 100)),
            13: make_landmark((100, 200)),
            15: make_landmark((200, 200)),
        }

    def test_tinh_dung_goc_khuyu(self):
        angles = self.calc.compute(self.landmarks)
        self.assertAlmostEqual(angles["LEFT_ELBOW"]["angle_2d"], 90.0, places=3)

    def test_luon_tra_du_moi_khop_du_thieu_du_lieu(self):
        """Khớp thiếu dữ liệu phải là None nhưng KEY vẫn tồn tại, để chuỗi
        thời gian đầu ra luôn cùng độ dài."""
        angles = self.calc.compute(self.landmarks)
        self.assertEqual(set(angles.keys()), set(self.calc.joint_names))
        self.assertIsNone(angles["RIGHT_KNEE"])

    def test_landmarks_rong_van_tra_du_key(self):
        angles = self.calc.compute({})
        self.assertEqual(set(angles.keys()), set(self.calc.joint_names))
        self.assertTrue(all(v is None for v in angles.values()))

    def test_loc_khop_bi_che_khuat(self):
        self.landmarks[15]["visibility"] = 0.2
        angles = self.calc.compute(self.landmarks)
        self.assertIsNone(angles["LEFT_ELBOW"])

    def test_nguong_visibility_dung_bien(self):
        """Đúng bằng ngưỡng thì vẫn tính, dưới ngưỡng mới loại."""
        self.landmarks[15]["visibility"] = 0.5
        self.assertIsNotNone(self.calc.compute(self.landmarks)["LEFT_ELBOW"])

        self.landmarks[15]["visibility"] = 0.49
        self.assertIsNone(self.calc.compute(self.landmarks)["LEFT_ELBOW"])

    def test_dung_toa_do_pixel_chu_khong_phai_chuan_hoa(self):
        """Khung hình không vuông làm hai trục co giãn khác hệ số, nên góc tính
        trên tọa độ chuẩn hóa sẽ lệch khỏi góc thật. Phải lấy theo pixel."""
        w, h = 1280, 720
        # Phải chọn điểm XIÊN. Nếu hai cạnh nằm đúng trục ngang/dọc thì góc
        # vẫn là 90 độ ở cả hai hệ, phép thử sẽ không phát hiện được gì.
        norm = [(0.3, 0.2), (0.5, 0.5), (0.9, 0.4)]
        pixels = [(x * w, y * h) for x, y in norm]

        landmarks = {}
        for index, (n, p) in zip((11, 13, 15), zip(norm, pixels)):
            landmarks[index] = {
                "pixel": p,
                "coor_2d": n,
                "coor_3d": (n[0], n[1], 0.0),
                "visibility": 1.0,
            }

        got = self.calc.compute(landmarks)["LEFT_ELBOW"]["angle_2d"]
        expect_pixel = AngleCalculator.angle_between(*pixels)
        expect_norm = AngleCalculator.angle_between(*norm)

        # Hai cách phải lệch nhau rõ rệt thì phép thử này mới có ý nghĩa.
        self.assertGreater(abs(expect_pixel - expect_norm), 1.0)
        self.assertAlmostEqual(got, expect_pixel, places=3)

    def test_goc_3d_lay_tu_world_landmarks(self):
        world = {
            11: {"coor_3d": (0.0, 0.0, 0.0), "visibility": 1.0},
            13: {"coor_3d": (0.0, 0.3, 0.0), "visibility": 1.0},
            15: {"coor_3d": (0.0, 0.3, 0.3), "visibility": 1.0},
        }
        angles = self.calc.compute(self.landmarks, world)
        self.assertAlmostEqual(angles["LEFT_ELBOW"]["angle_3d"], 90.0, places=3)

    def test_thieu_world_landmarks_thi_goc_3d_la_none(self):
        angles = self.calc.compute(self.landmarks)
        self.assertIsNone(angles["LEFT_ELBOW"]["angle_3d"])
        self.assertIsNotNone(angles["LEFT_ELBOW"]["angle_2d"])

    def test_chon_tap_con_khop(self):
        calc = AngleCalculator(joints=["LEFT_ELBOW", "RIGHT_KNEE"])
        self.assertEqual(calc.joint_names, ["LEFT_ELBOW", "RIGHT_KNEE"])
        self.assertEqual(set(calc.compute(self.landmarks).keys()),
                         {"LEFT_ELBOW", "RIGHT_KNEE"})


class TestArcGeometry(unittest.TestCase):
    """Cung tròn phải trùng với giá trị arccos trả về, nếu không thì hình vẽ
    nói một đằng còn con số nói một nẻo."""

    def test_do_mo_cung_bang_gia_tri_goc(self):
        cases = [
            ((100, 100), (100, 200), (200, 200)),
            ((0, 0), (50, 80), (120, 10)),
            ((300, 50), (100, 100), (90, 400)),
            ((10, 10), (10, 100), (10, 300)),
        ]
        for a, b, c in cases:
            with self.subTest(points=(a, b, c)):
                _, sweep, _ = AngleCalculator._arc_geometry(a, b, c)
                expected = AngleCalculator.angle_between(a, b, c)
                self.assertAlmostEqual(sweep, expected, places=3)

    def test_luon_ve_cung_nho(self):
        _, sweep, _ = AngleCalculator._arc_geometry((200, 200), (100, 200), (100, 100))
        self.assertLessEqual(sweep, 180.0)

    def test_ban_kinh_nam_trong_gioi_han(self):
        # Chi rất ngắn và rất dài đều phải cho bán kính đọc được trên màn hình.
        for c in [(101, 200), (5000, 200)]:
            with self.subTest(c=c):
                _, _, radius = AngleCalculator._arc_geometry((100, 100), (100, 200), c)
                self.assertGreaterEqual(radius, 18)
                self.assertLessEqual(radius, 55)


if __name__ == "__main__":
    unittest.main()
