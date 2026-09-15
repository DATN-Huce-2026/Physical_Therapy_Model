import unittest

from src.angle_smoother import AngleSmoother


class TestSmootherBasics(unittest.TestCase):

    def test_tham_so_khong_hop_le_bi_tu_choi(self):
        with self.assertRaises(ValueError):
            AngleSmoother(window=0)
        for alpha in (0, -0.1, 1.5):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    AngleSmoother(alpha=alpha)

    def test_gia_tri_none_khong_sinh_dau_ra(self):
        s = AngleSmoother()
        self.assertIsNone(s.update("k", None, 0))

    def test_frame_dau_tien_tra_dung_gia_tri_goc(self):
        """Không được chờ đầy cửa sổ mới có đầu ra, vì webcam cần hiện số ngay."""
        s = AngleSmoother(window=5)
        self.assertAlmostEqual(s.update("k", 90.0, 0), 90.0)

    def test_hoi_tu_ve_gia_tri_khong_doi(self):
        s = AngleSmoother(window=5, alpha=0.4)
        for frame in range(40):
            got = s.update("k", 120.0, frame)
        self.assertAlmostEqual(got, 120.0, places=6)

    def test_dau_ra_nam_trong_khoang_dau_vao(self):
        """Bộ lọc không được tạo ra giá trị vượt ngoài dải dữ liệu thật."""
        s = AngleSmoother()
        values = [100, 110, 95, 105, 98, 102, 107, 99]
        outputs = [s.update("k", v, i) for i, v in enumerate(values)]
        self.assertGreaterEqual(min(outputs), min(values))
        self.assertLessEqual(max(outputs), max(values))


class TestKhuGaiNhieu(unittest.TestCase):
    """Đây là lý do bộ lọc tồn tại: MediaPipe nhảy landmark một frame làm góc
    vọt hơn trăm độ rồi về chỗ cũ ngay."""

    def test_gai_don_le_bi_loai(self):
        s = AngleSmoother(window=5, alpha=0.4)
        chuoi = [100, 101, 100, 99, 242, 100, 101, 100]

        outputs = []
        for frame, value in enumerate(chuoi):
            outputs.append(s.update("k", value, frame))

        # Frame số 4 là gai 242 độ; đầu ra tại đó phải gần như không nhúc nhích.
        self.assertLess(abs(outputs[4] - 100), 5.0)
        self.assertLess(max(outputs), 110)

    def test_trung_binh_cong_khong_lam_duoc_viec_nay(self):
        """Đối chứng: nếu dùng trung bình trượt thay vì trung vị, gai sẽ lọt."""
        chuoi = [100, 101, 100, 99, 242]
        trung_binh = sum(chuoi) / len(chuoi)
        self.assertGreater(trung_binh, 125)  # bị gai kéo lệch hơn 25 độ

        s = AngleSmoother(window=5, alpha=1.0)  # alpha=1 -> chỉ còn tầng median
        for frame, value in enumerate(chuoi):
            got = s.update("k", value, frame)
        self.assertLess(abs(got - 100), 2.0)

    def test_van_bam_theo_chuyen_dong_that(self):
        """Khử gai nhưng không được làm đơ: chuyển động thật phải theo kịp."""
        s = AngleSmoother(window=5, alpha=0.4)
        for frame in range(60):
            got = s.update("k", 90.0 + frame, frame)
        # Giá trị thô cuối là 149; bộ lọc nhân quả luôn trễ một nhịp nhưng
        # không được trễ quá vài độ.
        self.assertGreater(got, 140)
        self.assertLess(got, 149)


class TestQuangMatDau(unittest.TestCase):

    def test_reset_sau_khi_mat_dau_qua_lau(self):
        """Nối liền hai đoạn cách nhau nhiều frame sẽ tạo ra chuyển động giả
        không có thật trong video."""
        s = AngleSmoother(window=5, alpha=0.4, max_gap=5)

        for frame in range(10):
            s.update("k", 30.0, frame)

        for frame in range(10, 40):
            s.update("k", None, frame)

        # Sau quãng mất dấu dài, giá trị mới phải được nhận nguyên vẹn
        # thay vì bị kéo về phía 30 độ cũ.
        got = s.update("k", 150.0, 40)
        self.assertAlmostEqual(got, 150.0, places=6)

    def test_mat_dau_ngan_thi_giu_trang_thai(self):
        """Mất dấu 1-2 frame là chuyện thường, không nên vứt hết lịch sử."""
        s = AngleSmoother(window=5, alpha=0.4, max_gap=5)

        for frame in range(10):
            s.update("k", 30.0, frame)

        s.update("k", None, 10)
        got = s.update("k", 150.0, 11)

        self.assertLess(got, 150.0)  # vẫn còn níu bởi lịch sử -> chưa reset

    def test_cac_khop_doc_lap_nhau(self):
        s = AngleSmoother()
        s.update(("LEFT_KNEE", "angle_2d"), 10.0, 0)
        got = s.update(("RIGHT_KNEE", "angle_2d"), 170.0, 0)
        self.assertAlmostEqual(got, 170.0)

    def test_reset_thu_cong(self):
        s = AngleSmoother()
        s.update("k", 30.0, 0)
        s.reset("k")
        self.assertAlmostEqual(s.update("k", 150.0, 1), 150.0)

        s.update("a", 10.0, 0)
        s.update("b", 20.0, 0)
        s.reset()
        self.assertAlmostEqual(s.update("a", 99.0, 1), 99.0)


class TestApply(unittest.TestCase):

    def test_bo_sung_khoa_smooth_va_giu_gia_tri_tho(self):
        s = AngleSmoother()
        angles = {"LEFT_ELBOW": {"angle_2d": 90.0, "angle_3d": 80.0}}

        s.apply(angles, 0)

        data = angles["LEFT_ELBOW"]
        self.assertAlmostEqual(data["angle_2d"], 90.0)      # thô còn nguyên
        self.assertAlmostEqual(data["angle_2d_smooth"], 90.0)
        self.assertAlmostEqual(data["angle_3d_smooth"], 80.0)

    def test_khop_none_khong_lam_vo_apply(self):
        s = AngleSmoother()
        angles = {"LEFT_ELBOW": None, "RIGHT_ELBOW": {"angle_2d": 90.0, "angle_3d": None}}

        s.apply(angles, 0)

        self.assertIsNone(angles["LEFT_ELBOW"])
        self.assertIsNone(angles["RIGHT_ELBOW"]["angle_3d_smooth"])

    def test_khop_mat_dau_dai_duoc_dem_qua_apply(self):
        """apply() phải thấy được frame mất dấu thì cơ chế reset mới chạy."""
        s = AngleSmoother(max_gap=3)

        for frame in range(5):
            s.apply({"LEFT_ELBOW": {"angle_2d": 30.0, "angle_3d": None}}, frame)

        for frame in range(5, 20):
            s.apply({"LEFT_ELBOW": None}, frame)

        angles = {"LEFT_ELBOW": {"angle_2d": 150.0, "angle_3d": None}}
        s.apply(angles, 20)

        self.assertAlmostEqual(angles["LEFT_ELBOW"]["angle_2d_smooth"], 150.0, places=6)


if __name__ == "__main__":
    unittest.main()
