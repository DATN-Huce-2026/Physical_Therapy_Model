import math

import cv2
import numpy as np
import mediapipe as mp

PL = mp.solutions.pose.PoseLandmark


class AngleCalculator:
    """Tính góc khớp từ dữ liệu landmark do PoseDetector trả về.

    - Góc 2D: tính trên tọa độ PIXEL. Không dùng tọa độ chuẩn hóa [0,1] vì
      khung hình không vuông (vd 1280x720) làm hai trục co giãn khác nhau,
      khiến góc bị méo. Pixel cho đúng góc nhìn thấy trên video.
    - Góc 3D: tính trên world landmarks (mét) -> không phụ thuộc phối cảnh.
    """

    # Bộ 3 khớp trọng yếu: (điểm đầu, ĐỈNH GÓC, điểm cuối)
    JOINT_TRIPLETS = {
        "LEFT_ELBOW":     (PL.LEFT_SHOULDER,  PL.LEFT_ELBOW,     PL.LEFT_WRIST),
        "RIGHT_ELBOW":    (PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW,    PL.RIGHT_WRIST),
        "LEFT_SHOULDER":  (PL.LEFT_ELBOW,     PL.LEFT_SHOULDER,  PL.LEFT_HIP),
        "RIGHT_SHOULDER": (PL.RIGHT_ELBOW,    PL.RIGHT_SHOULDER, PL.RIGHT_HIP),
        "LEFT_HIP":       (PL.LEFT_SHOULDER,  PL.LEFT_HIP,       PL.LEFT_KNEE),
        "RIGHT_HIP":      (PL.RIGHT_SHOULDER, PL.RIGHT_HIP,      PL.RIGHT_KNEE),
        "LEFT_KNEE":      (PL.LEFT_HIP,       PL.LEFT_KNEE,      PL.LEFT_ANKLE),
        "RIGHT_KNEE":     (PL.RIGHT_HIP,      PL.RIGHT_KNEE,     PL.RIGHT_ANKLE),
    }

    def __init__(self, min_visibility=0.5, joints=None):
        self.min_visibility = min_visibility

        if joints is None:
            self.triplets = dict(self.JOINT_TRIPLETS)
        else:
            self.triplets = {k: self.JOINT_TRIPLETS[k] for k in joints}

    @property
    def joint_names(self):
        return list(self.triplets.keys())

    @staticmethod
    def angle_between(point_a, point_b, point_c):
        """Góc tại đỉnh B tạo bởi 3 điểm A-B-C, theo công thức tích vô hướng.

            cos(theta) = (BA . BC) / (|BA| * |BC|)

        Dùng chung cho cả 2D và 3D (numpy tự suy theo số chiều đầu vào).
        Trả về độ (0-180), hoặc None nếu có vector suy biến (độ dài 0).
        """
        ba = np.asarray(point_a, dtype=np.float64) - np.asarray(point_b, dtype=np.float64)
        bc = np.asarray(point_c, dtype=np.float64) - np.asarray(point_b, dtype=np.float64)

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return None

        cos_theta = float(np.dot(ba, bc) / (norm_ba * norm_bc))
        # Chặn biên: sai số dấu phẩy động có thể đẩy giá trị ra ngoài [-1, 1]
        # làm arccos trả về NaN.
        cos_theta = max(-1.0, min(1.0, cos_theta))

        return math.degrees(math.acos(cos_theta))

    def compute(self, landmarks, world_landmarks=None):
        """Tính toàn bộ góc trọng yếu cho MỘT frame.

        landmarks: dict từ PoseDetector.get_landmarks(frame, normalize_pixel=True)
        world_landmarks: dict từ PoseDetector.get_world_landmarks() (tùy chọn)

        Trả về dict {tên_khớp: {...}} - khớp không đủ tin cậy có giá trị None
        nhưng KEY VẪN TỒN TẠI, để chuỗi thời gian đầu ra luôn cùng độ dài.
        """
        angles = {}

        for name, (idx_a, idx_b, idx_c) in self.triplets.items():
            ids = (idx_a.value, idx_b.value, idx_c.value)

            if not all(i in landmarks for i in ids):
                angles[name] = None
                continue

            points = [landmarks[i] for i in ids]
            visibility = min(p["visibility"] for p in points)

            # Bỏ qua khớp bị che khuất: góc tính từ điểm đoán mò sẽ nhiễu nặng.
            if visibility < self.min_visibility:
                angles[name] = None
                continue

            angle_2d = self.angle_between(*[p["pixel"] for p in points])

            angle_3d = None
            if world_landmarks and all(i in world_landmarks for i in ids):
                angle_3d = self.angle_between(
                    *[world_landmarks[i]["coor_3d"] for i in ids]
                )

            angles[name] = {
                "angle_2d": angle_2d,
                "angle_3d": angle_3d,
                "vertex_px": tuple(points[1]["pixel"]),
                "points_px": tuple(tuple(p["pixel"]) for p in points),
                "visibility": visibility,
            }

        return angles

    def draw(self, frame, angles, show_arc=True, value_key="angle_2d",
             color=(0, 255, 255), text_color=(255, 255, 255)):
        """Vẽ cung tròn + số đo góc tại đúng vị trí khớp, đè lên skeleton.

        value_key chọn nguồn số hiển thị: "angle_2d" (thô) hoặc "angle_2d_smooth"
        (đã lọc nhiễu). Cung tròn luôn vẽ theo tọa độ pixel thật của khớp nên
        vị trí không đổi, chỉ con số là khác.
        """
        for data in angles.values():
            if not data:
                continue

            # Chuỗi đã lọc có thể chưa có giá trị ở các frame đầu -> lùi về giá trị thô.
            value = data.get(value_key)
            if value is None:
                value = data.get("angle_2d")
            if value is None:
                continue

            point_a, point_b, point_c = data["points_px"]

            start_deg, sweep_deg, radius = self._arc_geometry(point_a, point_b, point_c)

            if show_arc:
                cv2.ellipse(
                    frame,
                    point_b,
                    (radius, radius),
                    0,
                    start_deg,
                    start_deg + sweep_deg,
                    color,
                    2,
                    cv2.LINE_AA
                )

            self._draw_value(frame, point_b, start_deg + sweep_deg / 2.0, radius, value, text_color)

        return frame

    @staticmethod
    def _arc_geometry(point_a, point_b, point_c):
        """Góc bắt đầu / độ mở của cung, theo hệ tọa độ ảnh của OpenCV.

        OpenCV đo góc ellipse theo chiều kim đồng hồ từ trục x dương, trùng
        đúng quy ước của atan2(dy, dx) khi trục y hướng xuống -> dùng trực tiếp.
        """
        ax, ay = point_a
        bx, by = point_b
        cx, cy = point_c

        len_ba = math.hypot(ax - bx, ay - by)
        len_bc = math.hypot(cx - bx, cy - by)

        # Bán kính co theo chi ngắn hơn để cung không tràn ra ngoài khớp.
        radius = int(max(18, min(55, min(len_ba, len_bc) * 0.35)))

        deg_ba = math.degrees(math.atan2(ay - by, ax - bx))
        deg_bc = math.degrees(math.atan2(cy - by, cx - bx))

        sweep = (deg_bc - deg_ba) % 360

        # Luôn vẽ cung nhỏ (<=180) để khớp với giá trị arccos trả về.
        if sweep > 180:
            deg_ba, sweep = deg_bc, 360 - sweep

        return deg_ba, sweep, radius

    @staticmethod
    def _draw_value(frame, vertex, mid_deg, radius, value, text_color):
        """Đặt text theo hướng phân giác của góc nên không đè lên xương."""
        bx, by = vertex
        offset = radius + 18

        text_x = int(bx + offset * math.cos(math.radians(mid_deg)))
        text_y = int(by + offset * math.sin(math.radians(mid_deg)))

        # Font Hershey của OpenCV không có ký tự "°" nên dùng chữ 'd' thay thế.
        label = f"{value:.0f}d"

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Kéo text về trong khung nếu khớp nằm sát mép hình.
        h, w, _ = frame.shape
        text_x = max(2, min(w - text_w - 2, text_x - text_w // 2))
        text_y = max(text_h + 2, min(h - baseline - 2, text_y + text_h // 2))

        cv2.rectangle(
            frame,
            (text_x - 4, text_y - text_h - 4),
            (text_x + text_w + 4, text_y + baseline),
            (0, 0, 0),
            cv2.FILLED
        )
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
            cv2.LINE_AA
        )
