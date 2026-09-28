import cv2
import mediapipe as mp
import numpy as np
import time 

class PoseDetector:
    def __init__(self, 
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.p_time = 0

    def find_pose(self, frame, draw=True):
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Frame rỗng hoặc không phải numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(f"Frame có shape không hợp lệ: {frame.shape}")

        frame = np.ascontiguousarray(frame)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)
        img_rgb.flags.writeable = False
        self.results = self.pose.process(img_rgb)
        img_rgb.flags.writeable = True

        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame

    def get_landmarks(self, frame, normalize_pixel=False):
        landmark_data = {}

        if not self.results.pose_landmarks:
            return landmark_data

        h, w, _ = frame.shape
        landmarks = self.results.pose_landmarks.landmark

        for index, lm in enumerate(landmarks):
            landmark_name = self.mp_pose.PoseLandmark(index).name

            x_val = lm.x
            y_val = lm.y

            if normalize_pixel:
                x_val = int(lm.x * w)
                y_val = int(lm.y * h)

            landmark_data[index] = {
                "name": landmark_name,
                "pixel": (x_val, y_val), # tọa độ pixel (int) 
                "coor_2d": (lm.x, lm.y), # tọa độ chuẩn hóa [0.0, 1.0] -> tính góc 2D
                "coor_3d": (lm.x, lm.y, lm.z), # tọa độ không gian 3D -> tính góc 3D nếu cần
                "visibility": float(lm.visibility) # độ tin cậy [0.0, 1.0] -> kiểm tra trước khi tính góc
            }
        return landmark_data

    def get_world_landmarks(self):
        """Tọa độ 3D thực (đơn vị mét, gốc tại trung điểm hông).

        Khác với coor_3d trong get_landmarks (đã chuẩn hóa theo khung hình nên
        bị méo theo tỉ lệ ảnh và phối cảnh), dữ liệu này có tỉ lệ metric đồng
        nhất trên cả 3 trục -> dùng để tính góc 3D mới chính xác.
        """
        world_data = {}

        if not getattr(self, "results", None) or not self.results.pose_world_landmarks:
            return world_data

        for index, lm in enumerate(self.results.pose_world_landmarks.landmark):
            world_data[index] = {
                "name": self.mp_pose.PoseLandmark(index).name,
                "coor_3d": (lm.x, lm.y, lm.z),
                "visibility": float(lm.visibility)
            }
        return world_data

    def draw_fps(self, frame, pos=(20, 50), color=(0, 255, 0)):
        c_time = time.time()
        if (c_time - self.p_time) > 0:
            fps = 1 / (c_time - self.p_time)
        else:
            fps = 0
        self.p_time = c_time

        cv2.putText(
            frame, 
            f"FPS: {int(fps)}",
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA
        )

        return frame, fps