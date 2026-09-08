import cv2
import mediapipe as mp
import time 
import json
import os

# Khởi tạo các module của MediaPipe Pose 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

# Cấu hình mô hình Pose 
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# cap = cv2.VideoCapture(r'D:\DATN\physio_pose_estimation\data\11.mp4')
cap = cv2.VideoCapture(0) # mở web cam

output_json_path = os.path.join("landmarks", "keypoints_wwebcam.json")

prev_time = 0

video_keypoints_data = []

frame_id = 0

print("Bắt đầu khởi động Camera/Video... Nhấn phím 'q' để thoát.")
# print("Bắt đầu: ")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Không thể đọc frame hoặc video đã kết thúc.")
        break

    # Lật ngược ảnh theo chiều ngang nếu dùng webcam để tạo cảm giác soi gương
    frame = cv2.flip(frame, 1)

    # OpenCV đọc ảnh dạng BGR, chuyển sang RGB cho MediaPipe 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Đưa ảnh vào model MediaPipe Pose để trích xuất tọa độ
    results = pose.process(rgb_frame)

    frame_entry = {
        "frame_id": frame_id,
        "has_pose": False,
        "landmarks": []
    }

    # Vẽ khung xương nếu tìm thấy người trong frame
    if results.pose_landmarks: 
        # Vẽ các điểm khớp và đường nối xương
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # landmarks = results.pose_landmarks.landmark
        frame_entry["has_pose"] = True

        # Cách 2A: Lưu thành danh sách các dictionary (dễ đọc, dễ xuất JSON)
        frame_data = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            frame_entry["landmarks"].append({
                "id": idx,
                "name": mp_pose.PoseLandmark(idx).name,
                "x": round(lm.x, 5),
                "y": round(lm.y, 5),
                "z": round(lm.z, 5),
                "visibility": round(lm.visibility, 4)
            })

        video_keypoints_data.append(frame_entry)
        frame_id += 1

        # In tiến trình mỗi 30 frame
        if frame_id % 30 == 0:
            print(f"Đã xử lý: {frame_id} frames")

    # tính toán và hiển thị FPS 
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time= curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('MediaPipe Pose Detection - Test Pipeline', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(video_keypoints_data, f, indent=2, ensure_ascii=False)

print(f"\nHoàn thành! Đã xuất {frame_id} frames ra file: {output_json_path}")

print("\nĐã đóng chương trình.")