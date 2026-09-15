"""Xuất tọa độ khớp thô của từng frame ra JSON.

Thay cho test_mediapipe_pose.py trước đây. Script cũ gọi thẳng MediaPipe nên
lặp lại toàn bộ logic của PoseDetector, và bỏ hẳn những frame không nhận ra
người khiến frame_id trong file không còn khớp với frame_id của video gốc.
Script này dùng chung PoseDetector và ghi đủ mọi frame.
"""

import argparse
import json
import os
import sys

import cv2

from src.pose_detector import PoseDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Xuất keypoint từng frame ra JSON")
    parser.add_argument("--source", default="0",
                        help="Chỉ số webcam (vd 0) hoặc đường dẫn file video")
    parser.add_argument("--out-dir", default="landmarks",
                        help="Thư mục chứa file JSON đầu ra")
    parser.add_argument("--no-display", action="store_true",
                        help="Chạy ngầm không mở cửa sổ")
    return parser.parse_args()


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    is_webcam = isinstance(source, int)
    stem = "webcam" if is_webcam else os.path.splitext(os.path.basename(source))[0]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Không mở được nguồn video: {args.source}")
        return 1

    detector = PoseDetector(model_complexity=1)

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    delay = 1 if is_webcam or not fps_src or fps_src <= 0 else max(1, int(1000 / fps_src))

    print("Bắt đầu trích xuất. Nhấn 'q' để thoát.")

    frames = []
    frame_id = 0
    detected = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Không thể đọc frame hoặc video đã kết thúc.")
            break

        if is_webcam:
            frame = cv2.flip(frame, 1)

        frame = detector.find_pose(frame, draw=True)
        landmarks = detector.get_landmarks(frame, normalize_pixel=True)

        entry = {
            "frame_id": frame_id,
            "has_pose": bool(landmarks),
            "landmarks": []
        }

        for index, data in landmarks.items():
            x_norm, y_norm, z_norm = data["coor_3d"]
            entry["landmarks"].append({
                "id": index,
                "name": data["name"],
                "x": round(x_norm, 5),
                "y": round(y_norm, 5),
                "z": round(z_norm, 5),
                "visibility": round(data["visibility"], 4),
                "pixel_x": data["pixel"][0],
                "pixel_y": data["pixel"][1],
            })

        if landmarks:
            detected += 1

        # Ghi cả frame không có người: frame_id phải trùng với frame_id của
        # video gốc thì module phân tích sau mới canh đúng mốc thời gian.
        frames.append(entry)
        frame_id += 1

        if frame_id % 30 == 0:
            print(f"Đã xử lý: {frame_id} frames")

        frame, _ = detector.draw_fps(frame)

        if not args.no_display:
            cv2.imshow("Export Keypoints", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"keypoints_{stem}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_frames": frame_id,
            "frames_with_pose": detected,
            "frames": frames,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nHoàn thành! {detected}/{frame_id} frame có người.")
    print(f"Đã xuất ra: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
