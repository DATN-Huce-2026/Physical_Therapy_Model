import argparse
import os
import sys
import time

import cv2

from src.pose_detector import PoseDetector
from src.angle_calculator import AngleCalculator
from src.angle_logger import AngleLogger
from src.angle_smoother import AngleSmoother


def parse_args():
    parser = argparse.ArgumentParser(description="Nhận diện tư thế và tính góc khớp")
    parser.add_argument("--source", default="0",
                        help="Chỉ số webcam (vd 0) hoặc đường dẫn file video")
    parser.add_argument("--out-dir", default="angles",
                        help="Thư mục xuất chuỗi giá trị góc")
    parser.add_argument("--min-visibility", type=float, default=0.5,
                        help="Ngưỡng tin cậy tối thiểu của khớp để tính góc")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Tắt lọc nhiễu, hiển thị và ghi log giá trị góc thô")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Số frame của bộ lọc trung vị (khử gai nhiễu)")
    parser.add_argument("--smooth-alpha", type=float, default=0.4,
                        help="Hệ số EMA: càng nhỏ càng mượt nhưng càng trễ")
    parser.add_argument("--no-arc", action="store_true",
                        help="Chỉ hiện số đo góc, không vẽ cung tròn")
    parser.add_argument("--no-display", action="store_true",
                        help="Chạy ngầm không mở cửa sổ (dùng để xuất log hàng loạt)")
    return parser.parse_args()


def resolve_source(source):
    """Chuỗi số -> chỉ số webcam, còn lại giữ nguyên là đường dẫn file."""
    if source.isdigit():
        return int(source)
    return source


def open_capture(source_arg):
    """Mở nguồn video, kèm nhãn dùng để đặt tên file log."""
    source = resolve_source(source_arg)
    is_webcam = isinstance(source, int)
    stem = "webcam" if is_webcam else os.path.splitext(os.path.basename(source))[0]

    return cv2.VideoCapture(source), stem, is_webcam


def playback_delay(cap, is_webcam):
    """Video file phát theo đúng fps gốc; webcam lấy frame nhanh nhất có thể."""
    fps_src = cap.get(cv2.CAP_PROP_FPS)

    if is_webcam or not fps_src or fps_src <= 0:
        return 1
    return max(1, int(1000 / fps_src))


def analyze_frame(frame, detector, calculator, smoother, frame_id, flip=False):
    """Nhận diện tư thế và tính góc cho một frame."""
    if flip:
        # Lật ngang để người tập nhìn như soi gương.
        frame = cv2.flip(frame, 1)

    frame = detector.find_pose(frame, draw=True)

    # normalize_pixel=True để 'pixel' là tọa độ điểm ảnh thật: vừa dùng tính
    # góc 2D đúng tỉ lệ, vừa dùng làm vị trí vẽ cung tròn.
    landmarks = detector.get_landmarks(frame, normalize_pixel=True)
    world_landmarks = detector.get_world_landmarks()

    # Luôn gọi compute kể cả khi mất dấu người: hàm trả về đủ mọi khớp với giá
    # trị None, nhờ đó bộ lọc đếm được quãng mất dấu để tự reset trạng thái,
    # thay vì nối liền hai đoạn cách xa nhau thành chuyển động giả.
    angles = calculator.compute(landmarks, world_landmarks)

    if smoother is not None:
        angles = smoother.apply(angles, frame_id)

    return frame, angles, bool(landmarks)


def render_frame(frame, detector, calculator, angles, has_pose, show_arc, value_key):
    """Vẽ góc và FPS đè lên khung skeleton."""
    if has_pose:
        frame = calculator.draw(frame, angles, show_arc=show_arc, value_key=value_key)

    frame, _ = detector.draw_fps(frame)
    return frame


def frame_timestamp(cap, is_webcam, start_time):
    """Video lấy mốc thời gian trong file để không lệch khi xử lý chậm hơn
    thời gian thực; webcam lấy theo đồng hồ."""
    if is_webcam:
        return time.time() - start_time
    return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


def print_summary(logger, use_smooth):
    label = "đã lọc nhiễu" if use_smooth else "thô"
    print(f"\nTầm vận động ghi nhận được (độ, tính trên chuỗi {label}):")

    for name, stat in logger.summary().items():
        if stat["count"] == 0:
            print(f"  {name:<16} không đủ dữ liệu tin cậy")
        else:
            print(f"  {name:<16} min={stat['min']:>6}  max={stat['max']:>6}  "
                  f"mean={stat['mean']:>6}  ROM={stat['rom']:>6}  ({stat['count']} frames)")


def run_capture(cap, args, detector, calculator, smoother, logger, is_webcam):
    """Vòng lặp xử lý chính. Trả về số frame đã xử lý."""
    delay = playback_delay(cap, is_webcam)
    use_smooth = smoother is not None
    value_key = "angle_2d_smooth" if use_smooth else "angle_2d"

    frame_id = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Không thể đọc frame hoặc video đã kết thúc.")
            break

        frame, angles, has_pose = analyze_frame(
            frame, detector, calculator, smoother, frame_id, flip=is_webcam
        )

        frame = render_frame(
            frame, detector, calculator, angles, has_pose,
            show_arc=not args.no_arc, value_key=value_key
        )

        # Ghi cả frame không có người để trục thời gian không bị đứt quãng.
        logger.add(frame_id, frame_timestamp(cap, is_webcam, start_time), angles, has_pose)
        frame_id += 1

        if args.no_display:
            continue

        cv2.imshow("Physiotheraphy Pose Estimation", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    return frame_id


def main():
    args = parse_args()

    cap, stem, is_webcam = open_capture(args.source)

    if not cap.isOpened():
        print(f"Không mở được nguồn video: {args.source}")
        return 1

    detector = PoseDetector(model_complexity=1)
    calculator = AngleCalculator(min_visibility=args.min_visibility)

    use_smooth = not args.no_smooth
    smoother = None
    if use_smooth:
        smoother = AngleSmoother(window=args.smooth_window, alpha=args.smooth_alpha)

    logger = AngleLogger(calculator.joint_names, include_smooth=use_smooth)

    print("Hệ thống đang khởi động. Nhấn 'q' để thoát.")
    print(f"Đang tính góc cho {len(calculator.joint_names)} khớp: "
          f"{', '.join(calculator.joint_names)}")

    try:
        total_frames = run_capture(
            cap, args, detector, calculator, smoother, logger, is_webcam
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    csv_path = logger.to_csv(os.path.join(args.out_dir, f"angles_{stem}.csv"))
    json_path = logger.to_json(os.path.join(args.out_dir, f"angles_{stem}.json"))

    print(f"\nĐã xử lý {total_frames} frames.")
    print(f"Chuỗi góc (CSV) : {csv_path}")
    print(f"Chuỗi góc (JSON): {json_path}")

    print_summary(logger, use_smooth)

    return 0


if __name__ == "__main__":
    sys.exit(main())
