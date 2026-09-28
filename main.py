import argparse
import os
import re
import sys
import time
from pathlib import Path

import cv2

from src.pose_detector import PoseDetector
from src.angle_calculator import AngleCalculator
from src.angle_logger import AngleLogger, ExerciseAngleLogger
from src.angle_smoother import AngleSmoother
from src.rep_segmenter import RepSegmenter
from create_labels import create_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Nhận diện tư thế và tính góc khớp")
    # Tham so cu cho che do mot video/webcam; giu lai dang comment theo yeu cau.
    '''
    parser.add_argument("--source", default="0",
                        help="Chỉ số webcam (vd 0) hoặc đường dẫn file video")
    parser.add_argument("--out-dir", default="angles",
                        help="Thư mục xuất chuỗi giá trị góc")
    '''
    parser.add_argument("--input-dir", default="data",
                        help="Folder video; folder con dau tien la ten bai tap")
    parser.add_argument("--output-file", default="angles/angles.csv",
                        help="Duong dan file CSV tong hop")
    parser.add_argument("--angle-dimension", default="2d_smooth",
                        choices=("2d", "2d_smooth", "3d", "3d_smooth"),
                        help="Loai goc ghi vao CSV")
    parser.add_argument("--rep-joints", nargs="+", default=None,
                        choices=tuple(AngleCalculator.JOINT_TRIPLETS),
                        help="Cac khop dung de tach rep, vd LEFT_KNEE RIGHT_KNEE")
    parser.add_argument("--rep-min-amplitude", type=float, default=4.0,
                        help="Bien do goc toi thieu (do) de tinh la mot lan doi chieu")
    parser.add_argument("--rep-min-distance", type=int, default=6,
                        help="So frame toi thieu giua hai diem doi chieu")
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


# Sua khoa nay de khop chinh xac ten thu muc bai tap cua bo du lieu.
# Vi du: videos/squat/rep_01.mp4 chi ghi goc hong va goi hai ben.
EXERCISE_JOINTS = {
    "elbow": ("LEFT_ELBOW", "RIGHT_ELBOW"),
    "shoulder": ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    "hip": ("LEFT_HIP", "RIGHT_HIP"),
    "knee": ("LEFT_KNEE", "RIGHT_KNEE"),
    "squat": ("LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"),
    "sit_to_stand": ("LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"),
}
EXERCISE_CODES = {
    "E01": "Abduction",
    "E02": "Adduction",
    "E03": "Lateral_Rotation",
    "E04": "Medial_Rotation",
    "E05": "Circumduction",
    "E06": "Wrist_Extension",
    "E07": "Hip_Joint_Flexion",
    "E08": "Lumbar_Flexion",
    "E09": "Back_Extension",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}


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

    try:
        frame = detector.find_pose(frame, draw=True)
    except (RuntimeError, ValueError, cv2.error) as error:
        # Một frame hỏng hoặc lỗi TFLite không nên làm dừng cả batch.
        print(f"Bo qua frame {frame_id} do loi MediaPipe: {error}")
        detector.results = None
        angles = calculator.compute({}, {})
        if smoother is not None:
            angles = smoother.apply(angles, frame_id)
        return frame, angles, False

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


def discover_videos(input_dir):
    """Tim tat ca video trong input_dir, ke ca cac thu muc con."""
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Khong tim thay folder video: {root}")
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def video_labels(video_path, input_dir):
    """Doi ma E01-E09 trong ten file thanh ten bai tap de ghi vao CSV."""
    root = Path(input_dir).resolve()
    match = re.search(r"(?<![A-Za-z0-9])E0[1-9](?![A-Za-z0-9])", video_path.stem.upper())
    if match and match.group() in EXERCISE_CODES:
        exercise = EXERCISE_CODES[match.group()]
    else:
        exercise = video_path.parent.name if video_path.parent.resolve() != root else video_path.stem
    return exercise, video_path.stem


def joints_for_exercise(exercise):
    """Tra ve cac goc can ghi; bai tap chua khai bao se giu toan bo goc."""
    key = exercise.strip().lower().replace("-", "_").replace(" ", "_")
    return EXERCISE_JOINTS.get(key, tuple(AngleCalculator.JOINT_TRIPLETS))


def detect_reps(samples, report_joint_names, args):
    """Chon tin hieu dem rep it bi anh huong boi cac khop khong lien quan."""
    segmenter = RepSegmenter(
        min_distance=args.rep_min_distance,
        min_amplitude=args.rep_min_amplitude,
    )
    if args.rep_joints:
        return segmenter.find_reps(samples, args.rep_joints), tuple(args.rep_joints)

    # Thu moi cap trai/phai dang co trong bai tap. Vi du squat thu cap goi va
    # cap hong rieng, thay vi lay trung vi cua ca bon khop (co the lam mo dao
    # chieu). Cap tim thay nhieu chu ky hoan chinh nhat se duoc chon.
    pairs = (
        ("LEFT_ELBOW", "RIGHT_ELBOW"),
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_KNEE", "RIGHT_KNEE"),
    )
    candidates = [pair for pair in pairs if all(name in report_joint_names for name in pair)]
    candidates.append(tuple(report_joint_names))

    best_reps = []
    best_joints = tuple(report_joint_names)
    for candidate in candidates:
        reps = segmenter.find_reps(samples, candidate)
        if len(reps) > len(best_reps):
            best_reps, best_joints = reps, candidate
    return best_reps, best_joints


def run_batch_capture(cap, args, detector, calculator, smoother, batch_logger,
                      exercise, rep_id):
    """Xu ly video, tach rep va chi ghi ROM cua tung rep vao CSV."""
    frame_id = 0
    value_key = f"angle_{args.angle_dimension}"
    samples = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame, angles, has_pose = analyze_frame(
            frame, detector, calculator, smoother, frame_id
        )
        sample = {}
        for joint_name, data in angles.items():
            value = data.get(value_key) if data else None
            if value is None and data and value_key.endswith("_smooth"):
                value = data.get(value_key.removesuffix("_smooth"))
            sample[joint_name] = value
        samples.append(sample)
        frame_id += 1

        # Che do hien thi video da duoc tat: khi chay batch chi tao CSV.
        # if args.no_display:
        #     continue
        #
        # frame = render_frame(
        #     frame, detector, calculator, angles, has_pose,
        #     show_arc=not args.no_arc, value_key=value_key
        # )
        # cv2.imshow("Physiotherapy Pose Estimation", frame)
        # if cv2.waitKey(delay) & 0xFF == ord("q"):
        #     break

    # Khop dung de dem rep co the khac voi cac khop can xuat trong CSV. Vi du,
    # squat nen dem bang hai dau goi, nhung van xuat ca goc hong.
    report_joint_names = joints_for_exercise(exercise)
    reps, signal_joint_names = detect_reps(samples, report_joint_names, args)
    for rep_number, (start, turning, end) in enumerate(reps, start=1):
        measurements = {}
        for joint_name in report_joint_names:
            values = [sample.get(joint_name) for sample in samples[start:end + 1]]
            values = [value for value in values if value is not None]
            measurements[joint_name] = {
                "start": samples[start].get(joint_name),
                "turning": samples[turning].get(joint_name),
                "rom": max(values) - min(values) if values else None,
            }
        batch_logger.add_rep(exercise, f"{rep_id}_{rep_number}", measurements)

    print(f"  Tim thay {len(reps)} rep hoan chinh (dem bang: {', '.join(signal_joint_names)}).")
    return frame_id


def main_batch():
    args = parse_args()
    try:
        videos = discover_videos(args.input_dir)
    except FileNotFoundError as error:
        print(error)
        return 1

    if not videos:
        print(f"Khong co video hop le trong: {args.input_dir}")
        return 1

    batch_logger = ExerciseAngleLogger()
    use_smooth = not args.no_smooth
    total_frames = 0

    for video_path in videos:
        exercise, rep_id = video_labels(video_path, args.input_dir)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Bo qua video khong mo duoc: {video_path}")
            continue

        detector = PoseDetector(model_complexity=1)
        # Khong gioi han theo bai tap: cot CSV duoc tao theo goc thuc su co
        # gia tri trong bat ky frame/video nao.
        calculator = AngleCalculator(min_visibility=args.min_visibility)
        smoother = (AngleSmoother(window=args.smooth_window, alpha=args.smooth_alpha)
                    if use_smooth else None)
        print(f"Dang xu ly {video_path.name}: exercise={exercise}, rep_id={rep_id}")
        try:
            total_frames += run_batch_capture(
                cap, args, detector, calculator, smoother, batch_logger, exercise, rep_id
            )
        finally:
            cap.release()

    # Khong mo cua so video trong che do batch nen khong can dong cua so.
    # cv2.destroyAllWindows()
    csv_path = batch_logger.to_csv(args.output_file)
    label_count, unmatched_rep_ids, labels_path = create_labels(args.output_file)
    print(f"Da xu ly {len(videos)} video, {total_frames} frame.")
    print(f"CSV tong hop: {csv_path}")
    print(f"CSV nhan: {labels_path} ({label_count} rep).")
    if unmatched_rep_ids:
        print(f"Bo qua {len(unmatched_rep_ids)} rep khong co video_name trong file marking.")
    return 0


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
    # Che do cu chi xu ly mot video/webcam:
    # sys.exit(main())
    sys.exit(main_batch())
