"""Bật camera, tách từng repetition và dự đoán đúng/sai bằng Random Forest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from main import analyze_frame, open_capture, playback_delay, render_frame
from src.angle_calculator import AngleCalculator
from src.angle_smoother import AngleSmoother
from src.pose_detector import PoseDetector
from src.repetition_tracker import RepetitionTracker
from training.live_inference import (
    InsufficientObservationError,
    LiveRepetitionClassifier,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "exercises.json"
WINDOW_NAME = "Realtime Physical Therapy Assessment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá từng repetition trực tiếp từ webcam/video"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Chỉ số webcam (vd 0) hoặc đường dẫn file video",
    )
    parser.add_argument("--exercise", default="Abduction")
    parser.add_argument(
        "--camera-view",
        default="front",
        help="Góc đặt camera giống dữ liệu train: front, left hoặc right",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--artifact-dir", default=str(ROOT / "artifacts"))
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--smooth-alpha", type=float, default=0.4)
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Độ rộng webcam mong muốn",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Độ cao webcam mong muốn",
    )
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument(
        "--angle-dimension",
        choices=(
            "angle_2d_smooth",
            "angle_3d_smooth",
            "angle_2d",
            "angle_3d",
        ),
        default="angle_2d_smooth",
        help="Nguồn góc dùng để tạo feature; phải giống lúc tạo dataset train",
    )
    parser.add_argument("--start-threshold", type=float, default=12.0)
    parser.add_argument("--min-rom", type=float, default=25.0)
    parser.add_argument("--reversal-threshold", type=float, default=8.0)
    parser.add_argument("--end-tolerance", type=float, default=10.0)
    parser.add_argument("--min-observed-ratio", type=float, default=0.5)
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Không lật ngang frame webcam trước khi chạy MediaPipe",
    )
    parser.add_argument("--no-arc", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def _joint_names(feature_names: tuple[str, ...]) -> tuple[str, ...]:
    joints: list[str] = []
    for feature in feature_names:
        for suffix in ("_start", "_turning", "_rom"):
            if feature.endswith(suffix):
                joint = feature[: -len(suffix)].upper()
                if joint not in joints:
                    joints.append(joint)
                break
    if not joints:
        raise ValueError("Không suy ra được tên khớp từ feature của model")
    return tuple(joints)


def _feedback_lines(prediction: dict[str, object], limit: int = 1) -> list[str]:
    comparison = prediction["reference_comparison"]
    features = comparison["features"]
    deviations: list[tuple[float, str]] = []
    for name, detail in features.items():
        if detail.get("status") != "observed":
            continue
        direction = detail.get("direction")
        if direction not in ("lower", "higher"):
            continue
        deviation = float(detail["signed_deviation_deg"])
        short_name = (
            name.replace("left_", "L ")
            .replace("right_", "R ")
            .replace("shoulder_", "shoulder ")
            .replace("elbow_", "elbow ")
            .replace("turning", "turn")
        )
        direction_label = "LOW" if direction == "lower" else "HIGH"
        label = f"{short_name}: {direction_label} ({deviation:+.1f}d)"
        deviations.append((abs(deviation), label))
    deviations.sort(reverse=True)
    return [label for _, label in deviations[:limit]]


def _draw_overlay(
    frame,
    tracker: RepetitionTracker,
    repetition_count: int,
    prediction: dict[str, object] | None,
    prediction_error: str | None,
    visible_joint_count: int,
    total_joint_count: int,
) -> tuple[int, int, int, int]:
    state_labels = {
        "calibrating": "CALIBRATING - hold still",
        "ready": "READY",
        "outbound": "MOVING OUT",
        "returning": "RETURNING",
    }
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (
            f"{state_labels[tracker.state]} | Reps: {repetition_count}",
            (0, 255, 255),
        ),
    ]
    if tracker.driver_joint:
        lines.append(
            (
                (
                    f"Driver: {tracker.driver_joint} | excursion: "
                    f"{tracker.peak_excursion:.1f}d"
                ),
                (255, 255, 255),
            )
        )

    if prediction is not None:
        correct = bool(prediction["correct"])
        probability = float(prediction["correct_probability"])
        color = (0, 220, 0) if correct else (0, 0, 255)
        label = "CORRECT" if correct else "INCORRECT"
        lines.append((f"Last: {label} ({probability:.1%})", color))
        lines.extend((line, (0, 180, 255)) for line in _feedback_lines(prediction))
    elif prediction_error:
        lines.append((f"Last: {prediction_error}", (0, 180, 255)))

    if visible_joint_count < total_joint_count:
        lines.append(
            (
                f"Framing: {visible_joint_count}/{total_joint_count} joints - move back",
                (0, 180, 255),
            )
        )

    lines.append(("q: quit | r: recalibrate", (190, 190, 190)))

    frame_height, frame_width = frame.shape[:2]
    font_scale = max(0.45, min(0.62, frame_width / 1280 * 0.62))
    thickness = 1 if frame_width < 900 else 2
    padding_x = max(10, int(frame_width * 0.012))
    padding_y = max(8, int(frame_height * 0.012))
    line_height = max(22, int(30 * font_scale / 0.62))
    text_sizes = [
        cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )[0]
        for text, _ in lines
    ]
    box_width = min(
        max(width for width, _ in text_sizes) + padding_x * 2,
        frame_width - 16,
    )
    box_height = padding_y * 2 + line_height * len(lines)
    box_x = 8
    box_y = max(58, int(frame_height * 0.09))

    # Chỉ làm tối vùng panel và vẫn giữ camera nhìn xuyên qua được.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for index, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (
                box_x + padding_x,
                box_y + padding_y + line_height * (index + 1) - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return box_x, box_y, box_width, box_height


def run(args: argparse.Namespace) -> int:
    dimensions = (
        args.camera_width,
        args.camera_height,
        args.window_width,
        args.window_height,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError("Kích thước camera/cửa sổ phải là số dương")

    classifier = LiveRepetitionClassifier(
        exercise=args.exercise,
        config_path=args.config,
        artifact_dir=args.artifact_dir,
        min_observed_ratio=args.min_observed_ratio,
    )
    joints = _joint_names(classifier.feature_names)

    cap, _, is_webcam = open_capture(args.source)
    if not cap.isOpened():
        print(f"Không mở được nguồn video: {args.source}")
        return 1

    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, args.window_width, args.window_height)

    detector = PoseDetector(model_complexity=1)
    calculator = AngleCalculator(
        min_visibility=args.min_visibility,
        joints=joints,
    )
    smoother = AngleSmoother(
        window=args.smooth_window,
        alpha=args.smooth_alpha,
    )
    tracker = RepetitionTracker(
        joint_names=joints,
        value_key=args.angle_dimension,
        start_threshold=args.start_threshold,
        min_rom=args.min_rom,
        reversal_threshold=args.reversal_threshold,
        end_tolerance=args.end_tolerance,
    )

    delay = playback_delay(cap, is_webcam)
    frame_id = 0
    repetition_count = 0
    last_prediction: dict[str, object] | None = None
    last_prediction_error: str | None = None

    print("Đã load model. Giữ tư thế bắt đầu ổn định khoảng 0,5 giây.")
    print(f"Camera view dùng cho model: {args.camera_view}")
    print("Nhấn 'q' để thoát, 'r' để hiệu chuẩn lại.")
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Không thể đọc frame hoặc video đã kết thúc.")
                break

            frame, angles, has_pose = analyze_frame(
                frame,
                detector,
                calculator,
                smoother,
                frame_id,
                flip=is_webcam and not args.no_mirror,
            )
            completed = tracker.update(frame_id, angles)
            if completed is not None:
                repetition_count += 1
                try:
                    model_features: dict[str, object] = {
                        **completed.feature_values,
                        "camera_view": args.camera_view,
                    }
                    last_prediction = classifier.predict(model_features)
                    last_prediction_error = None
                    output = {
                        "repetition": repetition_count,
                        "start_frame": completed.start_frame,
                        "turning_frame": completed.turning_frame,
                        "end_frame": completed.end_frame,
                        "driver_joint": completed.driver_joint,
                        "features": model_features,
                        "prediction": last_prediction,
                    }
                    print(json.dumps(output, ensure_ascii=False, indent=2))
                except InsufficientObservationError as error:
                    last_prediction = None
                    last_prediction_error = "NOT ENOUGH VISIBLE JOINTS"
                    print(f"Repetition {repetition_count}: {error}")

            frame = render_frame(
                frame,
                detector,
                calculator,
                angles,
                has_pose,
                show_arc=not args.no_arc,
                value_key=args.angle_dimension,
            )
            visible_joint_count = sum(
                data is not None and data.get(args.angle_dimension) is not None
                for data in angles.values()
            )
            _draw_overlay(
                frame,
                tracker,
                repetition_count,
                last_prediction,
                last_prediction_error,
                visible_joint_count,
                len(joints),
            )
            frame_id += 1

            if args.no_display:
                continue
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracker.reset()
                last_prediction = None
                last_prediction_error = None
                print("Đã reset bộ tách repetition.")
    except KeyboardInterrupt:
        print("Đã dừng theo yêu cầu.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
