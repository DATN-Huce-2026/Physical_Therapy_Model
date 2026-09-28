"""Tách một repetition từ chuỗi góc realtime và tạo feature cho model.

Tracker không phụ thuộc MediaPipe. Đầu vào của nó là kết quả đã được tạo bởi
``AngleCalculator``/``AngleSmoother`` cho từng frame. Điều này giúp phần phát
hiện nhịp có thể kiểm thử độc lập với webcam.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class CompletedRepetition:
    """Đặc trưng của một repetition vừa hoàn thành."""

    feature_values: dict[str, float | None]
    driver_joint: str
    start_frame: int
    turning_frame: int
    end_frame: int

    @property
    def observed_feature_ratio(self) -> float:
        if not self.feature_values:
            return 0.0
        observed = sum(value is not None for value in self.feature_values.values())
        return observed / len(self.feature_values)


class RepetitionTracker:
    """State machine phát hiện một chuyển động đi-ra rồi quay-về.

    Quy ước feature của bản realtime đầu tiên:

    - ``start``: trung vị các frame ổn định ngay trước khi chuyển động.
    - ``turning``: trung vị quanh frame có biên độ xa điểm bắt đầu nhất.
    - ``rom``: max - min trên toàn bộ repetition.

    Hai khớp vai được dùng làm tín hiệu phát hiện repetition của bài Abduction.
    Khi bắt đầu chuyển động, tracker tự chọn vai có thay đổi lớn hơn và khóa vai
    đó cho đến khi repetition kết thúc.
    """

    STATES = ("calibrating", "ready", "outbound", "returning")

    def __init__(
        self,
        joint_names: tuple[str, ...],
        *,
        driver_joints: tuple[str, ...] = ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        value_key: str = "angle_2d_smooth",
        baseline_window: int = 15,
        feature_window: int = 5,
        start_threshold: float = 12.0,
        start_hold_frames: int = 3,
        min_rom: float = 25.0,
        reversal_threshold: float = 8.0,
        reversal_hold_frames: int = 2,
        end_tolerance: float = 10.0,
        end_hold_frames: int = 5,
        max_missing_driver_frames: int = 15,
        max_rep_frames: int = 450,
    ) -> None:
        if baseline_window < 3:
            raise ValueError("baseline_window phải >= 3")
        if feature_window < 1:
            raise ValueError("feature_window phải >= 1")
        if start_hold_frames < 1 or reversal_hold_frames < 1 or end_hold_frames < 1:
            raise ValueError("Các tham số hold_frames phải >= 1")
        if not joint_names:
            raise ValueError("joint_names không được để trống")

        self.joint_names = tuple(joint_names)
        self.driver_joints = tuple(
            joint for joint in driver_joints if joint in self.joint_names
        )
        if not self.driver_joints:
            raise ValueError("Cần ít nhất một driver_joint có trong joint_names")

        self.value_key = value_key
        self.baseline_window = baseline_window
        self.feature_window = feature_window
        self.start_threshold = float(start_threshold)
        self.start_hold_frames = start_hold_frames
        self.min_rom = float(min_rom)
        self.reversal_threshold = float(reversal_threshold)
        self.reversal_hold_frames = reversal_hold_frames
        self.end_tolerance = float(end_tolerance)
        self.end_hold_frames = end_hold_frames
        self.max_missing_driver_frames = max_missing_driver_frames
        self.max_rep_frames = max_rep_frames

        self._idle_frames: deque[tuple[int, dict[str, float | None]]] = deque(
            maxlen=baseline_window
        )
        self._candidate_frames: list[tuple[int, dict[str, float | None]]] = []
        self._rep_frames: list[tuple[int, dict[str, float | None]]] = []
        self._candidate_joint: str | None = None
        self._candidate_direction = 0
        self._start_values: dict[str, float | None] = {}
        self._baseline_driver: float | None = None
        self._direction = 0
        self._driver_joint: str | None = None
        self._peak_excursion = 0.0
        self._peak_index = 0
        self._reversal_count = 0
        self._end_count = 0
        self._missing_driver_count = 0
        self.state = "calibrating"
        self.last_event = "Đang lấy mốc góc ban đầu"

    @property
    def driver_joint(self) -> str | None:
        return self._driver_joint

    @property
    def peak_excursion(self) -> float:
        return self._peak_excursion

    def reset(self) -> None:
        """Xóa toàn bộ trạng thái và hiệu chuẩn lại từ đầu."""
        self._reset_runtime()
        self.last_event = "Đã reset; đang lấy lại mốc góc ban đầu"

    def update(
        self, frame_id: int, angles: dict[str, dict[str, object] | None]
    ) -> CompletedRepetition | None:
        """Nhận góc của một frame; trả repetition khi vừa phát hiện hoàn thành."""
        frame_values = self._extract_values(angles)
        frame_record = (frame_id, frame_values)

        if self.state in ("calibrating", "ready"):
            self._update_idle(frame_record)
            return None

        return self._update_active(frame_record)

    def _extract_values(
        self, angles: dict[str, dict[str, object] | None]
    ) -> dict[str, float | None]:
        values: dict[str, float | None] = {}
        for joint in self.joint_names:
            data = angles.get(joint)
            raw_value = data.get(self.value_key) if data else None
            try:
                value = float(raw_value) if raw_value is not None else None
            except (TypeError, ValueError):
                value = None
            if value is not None and not math.isfinite(value):
                value = None
            values[joint] = value
        return values

    def _update_idle(self, record: tuple[int, dict[str, float | None]]) -> None:
        if len(self._idle_frames) < self.baseline_window:
            self._idle_frames.append(record)
            self.state = (
                "ready"
                if len(self._idle_frames) == self.baseline_window
                else "calibrating"
            )
            return

        baseline = self._joint_medians(self._idle_frames)
        frame_id, values = record
        movement = self._largest_driver_movement(values, baseline)

        if movement is None or abs(movement[1]) < self.start_threshold:
            for candidate in self._candidate_frames:
                self._idle_frames.append(candidate)
            self._clear_candidate()
            self._idle_frames.append(record)
            self.state = "ready"
            return

        joint, delta = movement
        direction = 1 if delta > 0 else -1
        if joint != self._candidate_joint or direction != self._candidate_direction:
            for candidate in self._candidate_frames:
                self._idle_frames.append(candidate)
            self._candidate_frames = []
            self._candidate_joint = joint
            self._candidate_direction = direction

        self._candidate_frames.append(record)
        if len(self._candidate_frames) >= self.start_hold_frames:
            self._begin_repetition(frame_id, baseline)

    def _largest_driver_movement(
        self,
        values: dict[str, float | None],
        baseline: dict[str, float | None],
    ) -> tuple[str, float] | None:
        movements: list[tuple[str, float]] = []
        for joint in self.driver_joints:
            value = values.get(joint)
            base = baseline.get(joint)
            if value is not None and base is not None:
                movements.append((joint, value - base))
        if not movements:
            return None
        return max(movements, key=lambda item: abs(item[1]))

    def _begin_repetition(
        self, frame_id: int, baseline: dict[str, float | None]
    ) -> None:
        self._driver_joint = self._candidate_joint
        self._direction = self._candidate_direction
        self._start_values = baseline
        self._baseline_driver = baseline.get(self._driver_joint or "")
        self._rep_frames = list(self._candidate_frames)
        self._peak_excursion = 0.0
        self._peak_index = 0
        self._reversal_count = 0
        self._end_count = 0
        self._missing_driver_count = 0
        self.state = "outbound"
        self.last_event = f"Bắt đầu repetition tại frame {self._rep_frames[0][0]}"

        for index, (_, values) in enumerate(self._rep_frames):
            excursion = self._signed_excursion(values)
            if excursion is not None and excursion >= self._peak_excursion:
                self._peak_excursion = excursion
                self._peak_index = index

        self._clear_candidate()

    def _update_active(
        self, record: tuple[int, dict[str, float | None]]
    ) -> CompletedRepetition | None:
        self._rep_frames.append(record)
        frame_id, values = record
        excursion = self._signed_excursion(values)

        if excursion is None:
            self._missing_driver_count += 1
            if self._missing_driver_count > self.max_missing_driver_frames:
                self._cancel_repetition("Mất dấu khớp điều khiển quá lâu")
            return None
        self._missing_driver_count = 0

        if excursion >= self._peak_excursion:
            self._peak_excursion = excursion
            self._peak_index = len(self._rep_frames) - 1
            self._reversal_count = 0

        if len(self._rep_frames) > self.max_rep_frames:
            self._cancel_repetition("Repetition kéo dài quá giới hạn")
            return None

        if self.state == "outbound":
            reversed_enough = (
                self._peak_excursion >= self.min_rom
                and excursion <= self._peak_excursion - self.reversal_threshold
            )
            self._reversal_count = self._reversal_count + 1 if reversed_enough else 0
            if self._reversal_count >= self.reversal_hold_frames:
                self.state = "returning"
                self.last_event = (
                    f"Đã qua điểm đổi chiều tại frame "
                    f"{self._rep_frames[self._peak_index][0]}"
                )
            return None

        if abs(excursion) <= self.end_tolerance:
            self._end_count += 1
        else:
            self._end_count = 0

        if self._end_count < self.end_hold_frames:
            return None

        completed = self._build_completed(frame_id)
        # Chỉ dùng các frame đã xác nhận quay về điểm đầu. Nếu lấy cả cửa sổ dài
        # hơn, các frame vẫn đang di chuyển sẽ làm lệch baseline của nhịp sau.
        seed_frames = self._rep_frames[-self.end_hold_frames :]
        self._reset_runtime(seed_frames)
        self.last_event = f"Hoàn thành repetition tại frame {frame_id}"
        return completed

    def _signed_excursion(self, values: dict[str, float | None]) -> float | None:
        if self._driver_joint is None or self._baseline_driver is None:
            return None
        value = values.get(self._driver_joint)
        if value is None:
            return None
        return self._direction * (value - self._baseline_driver)

    def _build_completed(self, end_frame: int) -> CompletedRepetition:
        if self._driver_joint is None:
            raise RuntimeError("Không có driver_joint cho repetition")

        turning_frame = self._rep_frames[self._peak_index][0]
        half_window = self.feature_window // 2
        turning_start = max(0, self._peak_index - half_window)
        turning_end = min(len(self._rep_frames), self._peak_index + half_window + 1)
        turning_frames = self._rep_frames[turning_start:turning_end]
        turning_values = self._joint_medians(turning_frames)

        features: dict[str, float | None] = {}
        for joint in self.joint_names:
            start_value = self._start_values.get(joint)
            turning_value = turning_values.get(joint)
            observed = [
                values[joint]
                for _, values in self._rep_frames
                if values.get(joint) is not None
            ]
            if start_value is not None:
                observed.append(start_value)
            rom = max(observed) - min(observed) if observed else None

            prefix = joint.lower()
            features[f"{prefix}_start"] = self._rounded(start_value)
            features[f"{prefix}_turning"] = self._rounded(turning_value)
            features[f"{prefix}_rom"] = self._rounded(rom)

        return CompletedRepetition(
            feature_values=features,
            driver_joint=self._driver_joint,
            start_frame=self._rep_frames[0][0],
            turning_frame=turning_frame,
            end_frame=end_frame,
        )

    def _cancel_repetition(self, reason: str) -> None:
        # Một nhịp bị hủy không cung cấp mốc đứng yên đáng tin cậy.
        self._reset_runtime()
        self.last_event = f"Bỏ repetition: {reason}"

    def _reset_runtime(
        self, seed_frames: list[tuple[int, dict[str, float | None]]] | None = None
    ) -> None:
        self._idle_frames.clear()
        if seed_frames:
            self._idle_frames.extend(seed_frames[-self.baseline_window :])
        self._clear_candidate()
        self._rep_frames = []
        self._start_values = {}
        self._baseline_driver = None
        self._direction = 0
        self._driver_joint = None
        self._peak_excursion = 0.0
        self._peak_index = 0
        self._reversal_count = 0
        self._end_count = 0
        self._missing_driver_count = 0
        self.state = (
            "ready" if len(self._idle_frames) >= self.baseline_window else "calibrating"
        )

    def _clear_candidate(self) -> None:
        self._candidate_frames = []
        self._candidate_joint = None
        self._candidate_direction = 0

    def _joint_medians(self, frames: object) -> dict[str, float | None]:
        frame_list = list(frames)
        result: dict[str, float | None] = {}
        for joint in self.joint_names:
            observed = [
                values[joint]
                for _, values in frame_list
                if values.get(joint) is not None
            ]
            result[joint] = float(median(observed)) if observed else None
        return result

    @staticmethod
    def _rounded(value: float | None) -> float | None:
        return round(float(value), 2) if value is not None else None
