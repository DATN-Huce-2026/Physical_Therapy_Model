"""Bảng nhãn của 6 head Model v5 (từ ``label_maps.py`` của repo train)."""

# Thứ tự lớp phải giống hệt lúc train.
TASK_LABELS: dict[str, tuple[str, ...]] = {
    "exercise": ("CTK", "ELK", "RTK"),
    "exercise_error_class": ("Correct", "Error1", "Error2", "Error3"),
    "global_evaluation": ("Correct", "Incorrect", "Incomplete", "Motionless"),
    "error_type": ("none", "Error1", "Error2", "Error3", "Other"),
    "severity": ("SmallError", "LargeError"),
    "body_part": ("Spine", "BothArms", "RightArm", "LeftArm"),
}
