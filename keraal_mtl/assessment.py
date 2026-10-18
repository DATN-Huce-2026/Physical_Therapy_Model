"""Diễn giải 6 đầu ra của Model v5: đúng/sai, sai ở đâu, điểm 0-100.

Ba head ``global_evaluation``, ``exercise_error_class`` và ``error_type`` đều có
một lớp "đúng" (Correct/none). Với mỗi head, điểm thành phần là

    P(đúng) / (P(đúng) + P(lớp lỗi mạnh nhất))

tức so lớp "đúng" với đối thủ mạnh nhất của nó. Không dùng thẳng P(đúng) vì model
được train với class weight đẩy xác suất các lớp hiếm (Incomplete, Motionless,
Other) lên, khiến P(đúng) thường < 50% ngay cả khi "đúng" là lớp cao nhất.
Điểm = 100 × trung bình ba điểm thành phần; lần tập là ĐÚNG khi điểm >=
``pass_score`` (mặc định 50, tức các head nghiêng về "đúng").

Khi SAI, loại lỗi lấy từ trung bình xác suất Error1/2/3 của hai head
``exercise_error_class`` và ``error_type`` (``Other`` chỉ có ở ``error_type``);
bộ phận sai và mức độ lấy từ head ``body_part`` và ``severity``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .labels import TASK_LABELS

DEFAULT_PASS_SCORE = 50.0
ERROR_CODES = ("Error1", "Error2", "Error3", "Other")

GLOBAL_EVALUATION_VI = {
    "Correct": "Đúng",
    "Incorrect": "Sai động tác",
    "Incomplete": "Chưa hoàn thành động tác",
    "Motionless": "Không có chuyển động",
}
SEVERITY_VI = {"SmallError": "nhẹ", "LargeError": "nặng"}
BODY_PART_VI = {
    "Spine": "cột sống / thân",
    "BothArms": "cả hai tay",
    "RightArm": "tay phải",
    "LeftArm": "tay trái",
}
GRADES = (
    (80.0, "Tốt"),
    (65.0, "Khá"),
    (50.0, "Đạt"),
    (35.0, "Chưa đạt"),
    (0.0, "Kém"),
)
DEFAULT_DESCRIPTIONS = Path(__file__).with_name("error_descriptions.json")


def grade_of(score: float) -> str:
    for minimum, label in GRADES:
        if score >= minimum:
            return label
    return GRADES[-1][1]


def load_descriptions(path: str | Path | None = None) -> dict[str, object]:
    with Path(path or DEFAULT_DESCRIPTIONS).open(encoding="utf-8") as file:
        return json.load(file)


def _top(task: str, probabilities: np.ndarray) -> tuple[str, float]:
    index = int(np.argmax(probabilities))
    return TASK_LABELS[task][index], round(float(probabilities[index]), 4)


def assess(
    probabilities: dict[str, np.ndarray],
    *,
    expected_exercise: str | None = None,
    pass_score: float = DEFAULT_PASS_SCORE,
    descriptions: dict[str, object] | None = None,
) -> dict[str, object]:
    """``probabilities``: task -> vector softmax theo thứ tự ``TASK_LABELS``."""
    if not 0 < pass_score < 100:
        raise ValueError("pass_score phải nằm trong (0, 100)")
    descriptions = descriptions or load_descriptions()
    probs = {task: np.asarray(probabilities[task], dtype=float) for task in TASK_LABELS}

    exercise, exercise_conf = _top("exercise", probs["exercise"])
    labels = TASK_LABELS
    # Lớp số 0 của ba head này đều là "đúng" (Correct/none).
    correct_heads = ("global_evaluation", "exercise_error_class", "error_type")
    head_scores = {
        task: float(probs[task][0] / max(probs[task][0] + probs[task][1:].max(), 1e-12))
        for task in correct_heads
    }
    score = round(100.0 * float(np.mean(list(head_scores.values()))), 1)
    passed = score >= pass_score

    # Gộp xác suất từng loại lỗi từ hai head cùng mã lỗi.
    eec = dict(zip(labels["exercise_error_class"], probs["exercise_error_class"]))
    err = dict(zip(labels["error_type"], probs["error_type"]))
    error_probabilities = {
        code: float(np.mean([eec[code], err[code]])) if code in eec else float(err[code])
        for code in ERROR_CODES
    }

    # Lý do sai: lớp có xác suất cao nhất trong các lớp khác "Correct".
    wrong_globals = probs["global_evaluation"][1:]
    reason = labels["global_evaluation"][1 + int(np.argmax(wrong_globals))]

    result: dict[str, object] = {
        "exercise": exercise,
        "exercise_name": descriptions["exercise_names"].get(exercise, exercise),
        "exercise_confidence": exercise_conf,
        "score": score,
        "pass_score": pass_score,
        "passed": passed,
        "verdict": "ĐÚNG" if passed else "SAI",
        "grade": grade_of(score),
        "head_predictions": {
            task: dict(zip(("label", "confidence"), _top(task, probs[task])))
            for task in TASK_LABELS
        },
        "warnings": [],
    }
    if expected_exercise and expected_exercise != exercise:
        result["warnings"].append(
            f"Model nhận ra bài {exercise} ({exercise_conf:.0%}) nhưng bài cần tập là "
            f"{expected_exercise}: có thể tập nhầm bài hoặc động tác sai nhiều."
        )
    if exercise_conf < 0.6:
        result["warnings"].append(
            f"Model không chắc đây là bài nào (độ tin cậy {exercise_conf:.0%})."
        )

    if passed:
        result["error"] = None
        return result

    error_code = max(error_probabilities, key=error_probabilities.get)
    body_part, body_conf = _top("body_part", probs["body_part"])
    severity, severity_conf = _top("severity", probs["severity"])
    error_texts = descriptions["errors"].get(exercise, {})
    description = error_texts.get(error_code, error_code)
    result["error"] = {
        "reason": reason,
        "reason_vi": GLOBAL_EVALUATION_VI[reason],
        "error_type": error_code,
        "error_type_confidence": round(error_probabilities[error_code], 4),
        "description": description,
        "body_part": body_part,
        "body_part_vi": BODY_PART_VI[body_part],
        "body_part_confidence": body_conf,
        "severity": severity,
        "severity_vi": SEVERITY_VI[severity],
        "severity_confidence": severity_conf,
        "message": (
            f"{GLOBAL_EVALUATION_VI[reason]}: {description} "
            f"({error_probabilities[error_code]:.0%}) - bộ phận: "
            f"{BODY_PART_VI[body_part]} ({body_conf:.0%}) - mức độ: "
            f"{SEVERITY_VI[severity]} ({severity_conf:.0%})"
        ),
    }
    return result

