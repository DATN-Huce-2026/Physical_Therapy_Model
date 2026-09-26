"""Create rep-level binary labels from angles.csv and physiotherapist scores."""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_ANGLES = ROOT / "angles" / "angles.csv"
DEFAULT_MARKING = ROOT / "Physiotherapist Exercise Marking .csv"
DEFAULT_OUTPUT = ROOT / "labels" / "labels.csv"


def load_video_scores(path):
    """Read the marking sheet, carrying merged video-name cells forward."""
    scores = {}
    current_video = ""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = csv.reader(source)
        top_headers = next(rows, None)
        sub_headers = next(rows, None)
        if not top_headers or not sub_headers:
            return scores
        headers = [
            sub.strip() or (top_headers[index].strip() if index < len(top_headers) else "")
            for index, sub in enumerate(sub_headers)
        ]
        video_index = headers.index("Video Name")
        score_index = headers.index("Average Score")
        for row in rows:
            current_video = (row[video_index] if len(row) > video_index else "").strip() or current_video
            raw_score = (row[score_index] if len(row) > score_index else "").strip()
            if current_video and raw_score:
                scores[current_video] = float(raw_score)
    return scores


def create_labels(angles_path=DEFAULT_ANGLES, marking_path=DEFAULT_MARKING,
                 output_path=DEFAULT_OUTPUT):
    angles_path = Path(angles_path)
    marking_path = Path(marking_path)
    output_path = Path(output_path)
    if not angles_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy angles.csv: {angles_path}")
    if not marking_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file marking: {marking_path}")

    scores = load_video_scores(marking_path)
    output_rows = []
    missing = set()
    with angles_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"exercise", "rep_id"}.issubset(reader.fieldnames or []):
            raise ValueError("angles.csv cần có hai cột exercise và rep_id")
        for row in reader:
            rep_id = (row.get("rep_id") or "").strip()
            video_name = next(
                (name for name in scores if rep_id == name or rep_id.startswith(name + "_")),
                None,
            )
            if video_name is None:
                missing.add(rep_id)
                continue
            output_rows.append({
                "exercise": row.get("exercise", ""),
                "rep_id": rep_id,
                "label": int(scores[video_name] >= 80),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["exercise", "rep_id", "label"])
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows), missing, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create labels.csv with label 1 for Average Score >= 80, else 0."
    )
    parser.add_argument("--angles", type=Path, default=DEFAULT_ANGLES)
    parser.add_argument("--marking", type=Path, default=DEFAULT_MARKING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        count, missing, output_path = create_labels(args.angles, args.marking, args.output)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Đã tạo {output_path} với {count} dòng.")
    if missing:
        print(f"Bỏ qua {len(missing)} rep không tìm thấy video_name tương ứng trong file marking.")


if __name__ == "__main__":
    main()
