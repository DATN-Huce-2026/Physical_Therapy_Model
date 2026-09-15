import csv
import json
import os


class AngleLogger:
    """Gom giá trị góc của từng frame thành chuỗi thời gian để phân tích.

    Nguyên tắc: GHI MỌI FRAME, kể cả frame không phát hiện được người (khi đó
    các cột góc để trống). Nhờ vậy frame_id trong file log luôn trùng với
    frame_id của video gốc, module phân tích/đếm nhịp về sau mới canh đúng mốc
    thời gian.
    """

    def __init__(self, joint_names, include_3d=True, include_smooth=True):
        self.joint_names = list(joint_names)
        self.include_3d = include_3d
        self.include_smooth = include_smooth
        self.records = []

    @property
    def dimensions(self):
        """Hậu tố cột theo thứ tự xuất hiện trong file log."""
        dims = ["2d"]
        if self.include_smooth:
            dims.append("2d_smooth")
        if self.include_3d:
            dims.append("3d")
            if self.include_smooth:
                dims.append("3d_smooth")
        return dims

    @property
    def columns(self):
        cols = ["frame_id", "timestamp_sec", "has_pose"]
        for name in self.joint_names:
            for dim in self.dimensions:
                cols.append(f"{name}_{dim}")
        return cols

    def add(self, frame_id, timestamp_sec, angles=None, has_pose=True):
        record = {
            "frame_id": frame_id,
            "timestamp_sec": round(timestamp_sec, 4),
            "has_pose": has_pose,
        }

        angles = angles or {}

        for name in self.joint_names:
            data = angles.get(name)

            for dim in self.dimensions:
                record[f"{name}_{dim}"] = self._round(data, f"angle_{dim}")

        self.records.append(record)
        return record

    @staticmethod
    def _round(data, key):
        if not data or data.get(key) is None:
            return None
        return round(data[key], 2)

    def series(self, joint_name, dimension="2d"):
        """Chuỗi giá trị của một khớp theo thời gian (None = thiếu dữ liệu)."""
        column = f"{joint_name}_{dimension}"
        return [record.get(column) for record in self.records]

    def summary(self, dimension=None):
        """Thống kê nhanh min/max/mean từng khớp - phục vụ đánh giá tầm vận động.

        Mặc định thống kê trên chuỗi ĐÃ LỌC nếu có, vì tầm vận động tính từ
        chuỗi thô sẽ bị gai nhiễu thổi phồng.
        """
        if dimension is None:
            dimension = "2d_smooth" if self.include_smooth else "2d"

        result = {}

        for name in self.joint_names:
            values = [v for v in self.series(name, dimension) if v is not None]

            if not values:
                result[name] = {"count": 0, "min": None, "max": None, "mean": None, "rom": None}
                continue

            result[name] = {
                "count": len(values),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "mean": round(sum(values) / len(values), 2),
                "rom": round(max(values) - min(values), 2),  # range of motion
            }

        return result

    def to_csv(self, path):
        self._ensure_dir(path)

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.records)

        return path

    def to_json(self, path, with_summary=True):
        self._ensure_dir(path)

        payload = {
            "total_frames": len(self.records),
            "joints": self.joint_names,
            "dimensions": self.dimensions,
            "frames": self.records,
        }

        if with_summary:
            payload["summary"] = self.summary()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return path

    @staticmethod
    def _ensure_dir(path):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
