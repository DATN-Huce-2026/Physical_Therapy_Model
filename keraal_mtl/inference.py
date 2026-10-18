"""Chạy Model v5 trên một chuỗi khung xương Kinect [T, 25, 3]."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict_sequence(model, xyz: np.ndarray) -> dict[str, np.ndarray]:
    """Trả xác suất softmax của 6 head.

    Giống lúc train: lấy SpineBase (khớp 0) ở frame đầu làm gốc tọa độ.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    xyz = xyz - xyz[0, 0]
    logits = model(
        torch.from_numpy(xyz).unsqueeze(0),
        torch.tensor([len(xyz)], dtype=torch.long),
    )
    return {
        task: torch.softmax(value[0], dim=-1).numpy() for task, value in logits.items()
    }
