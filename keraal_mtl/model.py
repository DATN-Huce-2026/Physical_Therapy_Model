"""Kiến trúc multi-task cho chuỗi khung xương Kinect KERAAL.

Sao chép từ repo train DATN-Huce-2026/physiotherapy_multi_task_learning
(commit 273f826, ``src/rehab_mtl/models/shared_backbone.py`` và
``datasets/label_maps.py``) để load được ``best_model.pt`` của Model v5 mà không
cần cài repo đó. Tên module/tham số phải giữ nguyên để khớp với state_dict.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .labels import TASK_LABELS


class KeraalSharedBackbone(nn.Module):
    """Frame encoder + BiGRU dùng chung, mỗi nhiệm vụ một head Linear."""

    def __init__(
        self,
        *,
        num_joints: int = 25,
        coordinates_per_joint: int = 3,
        frame_embedding_size: int = 64,
        recurrent_hidden_size: int = 64,
        dropout: float = 0.4,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "last"):
            raise ValueError("pooling phải là 'mean' hoặc 'last'")
        self.num_joints = num_joints
        self.coordinates_per_joint = coordinates_per_joint
        self.pooling = pooling

        self.frame_encoder = nn.Sequential(
            nn.Linear(num_joints * coordinates_per_joint, frame_embedding_size),
            nn.LayerNorm(frame_embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared_temporal_backbone = nn.GRU(
            input_size=frame_embedding_size,
            hidden_size=recurrent_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.feature_dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(recurrent_hidden_size * 2, len(labels))
                for task, labels in TASK_LABELS.items()
            }
        )

    def forward(
        self, xyz: torch.Tensor, lengths: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """``xyz``: [batch, time, 25, 3]; ``lengths``: số frame thật mỗi mẫu."""
        if xyz.ndim != 4 or tuple(xyz.shape[2:]) != (
            self.num_joints,
            self.coordinates_per_joint,
        ):
            raise ValueError("xyz phải có shape [batch, time, 25, 3]")
        batch_size, padded_time = xyz.shape[:2]
        lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.long)
        if bool((lengths_cpu < 1).any()) or bool((lengths_cpu > padded_time).any()):
            raise ValueError("lengths phải nằm trong [1, số frame đã pad]")

        features = self.frame_encoder(xyz.reshape(batch_size, padded_time, -1))
        packed = pack_padded_sequence(
            features, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_outputs, final_hidden = self.shared_temporal_backbone(packed)

        if self.pooling == "mean":
            outputs, _ = pad_packed_sequence(
                packed_outputs, batch_first=True, total_length=padded_time
            )
            time_index = torch.arange(padded_time, device=xyz.device).unsqueeze(0)
            mask = (time_index < lengths.unsqueeze(1)).unsqueeze(-1)
            pooled = (outputs * mask).sum(dim=1) / lengths.unsqueeze(1).clamp(
                min=1
            ).to(outputs.dtype)
        else:
            pooled = torch.cat((final_hidden[-2], final_hidden[-1]), dim=1)

        shared = self.feature_dropout(pooled)
        return {task: head(shared) for task, head in self.heads.items()}


def load_model(path: str | Path, *, pooling: str = "mean") -> KeraalSharedBackbone:
    """Load state_dict an toàn (``weights_only``) và chuyển sang chế độ eval."""
    model = KeraalSharedBackbone(pooling=pooling)
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
