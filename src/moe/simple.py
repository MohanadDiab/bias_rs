"""Simple MoE: fuse raw expert scores with a chosen box fusion method."""
from __future__ import annotations

from typing import Any

from src.data.schema import Detection
from src.fusion import fuse


def simple_moe(dets: list[Detection], fusion: str = "wbf", **fuse_kw: Any) -> list[Detection]:
    fused = fuse(dets, method=fusion, **fuse_kw)
    return [
        Detection(d.image_id, d.category_id, list(d.bbox), d.score, expert_id=f"simple_{fusion}")
        for d in fused
    ]
