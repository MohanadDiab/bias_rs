"""Calibrated MoE: apply per-expert φ then fuse."""
from __future__ import annotations

from typing import Any

from src.calibration.apply import apply_scores
from src.calibration.methods import Calibrator
from src.data.schema import Detection, group_by_expert
from src.fusion import fuse


def calibrated_moe(
    dets: list[Detection],
    calibrators: dict[str, Calibrator],
    fusion: str = "wbf",
    **fuse_kw: Any,
) -> list[Detection]:
    calibrated: list[Detection] = []
    for expert, group in group_by_expert(dets).items():
        phi = calibrators.get(expert)
        if phi is None:
            calibrated.extend(group)
        else:
            calibrated.extend(apply_scores(group, phi))
    fused = fuse(calibrated, method=fusion, **fuse_kw)
    return [
        Detection(d.image_id, d.category_id, list(d.bbox), d.score, expert_id=f"calibrated_{fusion}")
        for d in fused
    ]
