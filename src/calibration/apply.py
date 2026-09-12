"""Apply a fitted calibrator to detection scores."""
from __future__ import annotations

from src.calibration.methods import Calibrator
from src.data.schema import Detection


def apply_scores(dets: list[Detection], phi: Calibrator, expert_id: str | None = None) -> list[Detection]:
    if not dets:
        return []
    scores = phi.transform([d.score for d in dets])
    out = []
    for det, score in zip(dets, scores, strict=True):
        out.append(
            Detection(
                image_id=det.image_id,
                category_id=det.category_id,
                bbox=list(det.bbox),
                score=float(score),
                expert_id=expert_id or det.expert_id,
            )
        )
    return out
