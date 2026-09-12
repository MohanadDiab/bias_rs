"""Unified detection dump schema used by every backend and combiner."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PRED_FIELDS = ("image_id", "category_id", "bbox", "score", "expert_id")


@dataclass
class Detection:
    image_id: int
    category_id: int
    bbox: list[float]
    score: float
    expert_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": int(self.image_id),
            "category_id": int(self.category_id),
            "bbox": [float(x) for x in self.bbox],
            "score": float(self.score),
            "expert_id": str(self.expert_id),
        }


def detection_from_dict(row: dict[str, Any]) -> Detection:
    return Detection(
        image_id=int(row["image_id"]),
        category_id=int(row["category_id"]),
        bbox=[float(x) for x in row["bbox"]],
        score=float(row["score"]),
        expert_id=str(row.get("expert_id", row.get("expert", ""))),
    )


def dump_detections(dets: Iterable[Detection | dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for det in dets:
        if isinstance(det, Detection):
            payload.append(det.to_dict())
        else:
            payload.append(detection_from_dict(det).to_dict())
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_detections(path: str | Path) -> list[Detection]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "annotations" in raw:
        raw = raw["annotations"]
    return [detection_from_dict(row) for row in raw]


def group_by_image(dets: Iterable[Detection]) -> dict[int, list[Detection]]:
    out: dict[int, list[Detection]] = {}
    for det in dets:
        out.setdefault(det.image_id, []).append(det)
    return out


def group_by_expert(dets: Iterable[Detection]) -> dict[str, list[Detection]]:
    out: dict[str, list[Detection]] = {}
    for det in dets:
        out.setdefault(det.expert_id, []).append(det)
    return out


def as_coco_annotations(dets: Iterable[Detection]) -> list[dict[str, Any]]:
    rows = []
    for i, det in enumerate(dets, start=1):
        row = det.to_dict()
        row["id"] = i
        rows.append(row)
    return rows
