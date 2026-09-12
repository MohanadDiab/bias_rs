"""Gaussian Soft-NMS on unified detections."""
from __future__ import annotations

import math
from collections import defaultdict

from src.data.boxes import box_iou_xywh
from src.data.schema import Detection


def soft_nms(
    dets: list[Detection],
    iou_thr: float = 0.65,
    sigma: float = 0.5,
    score_thr: float = 0.001,
) -> list[Detection]:
    by_class: dict[int, list[Detection]] = defaultdict(list)
    for det in dets:
        by_class[det.category_id].append(det)
    keep: list[Detection] = []
    for group in by_class.values():
        remaining = [
            Detection(d.image_id, d.category_id, list(d.bbox), d.score, d.expert_id) for d in group
        ]
        while remaining:
            remaining.sort(key=lambda d: d.score, reverse=True)
            best = remaining.pop(0)
            if best.score < score_thr:
                break
            keep.append(best)
            nxt = []
            for det in remaining:
                iou = box_iou_xywh(best.bbox, det.bbox)
                if iou > 0:
                    det.score *= math.exp(-(iou * iou) / sigma)
                if det.score >= score_thr:
                    nxt.append(det)
            remaining = nxt
    return keep
