"""Per-class greedy NMS on unified detections."""
from __future__ import annotations

from collections import defaultdict

from src.data.boxes import box_iou_xywh
from src.data.schema import Detection


def nms(dets: list[Detection], iou_thr: float = 0.65) -> list[Detection]:
    by_class: dict[int, list[Detection]] = defaultdict(list)
    for det in dets:
        by_class[det.category_id].append(det)
    keep: list[Detection] = []
    for group in by_class.values():
        order = sorted(group, key=lambda d: d.score, reverse=True)
        suppressed = [False] * len(order)
        for i, a in enumerate(order):
            if suppressed[i]:
                continue
            keep.append(a)
            for j in range(i + 1, len(order)):
                if suppressed[j]:
                    continue
                if box_iou_xywh(a.bbox, order[j].bbox) >= iou_thr:
                    suppressed[j] = True
    return keep
