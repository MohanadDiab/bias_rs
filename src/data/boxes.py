"""Axis-aligned box utilities (COCO xywh and xyxy)."""
from __future__ import annotations

from typing import Sequence


def xywh_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return x, y, x + w, y + h


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)]


def clip_xywh(bbox: Sequence[float], img_w: int, img_h: int) -> list[float] | None:
    x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x2, y2 = x + w, y + h
    x = min(max(x, 0.0), float(img_w))
    y = min(max(y, 0.0), float(img_h))
    x2 = min(max(x2, 0.0), float(img_w))
    y2 = min(max(y2, 0.0), float(img_h))
    w, h = x2 - x, y2 - y
    if w < 1.0 or h < 1.0:
        return None
    return [x, y, w, h]


def box_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    return box_iou_xyxy(xywh_to_xyxy(a), xywh_to_xyxy(b))


def box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_giou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    iou = box_iou_xyxy(a, b)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    cx1, cy1 = min(ax1, bx1), min(ay1, by1)
    cx2, cy2 = max(ax2, bx2), max(ay2, by2)
    c_area = max(cx2 - cx1, 0.0) * max(cy2 - cy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    inter_w = max(min(ax2, bx2) - max(ax1, bx1), 0.0)
    inter_h = max(min(ay2, by2) - max(ay1, by1), 0.0)
    union = area_a + area_b - inter_w * inter_h
    if c_area <= 0:
        return iou
    return iou - (c_area - union) / c_area
