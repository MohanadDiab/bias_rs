"""Weighted Box Fusion on unified detections.

Fused score is the mean of cluster scores (manuscript / Solovyev).
Coordinates are confidence-weighted averages. Clustering is per class.
"""
from __future__ import annotations

from collections import defaultdict

from src.data.boxes import box_iou_xywh
from src.data.schema import Detection


def wbf(
    dets: list[Detection],
    iou_thr: float = 0.55,
    expert_id: str = "wbf",
) -> list[Detection]:
    by_key: dict[tuple[int, int], list[Detection]] = defaultdict(list)
    for det in dets:
        by_key[(det.image_id, det.category_id)].append(det)
    fused: list[Detection] = []
    for (image_id, category_id), group in by_key.items():
        order = sorted(group, key=lambda d: d.score, reverse=True)
        used = [False] * len(order)
        for i, seed in enumerate(order):
            if used[i]:
                continue
            cluster = [seed]
            used[i] = True
            for j in range(i + 1, len(order)):
                if used[j]:
                    continue
                if box_iou_xywh(seed.bbox, order[j].bbox) >= iou_thr:
                    cluster.append(order[j])
                    used[j] = True
            fused.append(_fuse_cluster(cluster, image_id, category_id, expert_id))
    return fused


def _fuse_cluster(
    cluster: list[Detection],
    image_id: int,
    category_id: int,
    expert_id: str,
) -> Detection:
    weights = [max(d.score, 1e-6) for d in cluster]
    z = sum(weights)
    bbox = [0.0, 0.0, 0.0, 0.0]
    for det, w in zip(cluster, weights, strict=True):
        for k in range(4):
            bbox[k] += w * float(det.bbox[k])
    bbox = [c / z for c in bbox]
    score = sum(d.score for d in cluster) / len(cluster)
    return Detection(
        image_id=image_id,
        category_id=category_id,
        bbox=bbox,
        score=float(score),
        expert_id=expert_id,
    )


def cluster_detections(dets: list[Detection], iou_thr: float = 0.55) -> list[list[Detection]]:
    """IoU clusters used by learned MoE (same threshold as WBF)."""
    by_image: dict[int, list[Detection]] = defaultdict(list)
    for det in dets:
        by_image[det.image_id].append(det)
    clusters: list[list[Detection]] = []
    for group in by_image.values():
        order = sorted(group, key=lambda d: d.score, reverse=True)
        used = [False] * len(order)
        for i, seed in enumerate(order):
            if used[i]:
                continue
            cluster = [seed]
            used[i] = True
            for j in range(i + 1, len(order)):
                if used[j]:
                    continue
                if box_iou_xywh(seed.bbox, order[j].bbox) >= iou_thr:
                    cluster.append(order[j])
                    used[j] = True
            clusters.append(cluster)
    return clusters
