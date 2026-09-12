"""Match detections to ground-truth boxes (greedy IoU)."""
from __future__ import annotations

from collections import defaultdict

from src.data.boxes import box_iou_xywh
from src.data.coco import anns_by_image
from src.data.schema import Detection


def match_detections(
    dets: list[Detection],
    coco_gt: dict,
    iou_thr: float = 0.5,
) -> list[tuple[Detection, bool]]:
    """Return (det, correct) pairs. A det is correct if matched to a same-class GT at IoU >= thr."""
    grouped = anns_by_image(coco_gt)
    by_image: dict[int, list[Detection]] = defaultdict(list)
    for det in dets:
        by_image[det.image_id].append(det)
    labeled: list[tuple[Detection, bool]] = []
    for image_id, preds in by_image.items():
        gts = grouped.get(image_id, [])
        used = [False] * len(gts)
        for det in sorted(preds, key=lambda d: d.score, reverse=True):
            best_j = -1
            best_iou = iou_thr
            for j, gt in enumerate(gts):
                if used[j]:
                    continue
                if int(gt["category_id"]) != det.category_id:
                    continue
                iou = box_iou_xywh(det.bbox, gt["bbox"])
                if iou >= best_iou:
                    best_iou = iou
                    best_j = j
            ok = best_j >= 0
            if ok:
                used[best_j] = True
            labeled.append((det, ok))
    return labeled


def correctness_arrays(
    dets: list[Detection],
    coco_gt: dict,
    iou_thr: float = 0.5,
) -> tuple[list[float], list[int]]:
    labeled = match_detections(dets, coco_gt, iou_thr=iou_thr)
    scores = [d.score for d, _ in labeled]
    labels = [1 if ok else 0 for _, ok in labeled]
    return scores, labels
