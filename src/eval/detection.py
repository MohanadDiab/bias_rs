"""Detection metrics (COCO AP/AR when pycocotools is available)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from src.data.schema import Detection, as_coco_annotations
from src.eval.matching import match_detections


def summarize_detections(dets: list[Detection], coco_gt: dict, iou_thr: float = 0.5) -> dict[str, Any]:
    labeled = match_detections(dets, coco_gt, iou_thr=iou_thr)
    n_gt = len(coco_gt.get("annotations", []))
    tp = sum(1 for _, ok in labeled if ok)
    fp = sum(1 for _, ok in labeled if not ok)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    metrics: dict[str, Any] = {
        "n_dets": len(dets),
        "n_gt": n_gt,
        "tp": tp,
        "fp": fp,
        "precision": precision,
        "recall": recall,
        "ap50_simple": precision * recall if labeled else 0.0,
    }
    coco_metrics = coco_eval(dets, coco_gt)
    if coco_metrics:
        metrics.update(coco_metrics)
    class_ap = classwise_ap(dets, coco_gt, iou_thr=iou_thr)
    metrics["classwise"] = class_ap
    return metrics


def classwise_ap(dets: list[Detection], coco_gt: dict, iou_thr: float = 0.5) -> dict[str, float]:
    names = {int(c["id"]): c["name"] for c in coco_gt.get("categories", [])}
    out: dict[str, float] = {}
    for cat_id, name in names.items():
        sub_dets = [d for d in dets if d.category_id == cat_id]
        sub_gt = {
            **coco_gt,
            "annotations": [a for a in coco_gt.get("annotations", []) if int(a["category_id"]) == cat_id],
        }
        labeled = match_detections(sub_dets, sub_gt, iou_thr=iou_thr)
        n_gt = len(sub_gt["annotations"])
        tp = sum(1 for _, ok in labeled if ok)
        fp = len(labeled) - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(n_gt, 1)
        out[name] = prec * rec
    return out


def coco_eval(dets: list[Detection], coco_gt: dict) -> dict[str, float] | None:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None
    if not dets:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / "gt.json"
        dt_path = Path(tmp) / "dt.json"
        gt_path.write_text(json.dumps(coco_gt), encoding="utf-8")
        dt_path.write_text(json.dumps(as_coco_annotations(dets)), encoding="utf-8")
        coco = COCO(str(gt_path))
        dt = coco.loadRes(str(dt_path))
        ev = COCOeval(coco, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        s = ev.stats
        return {
            "AP": float(s[0]),
            "AP50": float(s[1]),
            "AP75": float(s[2]),
            "AR": float(s[8]) if len(s) > 8 else float(s[-1]),
        }
