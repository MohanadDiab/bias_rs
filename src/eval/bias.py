"""Bias diagnostics: agreement, fused-vs-expert IoU, coverage by box size."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.data.boxes import box_iou_xywh
from src.data.coco import anns_by_image
from src.data.schema import Detection, group_by_expert, group_by_image
from src.eval.matching import match_detections


def pairwise_agreement(
    dets: list[Detection],
    iou_thr: float = 0.5,
) -> dict[str, float]:
    by_exp = group_by_expert(dets)
    experts = sorted(by_exp)
    out: dict[str, float] = {}
    for i, a in enumerate(experts):
        for b in experts[i + 1 :]:
            out[f"{a}__{b}"] = _agreement(by_exp[a], by_exp[b], iou_thr)
    return out


def _agreement(a: list[Detection], b: list[Detection], iou_thr: float) -> float:
    by_img_a = group_by_image(a)
    by_img_b = group_by_image(b)
    matched = 0
    total = 0
    for image_id in set(by_img_a) | set(by_img_b):
        ga = by_img_a.get(image_id, [])
        gb = by_img_b.get(image_id, [])
        used = [False] * len(gb)
        for da in ga:
            total += 1
            best = -1
            best_iou = iou_thr
            for j, db in enumerate(gb):
                if used[j] or da.category_id != db.category_id:
                    continue
                iou = box_iou_xywh(da.bbox, db.bbox)
                if iou >= best_iou:
                    best_iou = iou
                    best = j
            if best >= 0:
                used[best] = True
                matched += 1
        total += sum(1 for u in used if not u)
    if total == 0:
        return 0.0
    return matched / total


def fused_vs_expert_iou(fused: list[Detection], dets: list[Detection]) -> dict[str, float]:
    by_exp = group_by_expert(dets)
    out: dict[str, float] = {}
    for expert, group in by_exp.items():
        ious = []
        by_img_f = group_by_image(fused)
        by_img_e = group_by_image(group)
        for image_id, fgroup in by_img_f.items():
            egroup = by_img_e.get(image_id, [])
            for fd in fgroup:
                best = 0.0
                for ed in egroup:
                    if ed.category_id != fd.category_id:
                        continue
                    best = max(best, box_iou_xywh(fd.bbox, ed.bbox))
                ious.append(best)
        out[expert] = float(sum(ious) / len(ious)) if ious else 0.0
    return out


def coverage_by_size(
    dets: list[Detection],
    coco_gt: dict,
    iou_thr: float = 0.5,
    n_bins: int = 5,
) -> dict[str, Any]:
    areas = []
    for ann in coco_gt.get("annotations", []):
        w, h = float(ann["bbox"][2]), float(ann["bbox"][3])
        areas.append(w * h)
    if not areas:
        return {"bins": [], "per_expert": {}}
    edges = _quantile_edges(areas, n_bins)
    labeled = match_detections(dets, coco_gt, iou_thr=iou_thr)
    hits: dict[str, list[int]] = defaultdict(lambda: [0] * n_bins)
    totals = [0] * n_bins
    grouped = anns_by_image(coco_gt)
    gt_area = {int(a["id"]): float(a["bbox"][2]) * float(a["bbox"][3]) for a in coco_gt.get("annotations", [])}
    # approximate: bin GT boxes and mark covered if any correct det of that expert
    by_exp_ok: dict[str, set[tuple[int, int]]] = defaultdict(set)
    # we don't have gt ids in match_detections; use image+class+iou via second pass
    by_exp = group_by_expert(dets)
    for expert, group in by_exp.items():
        matched = match_detections(group, coco_gt, iou_thr=iou_thr)
        # count unique images with at least one correct in each size bin via GT loop
        _ = matched
    for ann in coco_gt.get("annotations", []):
        area = float(ann["bbox"][2]) * float(ann["bbox"][3])
        b = _bin_index(area, edges)
        totals[b] += 1
        for expert, group in by_exp.items():
            ok = False
            for det in group:
                if det.image_id != int(ann["image_id"]) or det.category_id != int(ann["category_id"]):
                    continue
                if box_iou_xywh(det.bbox, ann["bbox"]) >= iou_thr:
                    ok = True
                    break
            if ok:
                hits[expert][b] += 1
    per_expert = {
        expert: [h / t if t else 0.0 for h, t in zip(vals, totals, strict=True)]
        for expert, vals in hits.items()
    }
    return {"edges": edges, "gt_per_bin": totals, "per_expert": per_expert}


def _quantile_edges(values: list[float], n_bins: int) -> list[float]:
    xs = sorted(values)
    edges = [xs[0]]
    for i in range(1, n_bins):
        edges.append(xs[min(len(xs) - 1, int(i * len(xs) / n_bins))])
    edges.append(xs[-1])
    return edges


def _bin_index(value: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        if value <= edges[i + 1]:
            return i
    return len(edges) - 2
