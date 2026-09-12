"""Box fusion combiners (localization methods)."""
from __future__ import annotations

from collections import defaultdict

from src.data.schema import Detection
from src.fusion.nms import nms
from src.fusion.soft_nms import soft_nms
from src.fusion.wbf import wbf

CONF_THR = 0.3
NMS_IOU = 0.65
WBF_IOU = 0.55

FUSION_METHODS = ("nms", "soft_nms", "wbf")
PRUNE_METHODS = ("nms", "soft_nms")


def prune_per_expert(
    dets: list[Detection],
    conf_thr: float = CONF_THR,
    nms_iou: float = NMS_IOU,
    *,
    method: str = "nms",
    sigma: float = 0.5,
) -> list[Detection]:
    filtered = [d for d in dets if d.score >= conf_thr]
    by_expert: dict[str, list[Detection]] = defaultdict(list)
    for det in filtered:
        by_expert[det.expert_id].append(det)
    pruned: list[Detection] = []
    method = method.lower().replace("-", "_")
    if method not in PRUNE_METHODS and method != "softnms":
        raise ValueError(f"Unknown prune method '{method}'. Known: {PRUNE_METHODS}")
    for group in by_expert.values():
        if method in {"soft_nms", "softnms"}:
            pruned.extend(soft_nms(group, iou_thr=nms_iou, sigma=sigma))
        else:
            pruned.extend(nms(group, iou_thr=nms_iou))
    return pruned


def fuse(
    dets: list[Detection],
    method: str = "wbf",
    *,
    prune: bool = True,
    conf_thr: float = CONF_THR,
    nms_iou: float = NMS_IOU,
    wbf_iou: float = WBF_IOU,
    prune_method: str = "nms",
    join_iou: float | None = None,
    sigma: float = 0.5,
) -> list[Detection]:
    method = method.lower().replace("-", "_")
    work = (
        prune_per_expert(
            dets,
            conf_thr=conf_thr,
            nms_iou=nms_iou,
            method=prune_method,
            sigma=sigma,
        )
        if prune
        else [d for d in dets if d.score >= conf_thr]
    )
    join = nms_iou if join_iou is None else join_iou
    if method == "nms":
        return nms(work, iou_thr=join)
    if method in {"soft_nms", "softnms"}:
        return soft_nms(work, iou_thr=join, sigma=sigma)
    if method == "wbf":
        return wbf(work, iou_thr=wbf_iou)
    raise ValueError(f"Unknown fusion method '{method}'. Known: {FUSION_METHODS}")
