"""Apply calibration, gating, suppression, and joining from a pipeline recipe."""
from __future__ import annotations

from typing import Any

from src.calibration.apply import apply_scores
from src.calibration.methods import bakeoff
from src.data.coco import image_size_map
from src.data.schema import Detection, group_by_expert
from src.eval.matching import correctness_arrays
from src.fusion import fuse
from src.moe.gated import SoftmaxGate, apply_gate
from src.moe.learned import FusionMLP, learned_moe
from src.pipeline.load import fuse_kwargs


def _remap_experts(
    dets: list[Detection],
    calibrators: dict,
) -> list[Detection]:
    out: list[Detection] = []
    for expert, group in group_by_expert(dets).items():
        phi = calibrators.get(expert)
        out.extend(apply_scores(group, phi) if phi is not None else group)
    return out


def combine_pipeline(
    cfg: dict[str, Any],
    cal_dets: list[Detection],
    val_dets: list[Detection],
    coco_cal: dict,
    coco_val: dict,
) -> tuple[list[Detection], dict[str, Any]]:
    """Post-hoc stack: calibrate -> gate -> suppress+join.

    Matches the manuscript order: calibrated scores, per-expert suppression
    inside ``fuse``, then cross-expert joining. Gate reweights experts first.
    """
    pipe = cfg.get("pipeline") or {}
    meta: dict[str, Any] = {"pipeline": pipe}
    cal_work = list(cal_dets)
    val_work = list(val_dets)
    kw = fuse_kwargs(cfg)

    cal_cfg = pipe.get("calibrate") or {}
    if cal_cfg.get("enabled", False):
        methods = list(cal_cfg.get("methods") or ["ir", "ts", "platt", "beta", "dirichlet"])
        calibrators = {}
        bake: dict[str, Any] = {}
        for expert, group in group_by_expert(cal_work).items():
            scores, labels = correctness_arrays(group, coco_cal)
            name, phi, table = bakeoff(scores, labels, methods=methods)
            calibrators[expert] = phi
            bake[expert] = {"chosen": name, "ece": table}
        val_work = _remap_experts(val_work, calibrators)
        cal_work = _remap_experts(cal_work, calibrators)
        meta["calibration"] = bake

    gate_cfg = pipe.get("gate") or {}
    if gate_cfg.get("enabled", False):
        gate = SoftmaxGate().fit(cal_work, coco_cal)
        val_work = apply_gate(val_work, gate)
        meta["gate_experts"] = gate.experts

    join = pipe.get("join") or {}
    method = str(join.get("method", "wbf")).lower().replace("-", "_")
    if method == "learned":
        experts = sorted({d.expert_id for d in cal_work})
        head = FusionMLP(experts, num_classes=len(coco_cal.get("categories", [])))
        head.fit(
            cal_work,
            coco_cal,
            iou_thr=float(join.get("iou", 0.55)),
            kfold=3 if len(coco_cal.get("images", [])) < 200 else 1,
        )
        fused = learned_moe(val_work, head, image_size_map(coco_val))
        tag = "learned"
    else:
        fused = fuse(val_work, method=method, **kw)
        tag = method
    fused = [
        Detection(d.image_id, d.category_id, list(d.bbox), d.score, expert_id=f"pipeline_{tag}")
        for d in fused
    ]
    meta["join"] = method
    return fused, meta
