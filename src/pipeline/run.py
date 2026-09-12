"""Execute one pipeline recipe: train experts, dump preds, combine, eval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.coco import load_coco
from src.data.paths import resolve_split_json
from src.data.schema import Detection, dump_detections, load_detections
from src.eval.cost import cost_summary
from src.experiments.matrix import compose_expert_cfg
from src.experiments.runner import _evaluate, _train_predict
from src.pipeline.combine import combine_pipeline
from src.pipeline.load import load_pipeline
from src.training.config import ROOT, deep_merge


def _noise_tag(cfg: dict[str, Any]) -> str:
    family = str(cfg["noise"]["family"])
    pct = int(cfg["noise"]["ratio"])
    if pct <= 0 or family.lower() == "clean":
        return "clean"
    return f"{family}_{pct}"


def pipeline_out_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("output", {}).get("root") or "outputs/pipelines")
    if not root.is_absolute():
        root = ROOT / root
    ds = Path(cfg["dataset_yaml"]).stem
    return root / cfg["name"] / ds / _noise_tag(cfg)


def _expert_cfgs(pipe: dict[str, Any]) -> list[dict[str, Any]]:
    family = str(pipe["noise"]["family"])
    pct = int(pipe["noise"]["ratio"])
    if family.lower() == "clean":
        pct = 0
        family = "clean"
    cfgs = [
        compose_expert_cfg(pipe["dataset_yaml"], spec, family, pct)
        for spec in pipe["expert_specs"]
    ]
    train_override = pipe.get("train") or {}
    if train_override:
        for cfg in cfgs:
            cfg["train"] = deep_merge(cfg.get("train", {}), train_override)
    return cfgs


def _load_preds(cfg: dict[str, Any], split: str) -> list[Detection]:
    project = Path(cfg["train"].get("project", cfg["output"]["root"]))
    path = project / cfg["name"] / f"predictions_{split}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_detections(path)


def run_pipeline(
    pipe: dict[str, Any],
    *,
    do_train: bool,
    do_predict: bool,
    do_eval: bool,
    skip_missing: bool,
) -> dict[str, Any]:
    out_dir = pipeline_out_dir(pipe)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pipeline.json").write_text(
        json.dumps({k: v for k, v in pipe.items() if k != "expert_specs"}, indent=2, default=str),
        encoding="utf-8",
    )
    expert_cfgs = _expert_cfgs(pipe)
    run_dirs = []
    cal_dets: list[Detection] = []
    val_dets: list[Detection] = []
    eval_split = str((pipe.get("eval") or {}).get("split") or "val")
    for cfg in expert_cfgs:
        try:
            run_dir = _train_predict(cfg, do_train=do_train, do_predict=do_predict)
            run_dirs.append(run_dir)
            cal_split = cfg["dataset"]["cal_split"]
            cal_dets.extend(_load_preds(cfg, cal_split))
            val_dets.extend(_load_preds(cfg, eval_split))
        except FileNotFoundError as exc:
            if skip_missing:
                print(f"skip missing {cfg['name']}: {exc}")
                continue
            raise
    report: dict[str, Any] = {
        "name": pipe["name"],
        "dataset": pipe["dataset_yaml"],
        "noise": pipe["noise"],
        "n_experts_loaded": len({d.expert_id for d in val_dets}),
        "experts": [c["expert_id"] for c in expert_cfgs],
        "cost": cost_summary(run_dirs, n_experts=len(expert_cfgs)),
        "output": str(out_dir),
    }
    if not val_dets:
        report["status"] = "skipped_no_predictions"
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    if do_eval:
        coco_cal = load_coco(resolve_split_json(expert_cfgs[0], expert_cfgs[0]["dataset"]["cal_split"]))
        coco_val = load_coco(resolve_split_json(expert_cfgs[0], eval_split))
        fused, extras = combine_pipeline(pipe, cal_dets, val_dets, coco_cal, coco_val)
        dump_detections(fused, out_dir / f"predictions_{eval_split}.json")
        metrics = _evaluate(fused, val_dets, coco_val, out_dir)
        report.update(extras)
        report["metrics"] = metrics
        report["status"] = "ok"
    else:
        report["status"] = "predicted"
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def run_from_files(
    pipeline: str | Path | None,
    *,
    dataset: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    do_train: bool = False,
    do_predict: bool = False,
    do_eval: bool = True,
    skip_missing: bool = True,
) -> dict[str, Any]:
    cfg = load_pipeline(pipeline, dataset=dataset, overrides=overrides)
    return run_pipeline(
        cfg,
        do_train=do_train,
        do_predict=do_predict,
        do_eval=do_eval,
        skip_missing=skip_missing,
    )
