"""Run detect-and-combine jobs from a pipeline YAML or a legacy matrix."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from src.calibration.methods import bakeoff
from src.data.coco import image_size_map, load_coco
from src.data.paths import resolve_split_json
from src.data.schema import Detection, dump_detections, group_by_expert, load_detections
from src.eval.bias import coverage_by_size, fused_vs_expert_iou, pairwise_agreement
from src.eval.calibration import ece, reliability_diagram
from src.eval.cost import cost_summary
from src.eval.detection import summarize_detections
from src.eval.matching import correctness_arrays, match_detections
from src.eval.plots import save_reliability_plot, save_score_kde
from src.experiments.matrix import compose_expert_cfg, expand_jobs, load_matrix
from src.experiments.pipeline import expert_train_cfg, fuse_kwargs, run_out_dir
from src.experiments.schema import PipelineRun
from src.moe import FusionMLP, SoftmaxGate, calibrated_moe, gated_moe, learned_moe, simple_moe
from src.training.config import ROOT
from src.training.registry import get_trainer
from src.training.weights import find_run_checkpoint


def _job_dir(job: dict[str, Any]) -> Path:
    ds = Path(job["dataset_yaml"]).stem
    return (
        ROOT
        / "outputs"
        / "experiments"
        / str(job["family"])
        / ds
        / f"{job['family']}_{job['pct']}"
        / f"{job['expert_set']}__{job['moe']}__{job['fusion']}"
    )


def _predict_splits(cfg: dict[str, Any], requested: list[str] | None = None) -> list[str]:
    dataset = cfg["dataset"]
    aliases = {
        "cal": dataset.get("cal_split", "cal"),
        "val": dataset.get("val_split", "val"),
        "train": dataset.get("train_split", "train"),
    }
    names = requested or [dataset.get("cal_split", "cal"), dataset.get("val_split", "val")]
    resolved = []
    for name in names:
        resolved.append(aliases.get(str(name), str(name)))
    return list(dict.fromkeys(resolved))


def _train_predict(
    cfg: dict[str, Any],
    *,
    do_train: bool,
    do_predict: bool,
    prepare_only: bool = False,
    splits: list[str] | None = None,
) -> Path:
    trainer = get_trainer(cfg["backend"])
    data_path = trainer.prepare(cfg)
    project = Path(cfg["train"].get("project", cfg["output"]["root"]))
    run_dir = project / cfg["name"]
    if prepare_only:
        return run_dir
    if do_train:
        existing = find_run_checkpoint(cfg)
        if existing is not None:
            print(f"skip existing train {cfg['name']}: {existing}")
            cfg.setdefault("model", {})["weights"] = str(existing)
        else:
            run_dir = trainer.train(cfg, data_path)
            found = find_run_checkpoint(cfg)
            if found is not None:
                cfg.setdefault("model", {})["weights"] = str(found)
    if do_predict:
        for split in _predict_splits(cfg, splits):
            trainer.predict(cfg, data_path, split)
    return run_dir


def _load_split_preds(cfg: dict[str, Any], split: str) -> list[Detection]:
    project = Path(cfg["train"].get("project", cfg["output"]["root"]))
    path = project / cfg["name"] / f"predictions_{split}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_detections(path)


def _pipeline_from_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "moe": job["moe"],
        "combine": {"method": job["fusion"], "iou": job.get("wbf_iou", 0.55)},
        "prune": {
            "method": job.get("prune_method", "nms"),
            "iou": job.get("nms_iou", 0.65),
            "conf": job.get("conf_thr", 0.3),
        },
        "calibration": {
            "enabled": job["moe"] == "calibrated",
            "methods": job.get("cal_methods") or ["ir", "ts", "platt", "lr", "beta", "dirichlet"],
        },
        "gate": {"enabled": job["moe"] == "gated", "type": "softmax"},
        "eval": {"plots": True},
    }


def _combine(
    pipe: dict[str, Any],
    cal_dets: list[Detection],
    val_dets: list[Detection],
    coco_cal: dict,
    coco_val: dict,
) -> tuple[list[Detection], dict[str, Any]]:
    moe = str(pipe.get("moe") or "simple")
    combine = str((pipe.get("combine") or {}).get("method") or "wbf").lower().replace("-", "_")
    kwargs = fuse_kwargs({"pipeline": pipe})
    extras: dict[str, Any] = {"fusion": combine, "moe": moe, "fuse": kwargs}
    if moe == "learned" or combine == "learned":
        experts = sorted({d.expert_id for d in cal_dets})
        head = FusionMLP(experts, num_classes=len(coco_cal.get("categories", [])))
        head.fit(
            cal_dets,
            coco_cal,
            iou_thr=float((pipe.get("combine") or {}).get("iou", 0.55)),
            kfold=3 if len(coco_cal.get("images", [])) < 200 else 1,
        )
        fused = learned_moe(val_dets, head, image_size_map(coco_val))
        extras["fusion"] = "learned"
        return fused, extras
    if moe == "simple":
        return simple_moe(val_dets, fusion=combine, **kwargs), extras
    if moe == "calibrated":
        methods = list((pipe.get("calibration") or {}).get("methods") or [])
        calibrators = {}
        bake: dict[str, Any] = {}
        for expert, group in group_by_expert(cal_dets).items():
            scores, labels = correctness_arrays(group, coco_cal)
            name, phi, table = bakeoff(scores, labels, methods=methods or None)
            calibrators[expert] = phi
            bake[expert] = {"chosen": name, "ece": table}
        extras["calibration"] = bake
        return calibrated_moe(val_dets, calibrators, fusion=combine, **kwargs), extras
    if moe == "gated":
        gate = SoftmaxGate().fit(cal_dets, coco_cal)
        extras["gate_experts"] = gate.experts
        return gated_moe(val_dets, gate, fusion=combine, **kwargs), extras
    raise ValueError(f"Unknown moe '{moe}'")


def _evaluate(
    dets: list[Detection],
    raw: list[Detection],
    coco_gt: dict,
    out_dir: Path,
    *,
    plots: bool = True,
) -> dict[str, Any]:
    metrics = summarize_detections(dets, coco_gt)
    scores, labels = correctness_arrays(dets, coco_gt)
    metrics["ece"] = ece(scores, labels)
    rel = reliability_diagram(scores, labels)
    metrics["reliability"] = rel
    labeled = match_detections(dets, coco_gt)
    correct = [d.score for d, ok in labeled if ok]
    incorrect = [d.score for d, ok in labeled if not ok]
    if plots:
        save_reliability_plot(rel["confidence"], rel["accuracy"], out_dir / "reliability.png")
        save_score_kde(correct, incorrect, out_dir / "scores_correct_incorrect.png")
    metrics["agreement"] = pairwise_agreement(raw)
    metrics["fused_vs_expert_iou"] = fused_vs_expert_iou(dets, raw)
    metrics["coverage"] = coverage_by_size(raw, coco_gt)
    return metrics


def run_job(
    job: dict[str, Any],
    *,
    do_train: bool,
    do_predict: bool,
    do_eval: bool,
    skip_missing: bool,
) -> dict[str, Any]:
    out_dir = _job_dir(job)
    out_dir.mkdir(parents=True, exist_ok=True)
    expert_cfgs = [
        compose_expert_cfg(job["dataset_yaml"], spec, job["family"] if job["pct"] else "clean", job["pct"])
        for spec in job["experts"]
    ]
    run_dirs = []
    cal_dets: list[Detection] = []
    val_dets: list[Detection] = []
    for cfg in expert_cfgs:
        try:
            run_dir = _train_predict(cfg, do_train=do_train, do_predict=do_predict)
            run_dirs.append(run_dir)
            cal_dets.extend(_load_split_preds(cfg, cfg["dataset"]["cal_split"]))
            val_dets.extend(_load_split_preds(cfg, cfg["dataset"]["val_split"]))
        except FileNotFoundError as exc:
            if skip_missing:
                print(f"skip missing {cfg['name']}: {exc}")
                continue
            raise
    report: dict[str, Any] = {
        "job": {k: v for k, v in job.items() if k != "experts"},
        "n_experts_loaded": len({d.expert_id for d in val_dets}),
        "cost": cost_summary(run_dirs, n_experts=len(expert_cfgs)),
    }
    if not val_dets:
        report["status"] = "skipped_no_predictions"
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    if do_eval:
        coco_cal = load_coco(resolve_split_json(expert_cfgs[0], expert_cfgs[0]["dataset"]["cal_split"]))
        coco_val = load_coco(resolve_split_json(expert_cfgs[0], expert_cfgs[0]["dataset"]["val_split"]))
        fused, extras = _combine(_pipeline_from_job(job), cal_dets, val_dets, coco_cal, coco_val)
        dump_detections(fused, out_dir / "predictions_val.json")
        metrics = _evaluate(fused, val_dets, coco_val, out_dir)
        report.update(extras)
        report["metrics"] = metrics
        report["status"] = "ok"
    else:
        report["status"] = "predicted"
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def run_pipeline_run(
    run: PipelineRun,
    *,
    do_train: bool,
    do_predict: bool,
    do_eval: bool,
    skip_missing: bool,
    prepare_only: bool = False,
) -> dict[str, Any]:
    pipe = run.pipeline
    yaml_train = bool((pipe.get("train") or {}).get("enabled", True))
    yaml_eval = bool((pipe.get("eval") or {}).get("enabled", True))
    plots = bool((pipe.get("eval") or {}).get("plots", True))
    predict_splits = list((pipe.get("predict") or {}).get("splits") or ["cal", "val"])
    out_dir = run_out_dir(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pipeline.json").write_text(
        json.dumps(run.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    expert_cfgs = [expert_train_cfg(run, spec) for spec in run.experts]
    if prepare_only:
        for cfg in expert_cfgs:
            get_trainer(cfg["backend"]).prepare(cfg)
        report = {
            "job": run.job_meta(),
            "status": "prepared",
            "output": str(out_dir),
            "n_experts": len(expert_cfgs),
        }
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return report

    run_dirs = []
    cal_dets: list[Detection] = []
    val_dets: list[Detection] = []
    cal_split = run.dataset.get("cal_split", "cal")
    val_split = run.dataset.get("val_split", "val")
    for cfg in expert_cfgs:
        try:
            run_dir = _train_predict(
                cfg,
                do_train=do_train and yaml_train,
                do_predict=do_predict,
                splits=predict_splits,
            )
            run_dirs.append(run_dir)
            cal_dets.extend(_load_split_preds(cfg, cal_split))
            val_dets.extend(_load_split_preds(cfg, val_split))
        except FileNotFoundError as exc:
            if skip_missing:
                print(f"skip missing {cfg['name']}: {exc}")
                continue
            raise
    report: dict[str, Any] = {
        "job": run.job_meta(),
        "n_experts_loaded": len({d.expert_id for d in val_dets}),
        "cost": cost_summary(run_dirs, n_experts=len(expert_cfgs)),
        "output": str(out_dir),
    }
    if not val_dets:
        report["status"] = "skipped_no_predictions"
        (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    if do_eval and yaml_eval:
        coco_cal = load_coco(resolve_split_json(expert_cfgs[0], cal_split))
        coco_val = load_coco(resolve_split_json(expert_cfgs[0], val_split))
        fused, extras = _combine(pipe, cal_dets, val_dets, coco_cal, coco_val)
        dump_detections(fused, out_dir / "predictions_val.json")
        metrics = _evaluate(fused, val_dets, coco_val, out_dir, plots=plots)
        report.update(extras)
        report["metrics"] = metrics
        report["status"] = "ok"
    else:
        report["status"] = "predicted"
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def write_rollup(reports: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in reports:
        job = rep.get("job", {})
        metrics = rep.get("metrics") or {}
        rows.append(
            {
                "name": job.get("name"),
                "family": job.get("family"),
                "pct": job.get("pct"),
                "dataset": job.get("dataset_id") or Path(str(job.get("dataset_yaml", ""))).stem,
                "expert_set": job.get("expert_set"),
                "moe": job.get("moe"),
                "fusion": job.get("fusion"),
                "status": rep.get("status"),
                "AP": metrics.get("AP", metrics.get("ap50_simple")),
                "AR": metrics.get("AR", metrics.get("recall")),
                "ece": metrics.get("ece"),
            }
        )
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bias-targeted experiment matrix.")
    parser.add_argument("--matrix", "-m", required=True, type=Path)
    parser.add_argument("--train", action="store_true", help="Train experts that are missing weights")
    parser.add_argument("--predict", action="store_true", help="Dump cal/val predictions")
    parser.add_argument("--eval", dest="do_eval", action="store_true", help="Fit combiners and evaluate")
    parser.add_argument("--all", action="store_true", help="train + predict + eval")
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument("--no-skip-missing", action="store_false", dest="skip_missing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    do_train = args.train or args.all
    do_predict = args.predict or args.all
    do_eval = args.do_eval or args.all
    if not (do_train or do_predict or do_eval):
        do_eval = True
    matrix = load_matrix(args.matrix)
    jobs = expand_jobs(matrix)
    print(f"Expanded {len(jobs)} jobs from {args.matrix}")
    reports = []
    for job in jobs:
        print(json.dumps({k: job[k] for k in ("family", "pct", "fusion", "moe", "expert_set")}, default=str))
        reports.append(
            run_job(
                job,
                do_train=do_train,
                do_predict=do_predict,
                do_eval=do_eval,
                skip_missing=args.skip_missing,
            )
        )
    rollup = ROOT / "outputs" / "experiments" / Path(args.matrix).stem / "rollup.csv"
    write_rollup(reports, rollup)
    print(f"Wrote {rollup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
