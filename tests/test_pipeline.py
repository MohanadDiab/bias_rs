"""Config schema, sweep expansion, and fuse knobs from pipeline YAML."""
from __future__ import annotations

import pytest

from src.experiments.pipeline import expand_sweep, fuse_kwargs, load_and_expand
from src.experiments.schema import validate_pipeline
from src.training.config import ROOT, deep_merge


def _recipe(**overrides) -> dict:
    cfg = {
        "name": "tiny",
        "seed": 42,
        "dataset": {
            "root": "datasets/hit_uav",
            "ann_dir": "annotations",
            "train_split": "train",
            "val_split": "val",
            "cal_split": "test",
        },
        "noise": {"family": "L", "ratio": 10},
        "experts": [
            {
                "backend": "ultralytics",
                "expert_id": "yolo26n",
                "model": {"name": "yolo26n.pt"},
                "train": {"imgsz": 640, "batch": 8, "epochs": 1},
            }
        ],
        "pipeline": {
            "train": {"enabled": True},
            "predict": {"splits": ["cal", "val"], "conf": 0.3},
            "prune": {"method": "nms", "iou": 0.65, "conf": 0.3},
            "calibration": {"enabled": True, "methods": ["ir", "ts"]},
            "gate": {"enabled": False, "type": "softmax"},
            "combine": {"method": "wbf", "iou": 0.55},
            "moe": "simple",
            "eval": {"enabled": True, "plots": True},
        },
        "train": {"imgsz": 640, "batch": 8, "epochs": 1},
        "output": {"root": "outputs/experiments"},
    }
    return deep_merge(cfg, overrides)


def test_rejects_unknown_combine_method():
    cfg = _recipe()
    cfg["pipeline"]["combine"]["method"] = "not_a_real_fusion"
    with pytest.raises(ValueError, match="combine.method"):
        validate_pipeline(cfg)


def test_rejects_missing_dataset_root():
    cfg = _recipe()
    del cfg["dataset"]["root"]
    with pytest.raises(ValueError, match="dataset.root"):
        validate_pipeline(cfg)


def test_sweep_two_ratios_expands_to_two_runs():
    cfg = _recipe()
    cfg["sweep"] = {"noise_ratios": [0, 10]}
    runs = expand_sweep(cfg)
    assert len(runs) == 2
    assert {run.noise["ratio"] for run in runs} == {0, 10}
    assert all(run.pipeline["combine"]["method"] == "wbf" for run in runs)


def test_fuse_knobs_read_from_config():
    cfg = _recipe()
    cfg["pipeline"]["prune"] = {"method": "soft_nms", "iou": 0.42, "conf": 0.11}
    cfg["pipeline"]["combine"] = {"method": "wbf", "iou": 0.33}
    run = expand_sweep(cfg)[0]
    kwargs = fuse_kwargs(run)
    assert kwargs["nms_iou"] == 0.42
    assert kwargs["wbf_iou"] == 0.33
    assert kwargs["conf_thr"] == 0.11
    assert kwargs["prune_method"] == "soft_nms"
    assert kwargs["join_iou"] == 0.33


def test_l_jitter_recipe_covers_datasets_and_fusions():
    runs = load_and_expand(ROOT / "configs/pipelines/l_jitter.yaml")
    datasets = {run.dataset_id for run in runs}
    assert len(datasets) == 7
    fusions = {(run.pipeline.get("combine") or {}).get("method") for run in runs}
    assert fusions == {"nms", "soft_nms", "wbf"}
    assert any(run.noise["ratio"] == 50 for run in runs)
    assert any(run.noise["ratio"] == 0 for run in runs)
