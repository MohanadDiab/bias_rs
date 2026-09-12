"""Config schema and experiment matrix expansion."""
from __future__ import annotations

from src.experiments.matrix import expand_jobs, load_matrix
from src.training.config import ROOT, load_config


def test_training_yaml_has_cal_split():
    cfg = load_config(ROOT / "configs/training/hit_uav.yaml")
    assert cfg["dataset"]["cal_split"] == "test"
    assert cfg["dataset"]["train_split"] == "train"
    assert cfg["dataset"]["val_split"] == "val"


def test_dota_yaml_keeps_carved_cal_split():
    cfg = load_config(ROOT / "configs/training/dota_1024_v1.yaml")
    assert cfg["dataset"]["cal_split"] == "cal"


def test_l_runs_cover_all_dataset_configs():
    matrix = load_matrix(ROOT / "configs/experiments/l_runs.yaml")
    jobs = expand_jobs(matrix)
    datasets = {job["dataset_yaml"] for job in jobs}
    assert len(datasets) == 7
    fusions = {job["fusion"] for job in jobs}
    assert fusions == {"nms", "soft_nms", "wbf"}
    assert any(job["pct"] == 50 for job in jobs)
    assert any(job["pct"] == 0 for job in jobs)
