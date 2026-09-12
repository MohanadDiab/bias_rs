"""Load a pipeline recipe and resolve dataset / expert references."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.pipeline.defaults import DATASET_DIR, EXPERT_DIR, PIPELINE_DEFAULTS, PIPELINE_DIR
from src.training.config import ROOT, deep_merge, load_yaml


def resolve_ref(value: str | Path, *, kind: str) -> Path:
    """Resolve a short name or path to a YAML file.

    ``hit_uav`` -> ``configs/training/hit_uav.yaml``
    ``yolo26n`` -> ``configs/training/models/yolo26n.yaml``
    """
    raw = Path(str(value))
    if raw.suffix in {".yaml", ".yml"}:
        path = raw if raw.is_absolute() else ROOT / raw
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(path)
    stem = raw.stem
    if kind == "dataset":
        candidates = [
            ROOT / DATASET_DIR / f"{stem}.yaml",
            ROOT / DATASET_DIR / f"{stem}.yml",
        ]
    elif kind == "expert":
        candidates = [
            ROOT / EXPERT_DIR / f"{stem}.yaml",
            ROOT / EXPERT_DIR / f"{stem}.yml",
        ]
    elif kind == "pipeline":
        candidates = [
            ROOT / PIPELINE_DIR / f"{stem}.yaml",
            ROOT / PIPELINE_DIR / f"{stem}.yml",
            raw if raw.is_absolute() else ROOT / raw,
        ]
    else:
        raise ValueError(f"Unknown ref kind '{kind}'")
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No {kind} config for '{value}'. Looked in: {searched}")


def list_datasets() -> list[str]:
    skip = {"defaults.yaml"}
    return sorted(
        p.stem
        for p in (ROOT / DATASET_DIR).glob("*.yaml")
        if p.name not in skip
    )


def list_experts() -> list[str]:
    return sorted(p.stem for p in (ROOT / EXPERT_DIR).glob("*.yaml"))


def list_pipelines() -> list[str]:
    skip = {"defaults.yaml"}
    folder = ROOT / PIPELINE_DIR
    if not folder.exists():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml") if p.name not in skip)


def _normalize_experts(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        raw = PIPELINE_DEFAULTS["experts"]
    if not isinstance(raw, list):
        raise ValueError("pipeline.experts must be a list of names or mappings")
    specs: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            specs.append({"config": str(resolve_ref(item, kind="expert"))})
        elif isinstance(item, dict):
            cfg = dict(item)
            if "config" in cfg:
                cfg["config"] = str(resolve_ref(cfg["config"], kind="expert"))
            elif "expert_id" in cfg:
                cfg["config"] = str(resolve_ref(str(cfg["expert_id"]), kind="expert"))
            specs.append(cfg)
        else:
            raise ValueError(f"Bad expert entry: {item!r}")
    return specs


def load_pipeline(
    path: str | Path | None = None,
    *,
    dataset: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(PIPELINE_DEFAULTS)
    source = None
    if path is not None:
        raw = Path(str(path))
        source = raw.resolve() if raw.exists() else resolve_ref(str(path), kind="pipeline")
        cfg = deep_merge(cfg, load_yaml(source))
        cfg["_pipeline_path"] = str(source)
    if dataset is not None:
        cfg["dataset"] = dataset
    if overrides:
        cfg = deep_merge(cfg, overrides)
    if not cfg.get("dataset"):
        raise ValueError("Pipeline has no dataset. Set `dataset:` in the YAML or pass --dataset.")
    cfg["dataset_yaml"] = str(resolve_ref(cfg["dataset"], kind="dataset"))
    cfg["expert_specs"] = _normalize_experts(cfg.get("experts"))
    if not cfg["expert_specs"]:
        raise ValueError("Pipeline has no experts.")
    noise = cfg.setdefault("noise", {})
    family = str(noise.get("family") or "clean")
    pct = int(noise.get("ratio") or 0)
    if family.lower() in {"none", "off", ""}:
        family = "clean"
        pct = 0
    cfg["noise"] = {"family": family, "ratio": pct}
    ds_stem = Path(cfg["dataset_yaml"]).stem
    recipe = Path(cfg.get("_pipeline_path", "pipeline")).stem
    cfg["name"] = cfg.get("name") or f"{recipe}__{ds_stem}"
    return cfg


def fuse_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    pipe = cfg.get("pipeline") or {}
    suppress = pipe.get("suppress") or {}
    join = pipe.get("join") or {}
    return {
        "prune": bool(suppress.get("enabled", True)),
        "prune_method": str(suppress.get("method", "nms")),
        "conf_thr": float(suppress.get("conf_thr", 0.3)),
        "nms_iou": float(suppress.get("iou", 0.65)),
        "wbf_iou": float(join.get("iou", 0.55)),
        "sigma": float(suppress.get("sigma", 0.5)),
    }
