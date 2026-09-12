"""Load, merge, and expand self-contained pipeline YAMLs."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.experiments.schema import (
    PipelineRun,
    SWEEP_AXES,
    noise_family_pct,
    noise_tag,
    validate_pipeline,
)
from src.training.config import ROOT, deep_merge, load_yaml, resolve_path, validate_config

DEFAULTS_PATH = ROOT / "configs" / "pipelines" / "defaults.yaml"
TRAINING_DEFAULTS_PATH = ROOT / "configs" / "training" / "defaults.yaml"


def load_pipeline(
    path: str | Path,
    *,
    defaults_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge pipeline defaults with the recipe file. Does not expand ``sweep:``."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    else:
        config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    defaults = Path(defaults_path).resolve() if defaults_path else DEFAULTS_PATH
    base: dict[str, Any] = load_yaml(defaults) if defaults.exists() else {}
    cfg = deep_merge(base, load_yaml(config_path))
    cfg["_config_path"] = str(config_path)
    if not cfg.get("name") or cfg.get("name") == "pipeline":
        cfg["name"] = config_path.stem
    validate_pipeline(cfg, complete=False, loc=str(config_path))
    return cfg


def fuse_kwargs(run: PipelineRun | dict[str, Any]) -> dict[str, Any]:
    """Thresholds forwarded to ``fuse`` / MoE wrappers."""
    pipe = run.pipeline if isinstance(run, PipelineRun) else (run.get("pipeline") or {})
    prune = pipe.get("prune") or {}
    combine = pipe.get("combine") or {}
    predict = pipe.get("predict") or {}
    conf = prune.get("conf", predict.get("conf", 0.3))
    return {
        "prune": True,
        "conf_thr": float(conf),
        "nms_iou": float(prune.get("iou", 0.65)),
        "wbf_iou": float(combine.get("iou", 0.55)),
        "join_iou": float(combine.get("iou", prune.get("iou", 0.65))),
        "prune_method": str(prune.get("method", "nms")).lower().replace("-", "_"),
        "sigma": float(prune.get("sigma", 0.5)),
    }


def resolve_expert(spec: dict[str, Any]) -> dict[str, Any]:
    overlay: dict[str, Any] = {}
    config_path = spec.get("config")
    if config_path:
        path = Path(str(config_path))
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Expert config not found: {path}")
        overlay = load_yaml(path)
        overlay["_config"] = str(path.resolve())
    overlay = deep_merge(overlay, {k: v for k, v in spec.items() if k != "config"})
    if "backend" not in overlay:
        raise ValueError("Expert is missing backend (set backend or config:)")
    model = overlay.get("model") or {}
    if not isinstance(model, dict) or "name" not in model:
        raise ValueError("Expert is missing model.name")
    overlay["expert_id"] = str(
        overlay.get("expert_id") or Path(str(model["name"])).stem
    )
    return overlay


def apply_dataset_ref(run: dict[str, Any], spec: Any) -> dict[str, Any]:
    """Merge a training YAML path or inline dataset mapping into a recipe copy."""
    out = deepcopy(run)
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Dataset config not found: {path}")
        loaded = load_yaml(path)
        out["dataset"] = deep_merge(out.get("dataset") or {}, loaded.get("dataset") or {})
        if loaded.get("train"):
            out["train"] = deep_merge(out.get("train") or {}, loaded["train"])
        out["_dataset_yaml"] = str(path.resolve())
        out["_dataset_id"] = path.stem
        return out
    if not isinstance(spec, dict):
        raise ValueError(f"Bad sweep.datasets entry: {spec!r}")
    if "config" in spec:
        out = apply_dataset_ref(out, spec["config"])
        rest = {k: v for k, v in spec.items() if k != "config"}
        if rest.get("dataset"):
            out["dataset"] = deep_merge(out.get("dataset") or {}, rest["dataset"])
        elif "root" in rest:
            out["dataset"] = deep_merge(out.get("dataset") or {}, rest)
        if rest.get("train"):
            out["train"] = deep_merge(out.get("train") or {}, rest["train"])
        if rest.get("id"):
            out["_dataset_id"] = str(rest["id"])
        return out
    if "root" in spec:
        out["dataset"] = deep_merge(out.get("dataset") or {}, spec)
        out["_dataset_id"] = Path(str(spec["root"])).name
        return out
    out["dataset"] = deep_merge(out.get("dataset") or {}, spec.get("dataset") or {})
    if spec.get("train"):
        out["train"] = deep_merge(out.get("train") or {}, spec["train"])
    out["_dataset_id"] = str(
        spec.get("id")
        or Path(str((out.get("dataset") or {}).get("root", "dataset"))).name
    )
    return out


def _axis_values(sweep: dict[str, Any], key: str) -> list[Any]:
    if key not in sweep or sweep[key] is None:
        return [None]
    values = sweep[key]
    if not isinstance(values, list):
        raise ValueError(f"sweep.{key} must be a list")
    return list(values)


def _materialize(run: dict[str, Any]) -> PipelineRun:
    validate_pipeline(run, complete=True)
    experts = [resolve_expert(spec) for spec in run["experts"]]
    dataset = dict(run["dataset"])
    dataset_id = str(run.get("_dataset_id") or Path(str(dataset.get("root", "dataset"))).name)
    dataset_yaml = run.get("_dataset_yaml")
    source = run.get("_config_path")
    extras = {k: v for k, v in run.items() if k.startswith("_")}
    return PipelineRun(
        name=str(run["name"]),
        seed=int(run.get("seed") or 42),
        dataset=dataset,
        noise={
            "family": str((run.get("noise") or {}).get("family") or "clean"),
            "ratio": int((run.get("noise") or {}).get("ratio") or 0),
        },
        experts=experts,
        pipeline=dict(run.get("pipeline") or {}),
        train=dict(run.get("train") or {}),
        output=dict(run.get("output") or {"root": "outputs/experiments"}),
        source=str(source) if source else None,
        dataset_id=dataset_id,
        dataset_yaml=str(dataset_yaml) if dataset_yaml else None,
        extras=extras,
    )


def expand_sweep(cfg: dict[str, Any]) -> list[PipelineRun]:
    """Cartesian product over listed ``sweep:`` axes; other knobs stay fixed."""
    sweep = dict(cfg.get("sweep") or {})
    unknown = set(sweep) - SWEEP_AXES
    if unknown:
        raise ValueError(f"Unknown sweep axes {sorted(unknown)}. Allowed: {sorted(SWEEP_AXES)}")
    base = {k: deepcopy(v) for k, v in cfg.items() if k != "sweep"}
    runs: list[PipelineRun] = []
    for dataset in _axis_values(sweep, "datasets"):
        for ratio in _axis_values(sweep, "noise_ratios"):
            for combine in _axis_values(sweep, "combine_methods"):
                for moe in _axis_values(sweep, "moe"):
                    item = deepcopy(base)
                    if dataset is not None:
                        item = apply_dataset_ref(item, dataset)
                    if ratio is not None:
                        item.setdefault("noise", {})["ratio"] = int(ratio)
                    if combine is not None:
                        item.setdefault("pipeline", {}).setdefault("combine", {})["method"] = str(
                            combine
                        )
                    if moe is not None:
                        item.setdefault("pipeline", {})["moe"] = str(moe)
                    runs.append(_materialize(item))
    return runs


def load_and_expand(
    path: str | Path,
    *,
    defaults_path: str | Path | None = None,
) -> list[PipelineRun]:
    return expand_sweep(load_pipeline(path, defaults_path=defaults_path))


def expert_train_cfg(run: PipelineRun, expert: dict[str, Any]) -> dict[str, Any]:
    """Build the training config expected by ``get_trainer`` / ``validate_config``."""
    family, pct = noise_family_pct(run.noise)
    training_defaults = load_yaml(TRAINING_DEFAULTS_PATH) if TRAINING_DEFAULTS_PATH.exists() else {}
    train = deep_merge(training_defaults.get("train") or {}, run.train)
    train = deep_merge(train, expert.get("train") or {})
    ds_id = run.dataset_id or Path(str(run.dataset.get("root", "dataset"))).name
    tag = noise_tag(run.noise)
    predict = run.pipeline.get("predict") or {}
    output_root = run.output.get("root") or train.get("project") or "outputs/training"
    cfg: dict[str, Any] = {
        "name": f"{ds_id}__{expert['expert_id']}__{tag}",
        "backend": expert["backend"],
        "expert_id": expert["expert_id"],
        "dataset": {**run.dataset, "noise": {"family": family, "ratio": pct} if pct else {}},
        "model": dict(expert.get("model") or {}),
        "train": train,
        "predict": {"conf": float(predict.get("conf", 0.3))},
        "output": {"root": str(output_root)},
        "seed": run.seed,
    }
    cfg["dataset"]["root"] = str(resolve_path(cfg["dataset"]["root"]))
    cfg["train"]["project"] = str(resolve_path(cfg["train"].get("project", cfg["output"]["root"])))
    cfg["output"]["root"] = str(resolve_path(cfg["output"]["root"]))
    validate_config(cfg)
    return cfg


def run_out_dir(run: PipelineRun) -> Path:
    root = Path(run.output.get("root") or "outputs/experiments")
    if not root.is_absolute():
        root = ROOT / root
    combine = str((run.pipeline.get("combine") or {}).get("method") or "wbf")
    moe = str(run.pipeline.get("moe") or "simple")
    return root / run.name / run.dataset_id / noise_tag(run.noise) / f"{moe}__{combine}"
