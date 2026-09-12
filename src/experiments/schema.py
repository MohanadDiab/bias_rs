"""Pipeline YAML schema, constants, and fail-fast validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

NOISE_FAMILIES = frozenset({"clean", "L", "O", "C", "Mix"})
MOE_METHODS = frozenset({"simple", "calibrated", "gated", "learned"})
COMBINE_METHODS = frozenset({"nms", "soft_nms", "wbf", "learned"})
PRUNE_METHODS = frozenset({"nms", "soft_nms"})
CALIBRATION_METHODS = frozenset({"ir", "ts", "platt", "lr", "beta", "dirichlet", "identity"})
GATE_TYPES = frozenset({"softmax"})
SWEEP_AXES = frozenset({"datasets", "noise_ratios", "combine_methods", "moe"})
REQUIRED_DATASET = ("root", "ann_dir", "train_split", "val_split", "cal_split")


@dataclass
class PipelineRun:
    """One concrete detect-and-combine run after defaults merge and sweep expansion."""

    name: str
    seed: int
    dataset: dict[str, Any]
    noise: dict[str, Any]
    experts: list[dict[str, Any]]
    pipeline: dict[str, Any]
    train: dict[str, Any]
    output: dict[str, Any]
    source: str | None = None
    dataset_id: str = ""
    dataset_yaml: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def job_meta(self) -> dict[str, Any]:
        family, pct = noise_family_pct(self.noise)
        combine = (self.pipeline.get("combine") or {}).get("method")
        return {
            "name": self.name,
            "seed": self.seed,
            "family": family,
            "pct": pct,
            "dataset_id": self.dataset_id,
            "dataset_yaml": self.dataset_yaml,
            "moe": self.pipeline.get("moe"),
            "fusion": combine,
        }


def noise_family_pct(noise: dict[str, Any]) -> tuple[str, int]:
    family = str((noise or {}).get("family") or "clean")
    pct = int((noise or {}).get("ratio") or 0)
    if pct <= 0 or family.lower() == "clean":
        return "clean", 0
    return family, pct


def noise_tag(noise: dict[str, Any]) -> str:
    family, pct = noise_family_pct(noise)
    if family == "clean" or pct <= 0:
        return "clean"
    return f"{family}_{pct}"


def _loc(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _require_mapping(value: Any, loc: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{loc} must be a mapping")
    return value


def validate_pipeline(cfg: dict[str, Any], *, complete: bool = True, loc: str = "") -> None:
    """Fail fast on missing dataset.root, unknown combine.method, and similar errors."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Pipeline config must be a mapping{f' ({loc})' if loc else ''}")
    prefix = loc.rstrip(".") if loc else ""

    name = cfg.get("name")
    if not name or not str(name).strip():
        raise ValueError(f"Missing {_loc(prefix, 'name')}")

    noise = _require_mapping(cfg.get("noise") or {}, _loc(prefix, "noise"))
    family = str(noise.get("family") or "clean")
    if family not in NOISE_FAMILIES:
        raise ValueError(
            f"Unknown noise.family '{family}'. Known: {sorted(NOISE_FAMILIES)}"
        )
    try:
        ratio = int(noise.get("ratio") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("noise.ratio must be an integer percent") from exc
    if ratio < 0:
        raise ValueError("noise.ratio must be >= 0")

    experts = cfg.get("experts")
    if not isinstance(experts, list) or not experts:
        raise ValueError(f"{_loc(prefix, 'experts')} must be a non-empty list")
    for i, spec in enumerate(experts):
        if not isinstance(spec, dict):
            raise ValueError(f"experts[{i}] must be a mapping")

    pipe = _require_mapping(cfg.get("pipeline") or {}, _loc(prefix, "pipeline"))
    moe = str(pipe.get("moe") or "simple")
    if moe not in MOE_METHODS:
        raise ValueError(f"Unknown pipeline.moe '{moe}'. Known: {sorted(MOE_METHODS)}")

    prune = _require_mapping(pipe.get("prune") or {}, "pipeline.prune")
    prune_method = str(prune.get("method") or "nms").lower().replace("-", "_")
    if prune_method not in PRUNE_METHODS:
        raise ValueError(
            f"Unknown pipeline.prune.method '{prune_method}'. Known: {sorted(PRUNE_METHODS)}"
        )

    combine = _require_mapping(pipe.get("combine") or {}, "pipeline.combine")
    combine_method = str(combine.get("method") or "wbf").lower().replace("-", "_")
    if combine_method not in COMBINE_METHODS:
        raise ValueError(
            f"Unknown pipeline.combine.method '{combine_method}'. Known: {sorted(COMBINE_METHODS)}"
        )

    calibration = _require_mapping(pipe.get("calibration") or {}, "pipeline.calibration")
    methods = calibration.get("methods") or []
    if not isinstance(methods, list):
        raise ValueError("pipeline.calibration.methods must be a list")
    for name in methods:
        key = str(name).lower()
        if key not in CALIBRATION_METHODS:
            raise ValueError(
                f"Unknown calibration method '{name}'. Known: {sorted(CALIBRATION_METHODS)}"
            )

    gate = _require_mapping(pipe.get("gate") or {}, "pipeline.gate")
    gate_type = str(gate.get("type") or "softmax").lower()
    if gate_type not in GATE_TYPES:
        raise ValueError(f"Unknown pipeline.gate.type '{gate_type}'. Known: {sorted(GATE_TYPES)}")

    sweep = cfg.get("sweep")
    if sweep is not None:
        sweep_map = _require_mapping(sweep, "sweep")
        unknown = set(sweep_map) - SWEEP_AXES
        if unknown:
            raise ValueError(
                f"Unknown sweep axes {sorted(unknown)}. Allowed: {sorted(SWEEP_AXES)}"
            )

    has_sweep_datasets = isinstance(sweep, dict) and bool(sweep.get("datasets"))
    if complete or not has_sweep_datasets:
        dataset = _require_mapping(cfg.get("dataset"), _loc(prefix, "dataset"))
        for key in REQUIRED_DATASET:
            if key not in dataset or dataset[key] in (None, ""):
                raise ValueError(f"Missing dataset.{key}")
