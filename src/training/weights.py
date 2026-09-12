"""Checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Ultralytics writes weights/best.pt. RF-DETR writes checkpoint_best_total.pth.
RUN_CHECKPOINT_CANDIDATES = (
    "weights/best.pt",
    "best.pt",
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
    "checkpoint.pth",
    "best.pth",
)


def run_dir_from_cfg(cfg: dict[str, Any]) -> Path:
    project = Path(cfg["train"].get("project", cfg["output"]["root"]))
    return project / cfg["name"]


def find_run_checkpoint(cfg: dict[str, Any]) -> Path | None:
    run = run_dir_from_cfg(cfg)
    for rel in RUN_CHECKPOINT_CANDIDATES:
        cand = run / rel
        if cand.exists():
            return cand.resolve()
    return None


def resolve_weights(cfg: dict[str, Any]) -> str:
    explicit = cfg.get("model", {}).get("weights")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return str(path.resolve())
    found = find_run_checkpoint(cfg)
    if found is not None:
        return str(found)
    if explicit:
        return str(explicit)
    return str(cfg["model"]["name"])


def expert_id_from_cfg(cfg: dict[str, Any]) -> str:
    if cfg.get("expert_id"):
        return str(cfg["expert_id"])
    name = cfg.get("model", {}).get("name", "expert")
    return Path(str(name)).stem
