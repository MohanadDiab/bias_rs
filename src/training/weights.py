"""Checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_weights(cfg: dict[str, Any]) -> str:
    explicit = cfg.get("model", {}).get("weights")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return str(path.resolve())
        return str(explicit)
    project = Path(cfg["train"].get("project", cfg["output"]["root"]))
    run = project / cfg["name"]
    for cand in (
        run / "weights" / "best.pt",
        run / "best.pt",
        run / "checkpoint.pth",
        run / "best.pth",
    ):
        if cand.exists():
            return str(cand.resolve())
    return str(cfg["model"]["name"])


def expert_id_from_cfg(cfg: dict[str, Any]) -> str:
    if cfg.get("expert_id"):
        return str(cfg["expert_id"])
    name = cfg.get("model", {}).get("name", "expert")
    return Path(str(name)).stem
