"""Cost / latency summaries from run_meta.json files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_run_meta(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "run_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cost_summary(run_dirs: list[str | Path], n_experts: int) -> dict[str, Any]:
    train_s = []
    infer_ms = []
    for run in run_dirs:
        meta = load_run_meta(run)
        if "train_seconds" in meta:
            train_s.append(float(meta["train_seconds"]))
        for key, value in meta.items():
            if key.endswith("_seconds") and key.startswith("predict_"):
                n_img = meta.get(key.replace("_seconds", "_n_images"), 0) or 0
                if n_img:
                    infer_ms.append(float(value) / n_img * 1000.0)
            if key == "ms_per_image" and value is not None:
                infer_ms.append(float(value))
    return {
        "n_experts": n_experts,
        "train_seconds_sum": sum(train_s),
        "train_seconds_mean": (sum(train_s) / len(train_s)) if train_s else None,
        "ms_per_image_mean": (sum(infer_ms) / len(infer_ms)) if infer_ms else None,
    }
