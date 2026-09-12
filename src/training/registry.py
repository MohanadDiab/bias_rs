"""Trainer backend registry."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from src.training.backends.rfdetr_backend import RFDetrTrainer
from src.training.backends.tiny_fpn import TinyFPNTrainer
from src.training.backends.ultralytics_backend import UltralyticsTrainer

if TYPE_CHECKING:
    from src.training.base import BaseTrainer


def _mmdet_trainer():
    from src.training.backends.mmdet_backend import MMDetTrainer, mmdet_available

    if mmdet_available():
        return MMDetTrainer()
    warnings.warn(
        "mmdet is not installed; DINO/RTMDet runs use the tiny_fpn fallback "
        "(resnet variant for dino, csp variant for rtmdet).",
        RuntimeWarning,
        stacklevel=2,
    )
    return TinyFPNTrainer()


BACKENDS: dict[str, type] = {
    "ultralytics": UltralyticsTrainer,
    "rfdetr": RFDetrTrainer,
    "tiny_fpn": TinyFPNTrainer,
}


def get_trainer(backend: str):
    key = backend.lower().strip()
    if key in {"mmdet", "dino", "rtmdet"}:
        return _mmdet_trainer()
    if key not in BACKENDS:
        known = ", ".join(sorted([*BACKENDS, "mmdet"]))
        raise KeyError(f"Unknown training backend '{backend}'. Known: {known}")
    return BACKENDS[key]()
