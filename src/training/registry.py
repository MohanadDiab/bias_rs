"""Trainer backend registry."""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.training.backends.ultralytics_backend import UltralyticsTrainer

if TYPE_CHECKING:
    from src.training.base import BaseTrainer

BACKENDS: dict[str, type[BaseTrainer]] = {
    "ultralytics": UltralyticsTrainer,
}


def get_trainer(backend: str) -> BaseTrainer:
    key = backend.lower().strip()
    if key not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise KeyError(f"Unknown training backend '{backend}'. Known: {known}")
    return BACKENDS[key]()
