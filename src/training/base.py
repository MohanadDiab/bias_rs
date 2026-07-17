"""Abstract trainer interface for pluggable backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseTrainer(ABC):
    """Backend-agnostic training lifecycle."""

    @abstractmethod
    def prepare(self, cfg: dict[str, Any]) -> Path:
        """Materialize backend-ready data; return a data config path."""

    @abstractmethod
    def train(self, cfg: dict[str, Any], data_path: Path) -> Path:
        """Run training; return the run output directory."""
