"""Plot helpers for ECE reliability diagrams and correct/incorrect score KDEs."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def kde_1d(values: Sequence[float], grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(values, dtype=np.float64)
    if grid is None:
        grid = np.linspace(0.0, 1.0, 64)
    if xs.size == 0:
        return grid, np.zeros_like(grid)
    std = np.std(xs) or 0.08
    bw = 1.06 * std * (xs.size ** (-0.2))
    bw = max(bw, 0.02)
    dens = np.zeros_like(grid)
    for v in xs:
        dens += np.exp(-0.5 * ((grid - v) / bw) ** 2)
    dens /= dens.sum() * (grid[1] - grid[0]) if dens.sum() else 1.0
    return grid, dens


def save_reliability_plot(bin_conf: Sequence[float], bin_acc: Sequence[float], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls="--", c="0.5")
    ax.plot(bin_conf, bin_acc, marker="o")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_score_kde(
    correct: Sequence[float],
    incorrect: Sequence[float],
    path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid, d_ok = kde_1d(correct)
    _, d_bad = kde_1d(incorrect, grid)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(grid, d_ok, label="correct")
    ax.plot(grid, d_bad, label="incorrect")
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
