"""Expected calibration error and reliability bins."""
from __future__ import annotations

import math
from typing import Sequence


def ece(scores: Sequence[float], labels: Sequence[int], n_bins: int = 15) -> float:
    if not scores:
        return 0.0
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    bins = [0.0] * n_bins
    bin_conf = [0.0] * n_bins
    bin_acc = [0.0] * n_bins
    bin_count = [0] * n_bins
    n = len(scores)
    for s, y in zip(scores, labels, strict=True):
        s = min(max(float(s), 0.0), 1.0)
        idx = min(int(s * n_bins), n_bins - 1)
        bin_count[idx] += 1
        bin_conf[idx] += s
        bin_acc[idx] += float(y)
    total = 0.0
    for i in range(n_bins):
        if bin_count[i] == 0:
            continue
        acc = bin_acc[i] / bin_count[i]
        conf = bin_conf[i] / bin_count[i]
        total += (bin_count[i] / n) * abs(acc - conf)
        bins[i] = abs(acc - conf)
    return float(total)


def reliability_diagram(
    scores: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 15,
) -> dict[str, list[float]]:
    confs = [0.0] * n_bins
    accs = [0.0] * n_bins
    counts = [0] * n_bins
    for s, y in zip(scores, labels, strict=True):
        s = min(max(float(s), 0.0), 1.0)
        idx = min(int(s * n_bins), n_bins - 1)
        counts[idx] += 1
        confs[idx] += s
        accs[idx] += float(y)
    for i in range(n_bins):
        if counts[i]:
            confs[i] /= counts[i]
            accs[i] /= counts[i]
    edges = [i / n_bins for i in range(n_bins)]
    return {"bin_left": edges, "confidence": confs, "accuracy": accs, "count": [float(c) for c in counts]}


def nll(scores: Sequence[float], labels: Sequence[int], eps: float = 1e-7) -> float:
    if not scores:
        return 0.0
    total = 0.0
    for s, y in zip(scores, labels, strict=True):
        s = min(max(float(s), eps), 1.0 - eps)
        total += -math.log(s) if y else -math.log(1.0 - s)
    return total / len(scores)
