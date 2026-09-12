"""Per-expert confidence calibrators fitted on the clean cal split (numpy-only)."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from src.eval.calibration import ece, nll


def _clip(scores: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    return np.clip(arr, eps, 1.0 - eps)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


class Calibrator(ABC):
    name: str

    @abstractmethod
    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "Calibrator":
        ...

    @abstractmethod
    def transform(self, scores: Sequence[float]) -> list[float]:
        ...

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


class IdentityCalibrator(Calibrator):
    name = "identity"

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "IdentityCalibrator":
        return self

    def transform(self, scores: Sequence[float]) -> list[float]:
        return [float(min(max(s, 0.0), 1.0)) for s in scores]


def _pava(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pool Adjacent Violators for non-decreasing isotonic regression."""
    n = len(y)
    if w is None:
        w = np.ones(n)
    y = y.astype(np.float64).copy()
    w = w.astype(np.float64).copy()
    blocks = [[i] for i in range(n)]
    values = list(y)
    weights = list(w)
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1] + 1e-15:
            i += 1
            continue
        new_w = weights[i] + weights[i + 1]
        new_v = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / new_w
        values[i] = new_v
        weights[i] = new_w
        blocks[i] = blocks[i] + blocks[i + 1]
        del values[i + 1]
        del weights[i + 1]
        del blocks[i + 1]
        i = max(i - 1, 0)
    out = np.empty(n)
    for val, idx in zip(values, blocks, strict=True):
        out[idx] = val
    return np.clip(out, 0.0, 1.0)


class IsotonicCalibrator(Calibrator):
    name = "ir"

    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        x = _clip(scores)
        y = np.asarray(labels, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        self._x = x[order]
        self._y = _pava(y[order])
        return self

    def transform(self, scores: Sequence[float]) -> list[float]:
        if self._x is None or self._y is None or len(self._x) == 0:
            return [float(s) for s in scores]
        return np.interp(_clip(scores), self._x, self._y).tolist()


class TemperatureCalibrator(Calibrator):
    name = "ts"

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "TemperatureCalibrator":
        s = _clip(scores)
        y = np.asarray(labels, dtype=np.int32)
        best_t, best = 1.0, float("inf")
        for t in np.geomspace(0.05, 20.0, 40):
            cal = _temperature(s, float(t))
            loss = nll(cal, y)
            if loss < best:
                best_t, best = float(t), loss
        self.temperature = best_t
        return self

    def transform(self, scores: Sequence[float]) -> list[float]:
        return _temperature(_clip(scores), self.temperature).tolist()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "temperature": self.temperature}


class PlattCalibrator(Calibrator):
    name = "platt"

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "PlattCalibrator":
        s = _clip(scores)
        y = np.asarray(labels, dtype=np.float64)
        logit = np.log(s / (1.0 - s))
        a, b = 1.0, 0.0
        lr = 0.1
        for _ in range(200):
            p = _sigmoid(a * logit + b)
            err = p - y
            a -= lr * float(np.mean(err * logit))
            b -= lr * float(np.mean(err))
        self.a, self.b = float(a), float(b)
        return self

    def transform(self, scores: Sequence[float]) -> list[float]:
        s = _clip(scores)
        logit = np.log(s / (1.0 - s))
        return _sigmoid(self.a * logit + self.b).tolist()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "a": self.a, "b": self.b}


class LinearCalibrator(Calibrator):
    name = "lr"

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "LinearCalibrator":
        s = _clip(scores)
        y = np.asarray(labels, dtype=np.float64)
        xm, ym = float(s.mean()), float(y.mean())
        var = float(((s - xm) ** 2).sum())
        self.a = float(((s - xm) * (y - ym)).sum() / var) if var > 1e-12 else 1.0
        self.b = ym - self.a * xm
        return self

    def transform(self, scores: Sequence[float]) -> list[float]:
        return np.clip(self.a * _clip(scores) + self.b, 0.0, 1.0).tolist()


class BetaCalibrator(Calibrator):
    name = "beta"

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 1.0

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "BetaCalibrator":
        s = _clip(scores)
        y = np.asarray(labels, dtype=np.int32)
        best = (1.0, 1.0, float("inf"))
        for a in np.geomspace(0.2, 5.0, 16):
            for b in np.geomspace(0.2, 5.0, 16):
                cal = _beta(s, float(a), float(b))
                loss = nll(cal, y)
                if loss < best[2]:
                    best = (float(a), float(b), loss)
        self.a, self.b = best[0], best[1]
        return self

    def transform(self, scores: Sequence[float]) -> list[float]:
        return _beta(_clip(scores), self.a, self.b).tolist()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "a": self.a, "b": self.b}


class DirichletCalibrator(Calibrator):
    name = "dirichlet"

    def __init__(self) -> None:
        self.W = np.eye(2)
        self.bias = np.zeros(2)

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "DirichletCalibrator":
        s = _clip(scores)
        y = np.asarray(labels, dtype=np.int32)
        z = np.stack([np.log(s), np.log(1.0 - s)], axis=1)
        best_w, best = np.eye(2), float("inf")
        for scale in np.geomspace(0.25, 4.0, 12):
            W = np.eye(2) * float(scale)
            p = _softmax(z @ W.T)[:, 1]
            loss = nll(p, y)
            if loss < best:
                best_w, best = W, loss
        self.W = best_w
        return self

    def transform(self, scores: Sequence[float]) -> list[float]:
        s = _clip(scores)
        z = np.stack([np.log(s), np.log(1.0 - s)], axis=1)
        p = _softmax(z @ self.W.T + self.bias)[:, 1]
        return p.tolist()


def _temperature(scores: np.ndarray, t: float) -> np.ndarray:
    logit = np.log(scores / (1.0 - scores)) / t
    return _sigmoid(logit)


def _beta(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    num = np.power(scores, a)
    den = num + np.power(1.0 - scores, b)
    return num / np.clip(den, 1e-12, None)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -30, 30))
    return e / e.sum(axis=1, keepdims=True)


CALIBRATORS = {
    "ir": IsotonicCalibrator,
    "ts": TemperatureCalibrator,
    "platt": PlattCalibrator,
    "lr": LinearCalibrator,
    "beta": BetaCalibrator,
    "dirichlet": DirichletCalibrator,
    "identity": IdentityCalibrator,
}


def make_calibrator(name: str) -> Calibrator:
    key = name.lower()
    if key not in CALIBRATORS:
        raise KeyError(f"Unknown calibrator '{name}'. Known: {sorted(CALIBRATORS)}")
    return CALIBRATORS[key]()


def bakeoff(
    scores: Sequence[float],
    labels: Sequence[int],
    methods: Sequence[str] | None = None,
) -> tuple[str, Calibrator, dict[str, float]]:
    """Select the calibrator with lowest ECE on cal. Never uses val AP."""
    methods = list(methods or ("ir", "ts", "platt", "lr", "beta", "dirichlet"))
    best_name = "identity"
    best_model: Calibrator = IdentityCalibrator()
    best_ece = math.inf
    table: dict[str, float] = {}
    if len(scores) < 2 or len(set(int(x) for x in labels)) < 2:
        return best_name, best_model.fit(scores, labels), {"identity": 0.0}
    for name in methods:
        model = make_calibrator(name)
        try:
            model.fit(scores, labels)
            cal = model.transform(scores)
            value = ece(cal, labels)
        except Exception:
            continue
        table[name] = float(value)
        if value < best_ece:
            best_ece = value
            best_name = name
            best_model = model
    table["chosen"] = best_ece if best_ece < math.inf else 0.0
    return best_name, best_model, table
