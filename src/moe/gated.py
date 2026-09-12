"""Post-hoc image-level softmax gate over experts, fitted on the cal split."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.data.schema import Detection, group_by_expert, group_by_image
from src.eval.matching import match_detections
from src.fusion import fuse


FEATURE_DIM = 3  # count, mean score, max score per expert


def expert_features(dets: list[Detection], experts: list[str]) -> np.ndarray:
    by_exp = group_by_expert(dets)
    feats = []
    for expert in experts:
        group = by_exp.get(expert, [])
        scores = [d.score for d in group]
        count = float(len(group))
        mean = float(np.mean(scores)) if scores else 0.0
        mx = float(np.max(scores)) if scores else 0.0
        feats.extend([count, mean, mx])
    return np.asarray(feats, dtype=np.float64)


def _best_expert_for_image(
    dets: list[Detection],
    coco_gt: dict,
    experts: list[str],
) -> int:
    labeled = match_detections(dets, coco_gt)
    hits: dict[str, int] = defaultdict(int)
    for det, ok in labeled:
        if ok:
            hits[det.expert_id] += 1
    if not hits:
        scores = {e: sum(d.score for d in dets if d.expert_id == e) for e in experts}
        return int(np.argmax([scores.get(e, 0.0) for e in experts]))
    return experts.index(max(experts, key=lambda e: hits.get(e, 0)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(z, -30, 30))
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


class SoftmaxGate:
    def __init__(self) -> None:
        self.experts: list[str] = []
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self._uniform: np.ndarray | None = None

    def fit(self, dets: list[Detection], coco_gt: dict, experts: list[str] | None = None) -> "SoftmaxGate":
        self.experts = experts or sorted(group_by_expert(dets))
        k = len(self.experts)
        self._uniform = np.ones(k, dtype=np.float64) / max(k, 1)
        xs = []
        ys = []
        for group in group_by_image(dets).values():
            xs.append(expert_features(group, self.experts))
            ys.append(_best_expert_for_image(group, coco_gt, self.experts))
        if not xs:
            return self
        x = np.vstack(xs)
        y = np.asarray(ys, dtype=np.int64)
        if len(np.unique(y)) < 2:
            return self
        feat_dim = x.shape[1]
        w = np.zeros((k, feat_dim), dtype=np.float64)
        b = np.zeros(k, dtype=np.float64)
        lr = 0.05
        for _ in range(150):
            logits = x @ w.T + b
            p = _softmax(logits)
            onehot = np.zeros_like(p)
            onehot[np.arange(len(y)), y] = 1.0
            grad = (p - onehot) / max(len(y), 1)
            w -= lr * (grad.T @ x)
            b -= lr * grad.sum(axis=0)
        self.W, self.b = w, b
        return self

    def weights_for(self, dets: list[Detection]) -> dict[str, float]:
        k = len(self.experts)
        if self.W is None or self.b is None:
            w = self._uniform if self._uniform is not None else np.ones(k) / max(k, 1)
            return {e: float(w[i]) for i, e in enumerate(self.experts)}
        x = expert_features(dets, self.experts).reshape(1, -1)
        p = _softmax(x @ self.W.T + self.b)[0]
        p = p / p.sum()
        return {e: float(p[i]) for i, e in enumerate(self.experts)}


def apply_gate(dets: list[Detection], gate: SoftmaxGate) -> list[Detection]:
    out: list[Detection] = []
    for group in group_by_image(dets).values():
        weights = gate.weights_for(group)
        for det in group:
            w = weights.get(det.expert_id, 1.0)
            out.append(
                Detection(
                    image_id=det.image_id,
                    category_id=det.category_id,
                    bbox=list(det.bbox),
                    score=float(det.score) * w,
                    expert_id=det.expert_id,
                )
            )
    return out


def gated_moe(
    dets: list[Detection],
    gate: SoftmaxGate,
    fusion: str = "wbf",
    **fuse_kw: Any,
) -> list[Detection]:
    scaled = apply_gate(dets, gate)
    fused = fuse(scaled, method=fusion, **fuse_kw)
    return [
        Detection(d.image_id, d.category_id, list(d.bbox), d.score, expert_id=f"gated_{fusion}")
        for d in fused
    ]
