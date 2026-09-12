"""Fusion, ECE, and gated MoE unit tests."""
from __future__ import annotations

from src.calibration.methods import bakeoff
from src.data.schema import Detection
from src.eval.calibration import ece
from src.fusion import fuse
from src.fusion.wbf import wbf
from src.moe.gated import SoftmaxGate, expert_features


def _det(image_id, cat, bbox, score, expert) -> Detection:
    return Detection(image_id, cat, bbox, score, expert)


def test_wbf_averages_overlapping_boxes():
    a = _det(1, 1, [0, 0, 10, 10], 1.0, "e1")
    b = _det(1, 1, [2, 0, 10, 10], 1.0, "e2")
    fused = wbf([a, b], iou_thr=0.3)
    assert len(fused) == 1
    assert abs(fused[0].bbox[0] - 1.0) < 1e-6
    assert abs(fused[0].score - 1.0) < 1e-6


def test_nms_drops_overlap():
    a = _det(1, 1, [0, 0, 10, 10], 0.9, "e1")
    b = _det(1, 1, [1, 0, 10, 10], 0.5, "e1")
    out = fuse([a, b], method="nms", prune=False, conf_thr=0.0)
    assert len(out) == 1
    assert out[0].score == 0.9


def test_ece_perfect_is_zero():
    scores = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]
    assert ece(scores, labels, n_bins=2) < 1e-9


def test_ece_bounds():
    scores = [0.9, 0.9, 0.9, 0.9]
    labels = [0, 0, 0, 0]
    value = ece(scores, labels, n_bins=10)
    assert 0.0 <= value <= 1.0
    assert value > 0.5


def test_bakeoff_picks_by_ece_not_ap():
    scores = [0.1] * 20 + [0.9] * 20
    labels = [0] * 20 + [1] * 20
    name, phi, table = bakeoff(scores, labels)
    assert name in table or name == "identity"
    cal = phi.transform(scores)
    assert ece(cal, labels) <= ece(scores, labels) + 1e-6


def test_gate_weights_sum_to_one():
    dets = [
        _det(1, 1, [0, 0, 8, 8], 0.8, "a"),
        _det(1, 1, [20, 20, 8, 8], 0.4, "b"),
        _det(2, 1, [0, 0, 8, 8], 0.7, "a"),
        _det(2, 1, [20, 20, 8, 8], 0.9, "b"),
    ]
    coco = {
        "images": [{"id": 1, "width": 64, "height": 64, "file_name": "1.jpg"}, {"id": 2, "width": 64, "height": 64, "file_name": "2.jpg"}],
        "categories": [{"id": 1, "name": "a"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 8, 8]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [20, 20, 8, 8]},
        ],
    }
    gate = SoftmaxGate().fit(dets, coco, experts=["a", "b"])
    w = gate.weights_for(dets[:2])
    assert abs(sum(w.values()) - 1.0) < 1e-6
    feats = expert_features(dets[:2], ["a", "b"])
    assert feats.shape[0] == 6
