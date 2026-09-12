"""Evaluation helpers for detection, calibration, and bias diagnostics."""

from src.eval.bias import coverage_by_size, fused_vs_expert_iou, pairwise_agreement
from src.eval.calibration import ece, reliability_diagram
from src.eval.cost import cost_summary
from src.eval.detection import summarize_detections
from src.eval.matching import correctness_arrays, match_detections

__all__ = [
    "correctness_arrays",
    "cost_summary",
    "coverage_by_size",
    "ece",
    "fused_vs_expert_iou",
    "match_detections",
    "pairwise_agreement",
    "reliability_diagram",
    "summarize_detections",
]
