"""Dataset-level utilities: COCO IO, cal splits, isolated noise."""

from src.data.paths import RATIO_GRID, resolve_split_json
from src.data.schema import Detection, dump_detections, load_detections

__all__ = [
    "RATIO_GRID",
    "Detection",
    "dump_detections",
    "load_detections",
    "resolve_split_json",
]
