"""Config-defined detection pipeline."""

from src.pipeline.load import load_pipeline
from src.pipeline.run import run_from_files, run_pipeline

__all__ = ["load_pipeline", "run_from_files", "run_pipeline"]
