"""Filesystem helpers for dataset preparation."""
from __future__ import annotations

import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve())
        return
    except OSError:
        pass
    try:
        dst.hardlink_to(src)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)
