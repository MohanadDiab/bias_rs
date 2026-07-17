"""Report train/val image size stats for each processed dataset."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"

# Versions share images, so list each image tree once
DATASETS = [
    ("dota", "dota"),
    ("ai_tod", "ai_tod"),
    ("hit_uav", "hit_uav"),
    ("hrsc2016_ms", "hrsc2016_ms"),
    ("plant_detection", "plant_detection"),
]
SPLITS = ("train", "val")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_sizes(img_dir: Path) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for path in sorted(img_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        with Image.open(path) as im:
            sizes.append(im.size)  # (width, height)
    return sizes


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    if len(set(values)) == 1:
        return {"constant": True, "value": values[0]}
    return {
        "constant": False,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
    }


def format_dim(summary: dict, unit: str = "px") -> str:
    if not summary:
        return "n/a"
    if summary["constant"]:
        return f"{summary['value']:.0f} {unit} (constant)"
    return (
        f"min={summary['min']:.0f}  max={summary['max']:.0f}  "
        f"median={summary['median']:.1f}  mean={summary['mean']:.1f}  "
        f"std={summary['std']:.1f}"
    )


def format_split(split: str, sizes: list[tuple[int, int]]) -> list[str]:
    if not sizes:
        return [f"  {split}: no images"]
    widths = [float(w) for w, _ in sizes]
    heights = [float(h) for _, h in sizes]
    areas = [float(w * h) for w, h in sizes]
    unique = sorted(set(sizes))
    lines = [f"  {split}: n={len(sizes)}"]
    if len(unique) == 1:
        w, h = unique[0]
        lines.append(f"    size: {w}x{h} (constant)")
    else:
        lines.append(f"    unique WxH: {len(unique)}")
        lines.append(f"    width:  {format_dim(summarize(widths))}")
        lines.append(f"    height: {format_dim(summarize(heights))}")
        lines.append(f"    area:   {format_dim(summarize(areas), unit='px^2')}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "temp" / "image_size_stats.txt",
        help="Text report path",
    )
    args = parser.parse_args()

    lines: list[str] = []
    for folder, label in DATASETS:
        lines.append(f"### {label}")
        for split in args.splits:
            img_dir = DATASETS_ROOT / folder / "images" / split
            if not img_dir.exists():
                lines.append(f"  {split}: missing {img_dir}")
                continue
            lines.extend(format_split(split, collect_sizes(img_dir)))
        lines.append("")

    text = "\n".join(lines)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
