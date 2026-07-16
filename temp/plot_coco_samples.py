"""Plot one annotated COCO sample per processed dataset into temp/vis/."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = ROOT / "datasets"
DEFAULT_DATASETS = [
    "dota_v1",
    "dota_v1.5",
    "ai_tod",
    "hit_uav",
    "hrsc2016_ms",
    "plant_detection",
]
MAX_SIDE = 1920
MIN_ANNS = 5
MAX_ANNS = 50


def load_coco(ann_path: Path) -> dict:
    with open(ann_path, encoding="utf-8") as f:
        return json.load(f)


def count_anns_per_image(coco: dict) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for ann in coco["annotations"]:
        counts[ann["image_id"]] += 1
    return counts


def pick_image(coco: dict) -> dict | None:
    """Prefer image with most anns in [MIN_ANNS, MAX_ANNS]; else max anns."""
    counts = count_anns_per_image(coco)
    if not counts:
        return None
    by_id = {im["id"]: im for im in coco["images"]}
    in_range = [(iid, n) for iid, n in counts.items() if MIN_ANNS <= n <= MAX_ANNS and iid in by_id]
    if in_range:
        iid = max(in_range, key=lambda x: x[1])[0]
        return by_id[iid]
    iid = max(counts.items(), key=lambda x: x[1])[0]
    return by_id.get(iid)


def category_colors(categories: list[dict]) -> dict[int, tuple]:
    n = max(len(categories), 1)
    cmap_name = "tab20" if n > 10 else "tab10"
    cmap = plt.get_cmap(cmap_name)
    colors = {}
    for i, cat in enumerate(categories):
        colors[cat["id"]] = cmap(i % cmap.N)
    return colors


def scale_factor(width: int, height: int, max_side: int = MAX_SIDE) -> float:
    longest = max(width, height)
    if longest <= max_side:
        return 1.0
    return max_side / longest


def anns_for_image(coco: dict, image_id: int) -> list[dict]:
    return [a for a in coco["annotations"] if a["image_id"] == image_id]


def has_polygon(ann: dict) -> bool:
    seg = ann.get("segmentation")
    return isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list) and len(seg[0]) >= 6


def plot_sample(
    dataset: str,
    split: str,
    coco: dict,
    image_info: dict,
    img_path: Path,
    out_path: Path,
) -> None:
    cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    colors = category_colors(coco["categories"])
    anns = anns_for_image(coco, image_info["id"])

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        s = scale_factor(orig_w, orig_h)
        if s < 1.0:
            new_size = (max(1, int(orig_w * s)), max(1, int(orig_h * s)))
            im = im.resize(new_size, Image.Resampling.BILINEAR)
        img_arr = im

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(img_arr)
    ax.set_axis_off()

    present_cats: set[int] = set()
    for ann in anns:
        cid = ann["category_id"]
        present_cats.add(cid)
        color = colors.get(cid, (1.0, 0.0, 0.0, 1.0))
        name = cat_by_id.get(cid, str(cid))

        if has_polygon(ann):
            flat = ann["segmentation"][0]
            xs = [flat[i] * s for i in range(0, len(flat), 2)]
            ys = [flat[i] * s for i in range(1, len(flat), 2)]
            verts = list(zip(xs, ys))
            poly = Polygon(
                verts,
                closed=True,
                fill=True,
                facecolor=(*color[:3], 0.25),
                edgecolor=color,
                linewidth=1.5,
            )
            ax.add_patch(poly)
            tx, ty = min(xs), min(ys)
        else:
            x, y, w, h = ann["bbox"]
            x, y, w, h = x * s, y * s, w * s, h * s
            rect = Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                edgecolor=color,
                linewidth=1.5,
            )
            ax.add_patch(rect)
            tx, ty = x, y

        ax.text(
            tx,
            max(ty - 2, 0),
            name,
            color="white",
            fontsize=7,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=color, edgecolor="none", alpha=0.85),
        )

    legend_handles = [
        mpatches.Patch(color=colors[cid], label=cat_by_id.get(cid, str(cid)))
        for cid in sorted(present_cats, key=lambda c: cat_by_id.get(c, ""))
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
            title="Classes",
        )

    ax.set_title(f"{dataset} / {split} / {image_info['file_name']} ({len(anns)} objects)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_dataset(dataset: str, split: str, out_dir: Path) -> Path | None:
    ann_path = DATASETS_ROOT / dataset / "annotations" / f"instances_{split}.json"
    img_dir = DATASETS_ROOT / dataset / "images" / split
    if not ann_path.exists():
        print(f"[SKIP] {dataset}/{split}: missing {ann_path}")
        return None
    coco = load_coco(ann_path)
    image_info = pick_image(coco)
    if image_info is None:
        print(f"[SKIP] {dataset}/{split}: no annotated images")
        return None
    img_path = img_dir / Path(image_info["file_name"]).name
    if not img_path.exists():
        print(f"[SKIP] {dataset}/{split}: missing image {img_path}")
        return None
    out_path = out_dir / f"{dataset}_{split}_sample.png"
    n = len(anns_for_image(coco, image_info["id"]))
    print(f"[OK] {dataset}/{split}: {image_info['file_name']} ({n} anns) -> {out_path}")
    plot_sample(dataset, split, coco, image_info, img_path, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot COCO sample visualizations")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset folder names under datasets/",
    )
    parser.add_argument("--split", default="train", help="Split name (default: train)")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "temp" / "vis",
        help="Output directory for PNGs",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for ds in args.datasets:
        path = process_dataset(ds, args.split, args.out)
        if path is not None:
            written.append(path)
    print(f"\nWrote {len(written)} figure(s) to {args.out}")


if __name__ == "__main__":
    main()
