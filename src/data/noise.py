"""Isolated label-noise protocols for L / O / C / Mix runs.

Noise is applied only to the train split. Val, test, and cal stay clean.
Ratios are integer percents of annotations (1 means 1%).
"""
from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from src.data.boxes import clip_xywh
from src.data.coco import image_size_map, reindex_annotations

JITTER_SHIFT = 0.15
JITTER_SCALE = 0.15


def _sample_k(n: int, pct: int, rng: random.Random) -> list[int]:
    k = int(round(n * pct / 100.0))
    k = min(max(k, 0), n)
    if k == 0:
        return []
    return rng.sample(range(n), k)


def jitter_box(
    bbox: list[float],
    img_w: int,
    img_h: int,
    rng: random.Random,
    shift_frac: float = JITTER_SHIFT,
    scale_frac: float = JITTER_SCALE,
) -> list[float] | None:
    x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    cx = x + w / 2.0
    cy = y + h / 2.0
    cx += rng.uniform(-shift_frac, shift_frac) * w
    cy += rng.uniform(-shift_frac, shift_frac) * h
    w *= 1.0 + rng.uniform(-scale_frac, scale_frac)
    h *= 1.0 + rng.uniform(-scale_frac, scale_frac)
    w = max(w, 1.0)
    h = max(h, 1.0)
    return clip_xywh([cx - w / 2.0, cy - h / 2.0, w, h], img_w, img_h)


def apply_localization(
    coco: dict[str, Any],
    pct: int,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """L-runs: jitter a fraction of boxes. Class labels are unchanged."""
    out = deepcopy(coco)
    sizes = image_size_map(out)
    anns = out["annotations"]
    idxs = _sample_k(len(anns), pct, rng)
    n_changed = 0
    n_kept_original = 0
    for i in idxs:
        ann = anns[i]
        img_w, img_h = sizes.get(int(ann["image_id"]), (0, 0))
        new_box = jitter_box(ann["bbox"], img_w, img_h, rng)
        if new_box is None:
            n_kept_original += 1
            continue
        n_changed += 1
        ann["bbox"] = new_box
        ann["area"] = float(new_box[2] * new_box[3])
        ann["segmentation"] = []
    out = reindex_annotations(out)
    stats = {
        "family": "L",
        "pct": pct,
        "n_train_boxes": len(coco["annotations"]),
        "n_selected": len(idxs),
        "n_jittered": n_changed,
        "n_kept_original": n_kept_original,
        "realized_ratio": n_changed / max(len(coco["annotations"]), 1),
    }
    return out, stats


def apply_objectness(
    coco: dict[str, Any],
    pct: int,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """O-runs: omit half the budget and add the same number of spurious boxes."""
    out = deepcopy(coco)
    sizes = image_size_map(out)
    anns = out["annotations"]
    n = len(anns)
    budget = int(round(n * pct / 100.0))
    n_remove = budget // 2
    n_add = budget - n_remove
    remove_idxs = set(rng.sample(range(n), min(n_remove, n))) if n and n_remove else set()
    kept = [ann for i, ann in enumerate(anns) if i not in remove_idxs]

    cat_ids = [int(c["id"]) for c in out.get("categories", [])]
    sizes_wh = [(float(a["bbox"][2]), float(a["bbox"][3])) for a in anns if a.get("bbox") and len(a["bbox"]) >= 4]
    image_ids = [int(im["id"]) for im in out["images"]]
    img_by_id = {int(im["id"]): im for im in out["images"]}
    next_id = max((int(a["id"]) for a in anns), default=0) + 1
    added = []
    for _ in range(n_add):
        if not image_ids or not cat_ids:
            break
        image_id = rng.choice(image_ids)
        im = img_by_id[image_id]
        img_w, img_h = int(im["width"]), int(im["height"])
        if sizes_wh:
            w, h = rng.choice(sizes_wh)
        else:
            w, h = min(32.0, img_w / 4), min(32.0, img_h / 4)
        w = min(max(w, 2.0), float(img_w))
        h = min(max(h, 2.0), float(img_h))
        x = rng.uniform(0, max(img_w - w, 0.0))
        y = rng.uniform(0, max(img_h - h, 0.0))
        box = clip_xywh([x, y, w, h], img_w, img_h)
        if box is None:
            continue
        added.append(
            {
                "id": next_id,
                "image_id": image_id,
                "category_id": rng.choice(cat_ids),
                "bbox": box,
                "area": float(box[2] * box[3]),
                "iscrowd": 0,
                "segmentation": [],
                "ignore": 0,
            }
        )
        next_id += 1
    out["annotations"] = kept + added
    out = reindex_annotations(out)
    stats = {
        "family": "O",
        "pct": pct,
        "n_train_boxes": n,
        "n_removed": len(remove_idxs),
        "n_added": len(added),
        "omission_ratio": len(remove_idxs) / max(n, 1),
        "spurious_ratio": len(added) / max(n, 1),
        "realized_ratio": (len(remove_idxs) + len(added)) / max(n, 1),
    }
    return out, stats


def apply_class_flip(
    coco: dict[str, Any],
    pct: int,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """C-runs: flip class labels; boxes are unchanged."""
    out = deepcopy(coco)
    cat_ids = [int(c["id"]) for c in out.get("categories", [])]
    anns = out["annotations"]
    idxs = _sample_k(len(anns), pct, rng)
    n_flipped = 0
    for i in idxs:
        if len(cat_ids) < 2:
            break
        cur = int(anns[i]["category_id"])
        others = [c for c in cat_ids if c != cur]
        if not others:
            continue
        anns[i]["category_id"] = rng.choice(others)
        n_flipped += 1
    out = reindex_annotations(out)
    stats = {
        "family": "C",
        "pct": pct,
        "n_train_boxes": len(coco["annotations"]),
        "n_selected": len(idxs),
        "n_flipped": n_flipped,
        "realized_ratio": n_flipped / max(len(coco["annotations"]), 1),
    }
    return out, stats


def apply_mix(
    coco: dict[str, Any],
    ratios: dict[str, int],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply L, then O, then C at the given percents (documented Mix order)."""
    out = coco
    all_stats = {}
    out, all_stats["L"] = apply_localization(out, int(ratios.get("L", 0)), rng)
    out, all_stats["O"] = apply_objectness(out, int(ratios.get("O", 0)), rng)
    out, all_stats["C"] = apply_class_flip(out, int(ratios.get("C", 0)), rng)
    stats = {"family": "Mix", "order": ["L", "O", "C"], "parts": all_stats, "ratios": dict(ratios)}
    return out, stats


def apply_family(
    coco: dict[str, Any],
    family: str,
    pct: int,
    rng: random.Random,
    mix_ratios: dict[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    key = family.upper()
    if key == "L":
        return apply_localization(coco, pct, rng)
    if key == "O":
        return apply_objectness(coco, pct, rng)
    if key == "C":
        return apply_class_flip(coco, pct, rng)
    if key == "MIX":
        ratios = mix_ratios or {"L": pct, "O": pct, "C": pct}
        return apply_mix(coco, ratios, rng)
    raise ValueError(f"Unknown noise family '{family}'")


def boxes_changed(orig: dict[str, Any], noisy: dict[str, Any]) -> bool:
    orig_map = {int(a["id"]): a for a in orig["annotations"]}
    for ann in noisy["annotations"]:
        prev = orig_map.get(int(ann["id"]))
        if prev is None:
            continue
        if [float(x) for x in prev["bbox"][:4]] != [float(x) for x in ann["bbox"][:4]]:
            return True
    return False
