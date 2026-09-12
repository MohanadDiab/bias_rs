"""Tiny-object FPN detector used when MMDetection is unavailable.

Two variants keep the asymmetric set at four experts:
- ``resnet`` (DINO fallback): torchvision ResNet-18 + FPN
- ``csp`` (RTMDet fallback): lightweight CSP-like CNN + FPN
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.boxes import xywh_to_xyxy
from src.data.coco import anns_by_image, load_coco
from src.data.paths import resolve_split_json
from src.data.schema import Detection, dump_detections
from src.fusion.nms import nms
from src.training.base import BaseTrainer
from src.training.weights import expert_id_from_cfg, resolve_weights

STRIDES = (8, 16, 32, 64)


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def _build_modules():
    torch, nn, F = _torch()

    class ConvBNActMod(nn.Module):
        def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=k // 2, bias=False)
            self.bn = nn.BatchNorm2d(c_out)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class CspBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(ConvBNActMod(3, 32, 3, 2), ConvBNActMod(32, 64, 3, 2))
            self.c3 = self._stage(64, 128, 2)
            self.c4 = self._stage(128, 256, 2)
            self.c5 = self._stage(256, 256, 2)

        def _stage(self, c_in, c_out, stride):
            return nn.Sequential(
                ConvBNActMod(c_in, c_out, 3, stride),
                ConvBNActMod(c_out, c_out, 3, 1),
            )

        def forward(self, x):
            x = self.stem(x)
            c3 = self.c3(x)
            c4 = self.c4(c3)
            c5 = self.c5(c4)
            return [c3, c4, c5]

    class ResNetBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            from torchvision.models import resnet18

            net = resnet18(weights=None)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            c3 = self.layer2(x)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            return [c3, c4, c5]

    class FPN(nn.Module):
        def __init__(self, in_channels: list[int], out_ch: int = 128):
            super().__init__()
            self.laterals = nn.ModuleList(nn.Conv2d(c, out_ch, 1) for c in in_channels)
            self.smooth = nn.ModuleList(nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in in_channels)
            self.p6 = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

        def forward(self, feats):
            laterals = [lat(f) for lat, f in zip(self.laterals, feats, strict=True)]
            for i in range(len(laterals) - 1, 0, -1):
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            outs = [sm(f) for sm, f in zip(self.smooth, laterals, strict=True)]
            outs.append(self.p6(outs[-1]))
            return outs

    class Head(nn.Module):
        def __init__(self, in_ch: int, num_classes: int):
            super().__init__()
            self.cls = nn.Sequential(ConvBNActMod(in_ch, in_ch), nn.Conv2d(in_ch, num_classes, 3, padding=1))
            self.box = nn.Sequential(ConvBNActMod(in_ch, in_ch), nn.Conv2d(in_ch, 4, 3, padding=1))
            self.ctr = nn.Sequential(ConvBNActMod(in_ch, in_ch), nn.Conv2d(in_ch, 1, 3, padding=1))

        def forward(self, feats):
            return [(self.cls(f), self.box(f), self.ctr(f)) for f in feats]

    class TinyFPN(nn.Module):
        def __init__(self, num_classes: int, variant: str = "resnet"):
            super().__init__()
            self.variant = variant
            if variant == "csp":
                self.backbone = CspBackbone()
                in_ch = [128, 256, 256]
            else:
                self.backbone = ResNetBackbone()
                in_ch = [128, 256, 512]
            self.fpn = FPN(in_ch, 128)
            self.head = Head(128, num_classes)
            self.num_classes = num_classes

        def forward(self, x):
            return self.head(self.fpn(self.backbone(x)))

    return TinyFPN


def decode_batch(outputs, score_thr: float = 0.3) -> list[list[tuple[list[float], float, int]]]:
    torch, _, F = _torch()
    bsz = outputs[0][0].shape[0]
    per_image: list[list[tuple[list[float], float, int]]] = [[] for _ in range(bsz)]
    for level, (cls_map, box_map, ctr_map) in enumerate(outputs):
        stride = STRIDES[min(level, len(STRIDES) - 1)]
        cls_prob = cls_map.sigmoid()
        ctr = ctr_map.sigmoid()
        scores, labels = cls_prob.max(dim=1)
        scores = scores * ctr.squeeze(1)
        _, _, h, w = box_map.shape
        ys, xs = torch.meshgrid(
            torch.arange(h, device=box_map.device),
            torch.arange(w, device=box_map.device),
            indexing="ij",
        )
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        ltrb = F.relu(box_map) * stride
        x1 = (cx - ltrb[:, 0]).clamp(min=0)
        y1 = (cy - ltrb[:, 1]).clamp(min=0)
        x2 = cx + ltrb[:, 2]
        y2 = cy + ltrb[:, 3]
        for bi in range(bsz):
            mask = scores[bi] >= score_thr
            if not mask.any():
                continue
            boxes = torch.stack([x1[bi][mask], y1[bi][mask], x2[bi][mask], y2[bi][mask]], dim=1)
            sc = scores[bi][mask]
            lb = labels[bi][mask]
            for box, score, lab in zip(boxes, sc, lb, strict=True):
                x1v, y1v, x2v, y2v = box.tolist()
                per_image[bi].append(([x1v, y1v, x2v - x1v, y2v - y1v], float(score), int(lab)))
    return per_image


def _fcos_targets(gt_xyxy: np.ndarray, gt_cls: np.ndarray, h: int, w: int, stride: int, device):
    torch, _, _ = _torch()
    cls_t = torch.zeros((1, h, w), dtype=torch.long, device=device)
    box_t = torch.zeros((4, h, w), dtype=torch.float32, device=device)
    pos = torch.zeros((h, w), dtype=torch.bool, device=device)
    if gt_xyxy.size == 0:
        return cls_t, box_t, pos
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    cx = (xs + 0.5) * stride
    cy = (ys + 0.5) * stride
    for box, cls in zip(gt_xyxy, gt_cls, strict=True):
        x1, y1, x2, y2 = box.tolist()
        inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        l = cx - x1
        t = cy - y1
        r = x2 - cx
        b = y2 - cy
        if not inside.any():
            continue
        pos = pos | inside
        cls_t[0][inside] = int(cls) + 1
        box_t[0][inside] = l[inside] / stride
        box_t[1][inside] = t[inside] / stride
        box_t[2][inside] = r[inside] / stride
        box_t[3][inside] = b[inside] / stride
    return cls_t, box_t, pos


def _load_image(path: Path, imgsz: int):
    torch, _, _ = _torch()
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor, orig_w, orig_h


class TinyFPNTrainer(BaseTrainer):
    def prepare(self, cfg: dict[str, Any]) -> Path:
        return Path(cfg["dataset"]["root"])

    def _variant(self, cfg: dict[str, Any]) -> str:
        name = str(cfg["model"].get("name", "resnet")).lower()
        if "csp" in name or "rtmdet" in name:
            return "csp"
        return "resnet"

    def _device(self):
        torch, _, _ = _torch()
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, cfg: dict[str, Any], data_path: Path) -> Path:
        torch, nn, F = _torch()
        TinyFPN = _build_modules()
        coco = load_coco(resolve_split_json(cfg, cfg["dataset"]["train_split"]))
        cat_ids = sorted(int(c["id"]) for c in coco["categories"])
        cat_to_idx = {c: i for i, c in enumerate(cat_ids)}
        model = TinyFPN(num_classes=len(cat_ids), variant=self._variant(cfg)).to(self._device())
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("lr0", 1e-3)), weight_decay=1e-4)
        imgsz = int(cfg["train"]["imgsz"])
        grouped = anns_by_image(coco)
        image_dir = Path(cfg["dataset"]["root"]) / "images" / cfg["dataset"]["train_split"]
        epochs = int(cfg["train"]["epochs"])
        t0 = time.perf_counter()
        model.train()
        for _ in range(epochs):
            for im in coco["images"]:
                path = image_dir / Path(im["file_name"]).name
                tensor, orig_w, orig_h = _load_image(path, imgsz)
                tensor = tensor.unsqueeze(0).to(self._device())
                sx, sy = imgsz / orig_w, imgsz / orig_h
                gts = []
                cls = []
                for ann in grouped.get(int(im["id"]), []):
                    x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])
                    gts.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
                    cls.append(cat_to_idx[int(ann["category_id"])])
                gt_xyxy = np.asarray(gts, dtype=np.float32) if gts else np.zeros((0, 4), dtype=np.float32)
                gt_cls = np.asarray(cls, dtype=np.int64) if cls else np.zeros((0,), dtype=np.int64)
                outputs = model(tensor)
                loss = tensor.new_zeros(())
                for level, (cls_map, box_map, ctr_map) in enumerate(outputs):
                    stride = STRIDES[min(level, len(STRIDES) - 1)]
                    _, _, h, w = cls_map.shape
                    cls_t, box_t, pos = _fcos_targets(gt_xyxy, gt_cls, h, w, stride, tensor.device)
                    logits = cls_map[0]
                    # background class 0 in target, shift
                    target = cls_t[0].clamp(min=0)
                    # focal-style BCE on max class vs background
                    prob = logits.sigmoid()
                    pos_mask = pos
                    if pos_mask.any():
                        idx = (target[pos_mask] - 1).clamp(min=0)
                        pos_prob = prob[:, pos_mask].transpose(0, 1)
                        gather = pos_prob[torch.arange(pos_prob.shape[0], device=tensor.device), idx]
                        loss = loss + F.binary_cross_entropy(gather.clamp(1e-4, 1 - 1e-4), torch.ones_like(gather))
                        ltrb = F.relu(box_map[0])
                        loss = loss + F.l1_loss(ltrb[:, pos_mask], box_t[:, pos_mask])
                    neg_mask = ~pos_mask
                    if neg_mask.any():
                        loss = loss + 0.25 * prob[:, neg_mask].pow(2).mean()
                    loss = loss + 0.1 * ctr_map.sigmoid().mean()
                if float(loss) == 0:
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
        out_dir = Path(cfg["train"].get("project", cfg["output"]["root"])) / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": model.state_dict(),
            "num_classes": len(cat_ids),
            "variant": self._variant(cfg),
            "cat_ids": cat_ids,
            "imgsz": imgsz,
        }
        torch.save(ckpt, out_dir / "best.pt")
        meta = {
            "backend": "tiny_fpn",
            "variant": self._variant(cfg),
            "train_seconds": time.perf_counter() - t0,
            "run_name": cfg["name"],
            "fallback_for_mmdet": True,
        }
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_dir.resolve()

    def predict(self, cfg: dict[str, Any], data_path: Path, split: str) -> Path:
        torch, _, _ = _torch()
        TinyFPN = _build_modules()
        weights = resolve_weights(cfg)
        ckpt = torch.load(weights, map_location="cpu") if Path(weights).exists() else None
        coco = load_coco(resolve_split_json(cfg, split))
        cat_ids = (ckpt or {}).get("cat_ids") or sorted(int(c["id"]) for c in coco["categories"])
        variant = (ckpt or {}).get("variant") or self._variant(cfg)
        imgsz = int((ckpt or {}).get("imgsz") or cfg["train"]["imgsz"])
        model = TinyFPN(num_classes=len(cat_ids), variant=variant)
        if ckpt and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        model.to(self._device()).eval()
        images_dir = Path(cfg["dataset"]["root"]) / "images" / split
        expert_id = expert_id_from_cfg(cfg)
        dets: list[Detection] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for im in coco["images"]:
                path = images_dir / Path(im["file_name"]).name
                tensor, orig_w, orig_h = _load_image(path, imgsz)
                outputs = model(tensor.unsqueeze(0).to(self._device()))
                decoded = decode_batch(
                    outputs, score_thr=float(cfg.get("predict", {}).get("conf", 0.3))
                )[0]
                sx, sy = orig_w / imgsz, orig_h / imgsz
                scaled = []
                for bbox, score, lab in decoded:
                    x, y, w, h = bbox
                    scaled.append(
                        Detection(
                            image_id=int(im["id"]),
                            category_id=int(cat_ids[lab]) if lab < len(cat_ids) else int(lab),
                            bbox=[x * sx, y * sy, w * sx, h * sy],
                            score=score,
                            expert_id=expert_id,
                        )
                    )
                dets.extend(nms(scaled, iou_thr=0.65))
        elapsed = time.perf_counter() - t0
        out_dir = Path(cfg["train"].get("project", cfg["output"]["root"])) / cfg["name"]
        out_path = dump_detections(dets, out_dir / f"predictions_{split}.json")
        n_img = max(len(coco["images"]), 1)
        (out_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "backend": "tiny_fpn",
                    f"predict_{split}_seconds": elapsed,
                    "ms_per_image": elapsed / n_img * 1000.0,
                    "fallback_for_mmdet": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return out_path
