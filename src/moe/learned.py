"""Learned MoE: frozen experts, MLP fusion head trained on cal clusters."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.data.boxes import box_giou_xyxy, xywh_to_xyxy, xyxy_to_xywh
from src.data.coco import anns_by_image, image_size_map
from src.data.schema import Detection
from src.fusion.wbf import cluster_detections


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def _cluster_feature(
    cluster: list[Detection],
    experts: list[str],
    num_classes: int,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    by_exp: dict[str, Detection] = {}
    for det in cluster:
        prev = by_exp.get(det.expert_id)
        if prev is None or det.score > prev.score:
            by_exp[det.expert_id] = det
    feats: list[float] = []
    for expert in experts:
        det = by_exp.get(expert)
        if det is None:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            feats.extend([0.0] * num_classes)
            continue
        x, y, w, h = det.bbox
        feats.extend(
            [
                x / max(img_w, 1),
                y / max(img_h, 1),
                w / max(img_w, 1),
                h / max(img_h, 1),
                float(det.score),
                1.0,
            ]
        )
        onehot = [0.0] * num_classes
        # class index is not known here; caller passes category_id -> idx via num_classes and we use id if in range
        idx = int(det.category_id)
        if 0 <= idx < num_classes:
            onehot[idx] = 1.0
        elif 1 <= idx <= num_classes:
            onehot[idx - 1] = 1.0
        feats.extend(onehot)
    return np.asarray(feats, dtype=np.float32)


class FusionMLP:
    def __init__(self, experts: list[str], num_classes: int, hidden: int = 64):
        self.experts = experts
        self.num_classes = num_classes
        torch, nn, _ = _torch()
        in_dim = len(experts) * (6 + num_classes)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4 + num_classes + 1),
        )
        self.cat_ids: list[int] = list(range(num_classes))

    def _device(self):
        torch, _, _ = _torch()
        return next(self.net.parameters()).device

    def fit(
        self,
        dets: list[Detection],
        coco_gt: dict,
        *,
        iou_thr: float = 0.55,
        epochs: int = 20,
        lr: float = 1e-3,
        kfold: int = 1,
    ) -> "FusionMLP":
        torch, _, F = _torch()
        sizes = image_size_map(coco_gt)
        grouped_gt = anns_by_image(coco_gt)
        self.cat_ids = sorted(int(c["id"]) for c in coco_gt["categories"])
        self.num_classes = len(self.cat_ids)
        cat_to_idx = {c: i for i, c in enumerate(self.cat_ids)}
        # rebuild net if class count changed
        in_dim = len(self.experts) * (6 + self.num_classes)
        _, nn, _ = _torch()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4 + self.num_classes + 1),
        )
        xs = []
        box_t = []
        cls_t = []
        obj_t = []
        for cluster in cluster_detections(dets, iou_thr=iou_thr):
            image_id = cluster[0].image_id
            img_w, img_h = sizes.get(image_id, (1, 1))
            feat = _cluster_feature(cluster, self.experts, self.num_classes, img_w, img_h)
            seed = max(cluster, key=lambda d: d.score)
            sx1, sy1, sx2, sy2 = xywh_to_xyxy(seed.bbox)
            matched = None
            best_iou = 0.5
            for gt in grouped_gt.get(image_id, []):
                gx1, gy1, gx2, gy2 = xywh_to_xyxy(gt["bbox"])
                # reuse giou-derived iou
                from src.data.boxes import box_iou_xyxy

                iou = box_iou_xyxy((sx1, sy1, sx2, sy2), (gx1, gy1, gx2, gy2))
                if iou >= best_iou:
                    best_iou = iou
                    matched = gt
            xs.append(feat)
            if matched is None:
                box_t.append([sx1 / img_w, sy1 / img_h, sx2 / img_w, sy2 / img_h])
                cls_t.append(0)
                obj_t.append(0.0)
            else:
                gx1, gy1, gx2, gy2 = xywh_to_xyxy(matched["bbox"])
                box_t.append([gx1 / img_w, gy1 / img_h, gx2 / img_w, gy2 / img_h])
                cls_t.append(cat_to_idx.get(int(matched["category_id"]), 0))
                obj_t.append(1.0)
        if not xs:
            return self
        x = torch.tensor(np.stack(xs), dtype=torch.float32)
        y_box = torch.tensor(np.asarray(box_t), dtype=torch.float32)
        y_cls = torch.tensor(np.asarray(cls_t), dtype=torch.long)
        y_obj = torch.tensor(np.asarray(obj_t), dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        n = len(x)
        folds = max(int(kfold), 1)
        for _ in range(epochs):
            if folds > 1 and n >= folds:
                idx = torch.randperm(n)
            else:
                idx = torch.arange(n)
            pred = self.net(x[idx])
            boxes, logits, obj = pred[:, :4], pred[:, 4:-1], pred[:, -1]
            loss = F.l1_loss(boxes.sigmoid(), y_box[idx])
            loss = loss + F.cross_entropy(logits, y_cls[idx])
            loss = loss + F.binary_cross_entropy_with_logits(obj, y_obj[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    def predict_clusters(
        self,
        dets: list[Detection],
        sizes: dict[int, tuple[int, int]],
        iou_thr: float = 0.55,
        score_thr: float = 0.3,
    ) -> list[Detection]:
        torch, _, _ = _torch()
        cat_to_idx = {c: i for i, c in enumerate(self.cat_ids)}
        idx_to_cat = {i: c for c, i in cat_to_idx.items()}
        out: list[Detection] = []
        self.net.eval()
        with torch.no_grad():
            for cluster in cluster_detections(dets, iou_thr=iou_thr):
                image_id = cluster[0].image_id
                img_w, img_h = sizes.get(image_id, (1, 1))
                feat = _cluster_feature(cluster, self.experts, self.num_classes, img_w, img_h)
                pred = self.net(torch.tensor(feat).unsqueeze(0))[0]
                box = pred[:4].sigmoid().tolist()
                logits = pred[4:-1]
                obj = float(torch.sigmoid(pred[-1]))
                if obj < score_thr:
                    continue
                cls_i = int(logits.argmax().item())
                x1, y1, x2, y2 = box[0] * img_w, box[1] * img_h, box[2] * img_w, box[3] * img_h
                out.append(
                    Detection(
                        image_id=image_id,
                        category_id=int(idx_to_cat.get(cls_i, cls_i)),
                        bbox=xyxy_to_xywh([x1, y1, x2, y2]),
                        score=obj,
                        expert_id="learned",
                    )
                )
        return out


def learned_moe(
    dets: list[Detection],
    head: FusionMLP,
    sizes: dict[int, tuple[int, int]],
) -> list[Detection]:
    return head.predict_clusters(dets, sizes)
