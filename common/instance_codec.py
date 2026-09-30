"""Box NMS and COCO-style mask RLE used by open detection / segmentation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def class_aware_nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[str],
    iou_thr: float = 0.5,
) -> List[int]:
    """Return kept indices (high score first), NMS within the same label."""
    if not boxes:
        return []
    boxes_np = np.asarray(boxes, dtype=np.float32)
    scores_np = np.asarray(scores, dtype=np.float32)
    keep: List[int] = []
    for label in dict.fromkeys(labels):
        idxs = [i for i, lab in enumerate(labels) if lab == label]
        order = sorted(idxs, key=lambda i: float(scores_np[i]), reverse=True)
        while order:
            i = order.pop(0)
            keep.append(i)
            if not order:
                break
            ious = box_iou_xyxy(boxes_np[i], boxes_np[order])
            order = [order[j] for j, iou in enumerate(ious) if iou <= iou_thr]
    keep.sort(key=lambda i: float(scores_np[i]), reverse=True)
    return keep


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Uncompressed COCO RLE (counts as int list, Fortran order)."""
    binary = np.asfortranarray(np.asarray(mask).astype(np.uint8))
    if binary.ndim != 2:
        raise ValueError(f"mask must be HxW, got {binary.shape}")
    h, w = binary.shape
    pixels = binary.ravel(order="F")
    if pixels.size == 0:
        return {"size": [int(h), int(w)], "counts": []}
    diffs = np.diff(pixels)
    starts = np.concatenate(([0], np.where(diffs != 0)[0] + 1, [pixels.size]))
    counts: List[int] = []
    if int(pixels[0]) == 1:
        counts.append(0)
    for i in range(len(starts) - 1):
        counts.append(int(starts[i + 1] - starts[i]))
    return {"size": [int(h), int(w)], "counts": counts}


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    h, w = rle["size"]
    counts = rle.get("counts") or []
    pixels = np.zeros(int(h) * int(w), dtype=np.uint8)
    pos = 0
    val = 0
    for run in counts:
        nxt = pos + int(run)
        if val:
            pixels[pos:nxt] = 1
        pos = nxt
        val = 1 - val
    return pixels.reshape((int(h), int(w)), order="F")


def bbox_from_mask(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(np.asarray(mask) > 0)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
