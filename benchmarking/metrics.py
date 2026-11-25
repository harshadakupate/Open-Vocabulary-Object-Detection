"""Detection metrics: IoU, one-to-one matching, precision/recall/F1, AP and mAP.

Matching procedure (documented, used everywhere):
  1. Predictions of one class are sorted by confidence, highest first.
  2. Each prediction is compared with the not-yet-matched ground-truth boxes of the SAME class
     in the SAME image; the one with the highest IoU is taken if IoU >= threshold -> True Positive.
  3. Otherwise the prediction is a False Positive. Each ground-truth box can be matched once.
  4. Unmatched ground-truth boxes are False Negatives.
AP uses the COCO 101-point interpolated precision envelope. mAP@50-95 averages AP over
IoU thresholds 0.50, 0.55, ..., 0.95. Precision/Recall/F1 are computed at the operating
confidence threshold; AP uses all predictions (low threshold) so the full ranking is evaluated.
"""
from __future__ import annotations

import math

import numpy as np

IOU_THRESHOLDS_50_95 = [round(0.5 + 0.05 * i, 2) for i in range(10)]
NAN = float("nan")


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in xyxy format -> (N,M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def pair_boxes(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> list[float]:
    """Greedy one-to-one pairing of two box sets by highest IoU. Returns IoUs of the pairs."""
    m = iou_xyxy(a, b).copy()
    ious: list[float] = []
    while m.size:
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        if m[i, j] < threshold:
            break
        ious.append(float(m[i, j]))
        m[i, :] = -1.0
        m[:, j] = -1.0
    return ious


def _norm(label) -> str:
    return " ".join(str(label).lower().split())


def match_class_predictions(preds: list[tuple], gt_boxes_by_image: dict, iou_threshold: float) -> list[tuple[float, bool]]:
    """preds: [(image_id, score, box)]. Returns [(score, is_true_positive)] sorted by score desc."""
    ordered = sorted(preds, key=lambda p: -p[1])
    used = {img: np.zeros(len(boxes), dtype=bool) for img, boxes in gt_boxes_by_image.items()}
    flags: list[tuple[float, bool]] = []
    for image_id, score, box in ordered:
        gts = gt_boxes_by_image.get(image_id)
        is_tp = False
        if gts is not None and len(gts):
            ious = iou_xyxy(np.asarray([box]), gts)[0]
            ious = np.where(used[image_id], -1.0, ious)
            j = int(np.argmax(ious))
            if ious[j] >= iou_threshold:
                used[image_id][j] = True
                is_tp = True
        flags.append((float(score), is_tp))
    return flags


def average_precision(flags: list[tuple[float, bool]], num_gt: int) -> float:
    """COCO-style 101-point interpolated AP from score-sorted TP flags."""
    if num_gt == 0:
        return NAN
    if not flags:
        return 0.0
    tp = np.cumsum([1 if f[1] else 0 for f in flags], dtype=float)
    fp = np.cumsum([0 if f[1] else 1 for f in flags], dtype=float)
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(recall, grid, side="left")
    sampled = np.array([envelope[i] if i < len(envelope) else 0.0 for i in idx])
    return float(sampled.mean())


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else NAN
    recall = tp / (tp + fn) if (tp + fn) > 0 else NAN
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else NAN  # == 2PR/(P+R)
    return precision, recall, f1


def confidence_stats(scores: list[float]) -> dict:
    """Average/max/min confidence and detection count (confidence is NOT accuracy)."""
    if not scores:
        return {"avg_confidence": NAN, "max_confidence": NAN, "min_confidence": NAN, "detection_count": 0}
    arr = np.asarray(scores, dtype=float)
    return {"avg_confidence": float(arr.mean()), "max_confidence": float(arr.max()),
            "min_confidence": float(arr.min()), "detection_count": int(arr.size)}


def evaluate_detections(predictions: dict, ground_truths: dict, classes: list[str],
                        iou_threshold: float = 0.5, operating_conf: float = 0.0) -> dict:
    """Compute overall and per-class precision, recall, F1, AP@50 and AP@50-95.

    predictions:   {image_id: {"boxes": [[x1,y1,x2,y2]], "labels": [str], "scores": [float]}}
    ground_truths: {image_id: {"boxes": [[x1,y1,x2,y2]], "labels": [str]}}
    classes:       categories to evaluate. Predictions of other labels are ignored (counted separately).
    """
    class_names = []
    for c in classes:
        if _norm(c) not in class_names:
            class_names.append(_norm(c))
    image_ids = list(dict.fromkeys(list(ground_truths) + list(predictions)))

    per_class_preds = {c: [] for c in class_names}
    per_class_gt = {c: {} for c in class_names}
    ignored = 0
    for img in image_ids:
        gt = ground_truths.get(img, {"boxes": [], "labels": []})
        for c in class_names:
            boxes = [b for b, l in zip(gt["boxes"], gt["labels"]) if _norm(l) == c]
            per_class_gt[c][img] = np.asarray(boxes, dtype=float).reshape(-1, 4)
        pred = predictions.get(img)
        if pred:
            for b, l, s in zip(pred["boxes"], pred["labels"], pred["scores"]):
                if _norm(l) in per_class_preds:
                    per_class_preds[_norm(l)].append((img, float(s), b))
                else:
                    ignored += 1

    rows = []
    tot_tp = tot_fp = tot_fn = 0
    ap50_vals, ap5095_vals = [], []
    for c in class_names:
        n_gt = int(sum(len(v) for v in per_class_gt[c].values()))
        all_preds = per_class_preds[c]
        kept = [p for p in all_preds if p[1] >= operating_conf]
        flags = match_class_predictions(kept, per_class_gt[c], iou_threshold)
        tp = sum(1 for _, t in flags if t)
        fp = len(flags) - tp
        fn = n_gt - tp
        precision, recall, f1 = _prf(tp, fp, fn)
        aps = [average_precision(match_class_predictions(all_preds, per_class_gt[c], t), n_gt)
               for t in IOU_THRESHOLDS_50_95]
        ap50 = aps[0]
        ap5095 = float(np.mean(aps)) if n_gt > 0 else NAN
        if n_gt > 0:
            ap50_vals.append(ap50)
            ap5095_vals.append(ap5095)
        tot_tp, tot_fp, tot_fn = tot_tp + tp, tot_fp + fp, tot_fn + fn
        rows.append({"category": c, "num_gt": n_gt, "num_pred": len(kept), "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1, "ap50": ap50, "ap50_95": ap5095})

    precision, recall, f1 = _prf(tot_tp, tot_fp, tot_fn)
    kept_scores = [p[1] for c in class_names for p in per_class_preds[c] if p[1] >= operating_conf]
    overall = {
        "precision": precision, "recall": recall, "f1": f1,
        "map50": float(np.mean(ap50_vals)) if ap50_vals else NAN,
        "map50_95": float(np.mean(ap5095_vals)) if ap5095_vals else NAN,
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "iou_threshold": iou_threshold, "operating_conf": operating_conf,
        "ignored_predictions": ignored,
        **confidence_stats(kept_scores),
    }
    return {"overall": overall, "per_class": rows}


def is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)
