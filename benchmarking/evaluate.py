"""Evaluate one model on an annotated image set (COCO-style ground truth).

Usage:
  python -m benchmarking.evaluate --model "Grounding DINO" \
      --images-dir dataset/images --annotations dataset/annotations/annotations.json

Ground truth format: COCO detection JSON ("images", "annotations", "categories"), bbox = [x, y, w, h].
Nothing is computed without ground truth; a missing/invalid annotation file stops with a clear error.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarking.metrics import evaluate_detections
from benchmarking.performance_monitor import PerformanceMonitor
from detection.image_detection import run_image_detection
from utils.config_loader import ensure_project_dirs, load_config, resolve_path
from utils.device import resolve_device
from utils.image_utils import ImageLoadError, load_image_rgb
from utils.logger import get_logger

logger = get_logger("evaluate")


class AnnotationFormatError(ValueError):
    """Raised when the annotation file is missing or not valid COCO-style JSON."""


def load_coco_annotations(path: str | Path) -> tuple[dict, list[str], list[str]]:
    """Return (ground_truths by file_name, ordered file names, category names as written)."""
    path = Path(path)
    if not path.exists():
        raise AnnotationFormatError(
            f"Annotation file not found: {path}. Ground truth is required - evaluation has NOT been completed.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotationFormatError(f"Annotation file is not valid JSON: {exc}") from exc
    for key in ("images", "annotations", "categories"):
        if not isinstance(data, dict) or key not in data:
            raise AnnotationFormatError(f"Annotation file must be COCO-style and contain '{key}'.")
    images = {img["id"]: img["file_name"] for img in data["images"] if "id" in img and "file_name" in img}
    categories = {c["id"]: str(c["name"]) for c in data["categories"] if "id" in c and "name" in c}
    if not images or not categories:
        raise AnnotationFormatError("Annotation file has no usable 'images' or 'categories'.")
    gts = {name: {"boxes": [], "labels": []} for name in images.values()}
    for ann in data["annotations"]:
        try:
            x, y, w, h = (float(v) for v in ann["bbox"])
            name = images[ann["image_id"]]
            label = categories[ann["category_id"]]
        except (KeyError, ValueError, TypeError) as exc:
            raise AnnotationFormatError(f"Invalid annotation entry {ann!r}: {exc}") from exc
        if w <= 0 or h <= 0:
            raise AnnotationFormatError(f"Annotation {ann.get('id')} has a non-positive box size.")
        gts[name]["boxes"].append([x, y, x + w, y + h])
        gts[name]["labels"].append(label)
    return gts, list(images.values()), list(categories.values())


def _clean(obj):
    """Make values JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_evaluation(manager, model_name: str, images_dir: str | Path, annotations_path: str | Path,
                   prompts: list[str] | None = None, device: str = "auto", conf: float | None = None,
                   text_threshold: float | None = None, imgsz: int | None = None,
                   max_side: int | None = None, warmup_runs: int | None = None,
                   out_dir: str | Path | None = None, limit: int | None = None,
                   progress_callback=None) -> dict:
    """Run a model over every annotated image and compute metrics + performance statistics."""
    from models.model_manager import model_slug, normalize_model_name

    cfg = load_config()
    bench = cfg["benchmark"]
    model_name = normalize_model_name(model_name)
    device = resolve_device(device)
    conf = manager.default_conf(model_name) if conf is None else conf
    text_threshold = cfg["detection"]["dino_text_threshold"] if text_threshold is None else text_threshold
    warmup_runs = bench["warmup_runs"] if warmup_runs is None else warmup_runs
    out_dir = Path(out_dir) if out_dir else resolve_path(cfg["paths"]["reports_out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = resolve_path(images_dir)

    gts, file_names, categories = load_coco_annotations(resolve_path(annotations_path))
    if limit:
        file_names = file_names[:limit]
    prompts = prompts or categories
    missing = [c for c in categories if c.lower() not in {p.lower() for p in prompts}]
    if missing:
        logger.warning("Categories in annotations but not in prompts (will have zero recall): %s", missing)

    existing = [f for f in file_names if (images_dir / f).exists()]
    skipped = [f for f in file_names if f not in existing]
    if skipped:
        logger.warning("%d annotated images not found in %s (skipped).", len(skipped), images_dir)
    if not existing:
        raise AnnotationFormatError(f"None of the annotated images were found in {images_dir}.")

    # Untimed warm-up on real data (model load + CUDA kernels excluded from steady-state timings).
    first_img = load_image_rgb(images_dir / existing[0])
    manager.get_detector(model_name, device)
    manager.warmup(model_name, first_img, prompts, device=device, runs=warmup_runs, conf=conf,
                   text_threshold=text_threshold, imgsz=imgsz)

    infer_conf = min(conf, bench["map_conf_threshold"])  # low threshold -> full ranking for AP
    monitor = PerformanceMonitor(device, interval=bench["monitor_interval_s"])
    predictions, per_image = {}, []
    monitor.start()
    try:
        for i, name in enumerate(existing, start=1):
            t0 = time.perf_counter()
            try:
                image = load_image_rgb(images_dir / name)
            except ImageLoadError as exc:
                logger.warning("Skipping unreadable image %s: %s", name, exc)
                skipped.append(name)
                continue
            result = run_image_detection(manager, model_name, image, prompts, device=device,
                                         conf=infer_conf, text_threshold=text_threshold,
                                         max_side=max_side, imgsz=imgsz)
            e2e = time.perf_counter() - t0
            monitor.record_inference(result["inference_time"])
            monitor.record_item(e2e)
            predictions[name] = {"boxes": result["boxes"], "labels": result["labels"], "scores": result["scores"]}
            per_image.append({"image": name, "inference_ms": result["inference_time"] * 1000,
                              "end_to_end_ms": e2e * 1000,
                              "detections_at_operating_conf": int(sum(s >= conf for s in result["scores"]))})
            if progress_callback:
                progress_callback(i, len(existing), name)
    finally:
        monitor.stop()

    eval_gts = {n: gts[n] for n in predictions}
    metrics = evaluate_detections(predictions, eval_gts, categories, bench["iou_threshold"], conf)
    perf = monitor.summary()

    summary = {
        "model_name": model_name, "device": device,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "images_dir": str(images_dir), "annotations": str(annotations_path),
        "num_images_evaluated": len(predictions), "num_images_skipped": len(skipped),
        "categories": categories, "prompts": prompts,
        "operating_conf": conf, "text_threshold": text_threshold, "inference_conf_for_ap": infer_conf,
        "warmup_runs": warmup_runs, "max_side": max_side, "imgsz": imgsz,
        "overall": metrics["overall"], "per_class": metrics["per_class"], "performance": perf,
    }
    slug = model_slug(model_name)
    (out_dir / f"summary_{slug}.json").write_text(json.dumps(_clean(summary), indent=2), encoding="utf-8")
    (out_dir / f"predictions_{slug}.json").write_text(json.dumps(_clean(predictions)), encoding="utf-8")
    pd.DataFrame(metrics["per_class"]).to_csv(out_dir / f"per_category_{slug}.csv", index=False)
    pd.DataFrame(per_image).to_csv(out_dir / f"per_image_{slug}.csv", index=False)
    monitor.export_csv(out_dir / f"monitor_{slug}.csv")
    logger.info("Saved evaluation for %s to %s", model_name, out_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Evaluate one open-vocabulary model against ground truth.")
    p.add_argument("--model", required=True, help="'Grounding DINO' or 'YOLO-World'")
    p.add_argument("--images-dir", default=cfg["benchmark"]["images_dir"])
    p.add_argument("--annotations", default=cfg["benchmark"]["annotations"])
    p.add_argument("--device", default=cfg["device"]["default"])
    p.add_argument("--conf", type=float, default=None, help="Operating confidence / box threshold")
    p.add_argument("--text-threshold", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--max-side", type=int, default=None)
    p.add_argument("--warmup-runs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images")
    p.add_argument("--prompts", default=None, help="Optional comma-separated prompts (default: annotation categories)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from detection.prompt_manager import parse_prompts
    from models.model_manager import ModelManager

    ensure_project_dirs()
    try:
        summary = run_evaluation(ModelManager(), args.model, args.images_dir, args.annotations,
                                 parse_prompts(args.prompts) if args.prompts else None, args.device,
                                 args.conf, args.text_threshold, args.imgsz, args.max_side,
                                 args.warmup_runs, limit=args.limit)
    except AnnotationFormatError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    o, p = summary["overall"], summary["performance"]
    print(f"\n{summary['model_name']} on {summary['device']} ({summary['num_images_evaluated']} images)")
    print(f"  precision={o['precision']:.3f} recall={o['recall']:.3f} f1={o['f1']:.3f} "
          f"mAP@50={o['map50']:.3f} mAP@50-95={o['map50_95']:.3f}")
    print(f"  avg inference={p['avg_inference_ms']:.1f} ms | inference FPS={p['inference_fps']:.2f} | "
          f"end-to-end FPS={p['end_to_end_fps']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
