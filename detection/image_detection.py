"""Image detection pipeline (also usable from the command line)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from detection.prompt_manager import parse_prompts, require_prompts
from detection.visualization import draw_detections, save_annotated_image
from utils.config_loader import ensure_project_dirs, load_config
from utils.image_utils import load_image_rgb, resize_long_side, validate_extension
from utils.logger import get_logger

logger = get_logger("image_detection")


def run_image_detection(manager, model_name: str, image_rgb: np.ndarray, prompts: list[str],
                        device: str = "auto", conf: float | None = None,
                        text_threshold: float | None = None, max_side: int | None = None,
                        imgsz: int | None = None) -> dict:
    """Run one model on one image. Returned boxes are in ORIGINAL image coordinates.

    The image may be downscaled for inference (max_side); boxes are mapped back afterwards.
    """
    prompts = require_prompts(prompts)
    cfg = load_config()["image"]
    max_side = cfg["inference_max_side"] if max_side is None else max_side
    orig_h, orig_w = image_rgb.shape[:2]

    small, scale = resize_long_side(image_rgb, max_side)
    result = manager.detect(model_name, small, prompts, device=device, conf=conf,
                            text_threshold=text_threshold, imgsz=imgsz)
    if scale != 1.0:
        result["boxes"] = [
            [round(min(max(v / scale, 0.0), lim), 2) for v, lim in zip(box, (orig_w, orig_h, orig_w, orig_h))]
            for box in result["boxes"]
        ]
    result["image_width"], result["image_height"] = orig_w, orig_h
    result["inference_scale"] = scale
    result["prompts"] = prompts
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open-vocabulary detection on one image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", default="Grounding DINO", help="'Grounding DINO' or 'YOLO-World'")
    parser.add_argument("--prompts", required=True, help="Comma-separated categories, e.g. 'person, safety helmet'")
    parser.add_argument("--device", default=load_config()["device"]["default"])
    parser.add_argument("--conf", type=float, default=None, help="Confidence / box threshold")
    parser.add_argument("--text-threshold", type=float, default=None)
    parser.add_argument("--max-side", type=int, default=None)
    args = parser.parse_args(argv)

    from models.model_manager import ModelManager, normalize_model_name

    ensure_project_dirs()
    validate_extension(args.image, load_config()["image"]["allowed_extensions"])
    image = load_image_rgb(args.image)
    result = run_image_detection(ModelManager(), normalize_model_name(args.model), image,
                                 parse_prompts(args.prompts), args.device, args.conf,
                                 args.text_threshold, args.max_side)
    out = save_annotated_image(draw_detections(image, result), stem=Path(args.image).stem)
    print(json.dumps({k: v for k, v in result.items() if k != "prompts"}, indent=2))
    print(f"\nAnnotated image saved to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
