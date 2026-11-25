"""Drawing detections and turning results into tables."""
from __future__ import annotations

import colorsys
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.config_loader import load_config, resolve_path


def color_for_label(label: str) -> tuple[int, int, int]:
    """Stable RGB colour per category (same label -> same colour across runs)."""
    digest = hashlib.md5(label.lower().encode("utf-8")).digest()
    r, g, b = colorsys.hsv_to_rgb(digest[0] / 255.0, 0.75, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_detections(image_rgb: np.ndarray, result: dict, header: str | None = None,
                    show_summary: bool = True) -> np.ndarray:
    """Return a copy of the image with boxes, 'label NN%' tags and a summary bar."""
    canvas = image_rgb.copy()
    h, w = canvas.shape[:2]
    thickness = max(2, int(round(max(h, w) / 500)))
    font_scale = max(0.5, min(h, w) / 900)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, thickness - 1)

    for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color = color_for_label(label)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        tag = f"{label} {score * 100:.0f}%"
        (tw, th), baseline = cv2.getTextSize(tag, font, font_scale, text_thickness)
        top = y1 - th - baseline - 4
        if top < 0:
            top = y1
        cv2.rectangle(canvas, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1)
        cv2.putText(canvas, tag, (x1 + 3, top + th + 1), font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

    if show_summary or header:
        text = header or (f"{result['model_name']} | {result['num_detections']} detections | "
                          f"{result['inference_time'] * 1000:.1f} ms")
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        cv2.rectangle(canvas, (0, 0), (tw + 12, th + baseline + 10), (20, 24, 31), -1)
        cv2.putText(canvas, text, (6, th + 5), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return canvas


def result_to_dataframe(result: dict) -> pd.DataFrame:
    """Detections as a table (xyxy pixels in the original image)."""
    columns = ["#", "label", "confidence_%", "x1", "y1", "x2", "y2", "width", "height"]
    rows = []
    for i, (box, label, score) in enumerate(zip(result["boxes"], result["labels"], result["scores"]), start=1):
        x1, y1, x2, y2 = box
        rows.append({"#": i, "label": label, "confidence_%": round(score * 100, 1),
                     "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                     "width": round(x2 - x1, 1), "height": round(y2 - y1, 1)})
    return pd.DataFrame(rows, columns=columns)


def save_annotated_image(image_rgb: np.ndarray, directory: str | Path | None = None,
                         stem: str = "detection") -> Path:
    """Save an RGB image as PNG under outputs/images (unicode-safe). Returns the file path."""
    directory = Path(directory) if directory else resolve_path(load_config()["paths"]["images_out"])
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60] or "detection"
    path = directory / f"{safe_stem}_{stamp}.png"
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError("Could not encode the annotated image.")
    path.write_bytes(buf.tobytes())
    return path
