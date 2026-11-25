"""Detector wrappers sharing one common output format."""
from __future__ import annotations


class ModelLoadError(RuntimeError):
    """Raised when a model or its weights cannot be loaded."""


class DetectionError(RuntimeError):
    """Raised when inference fails (e.g. GPU out of memory)."""


def build_result(model_name: str, boxes, labels, scores, inference_time: float,
                 image_shape: tuple, **extra) -> dict:
    """Build the standard detection dict. Boxes are xyxy in the coordinates of image_shape."""
    height, width = int(image_shape[0]), int(image_shape[1])
    clipped = []
    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
        y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
        clipped.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
    result = {
        "model_name": model_name,
        "boxes": clipped,
        "labels": [str(label) for label in labels],
        "scores": [round(float(s), 4) for s in scores],
        "inference_time": float(inference_time),  # seconds; model forward + post-processing
        "image_width": width,
        "image_height": height,
        "num_detections": len(clipped),
    }
    result.update(extra)
    return result
