"""Shared helpers for tests."""
from __future__ import annotations

import cv2
import numpy as np


def synthetic_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Deterministic RGB test image (not a claim about detections - used for contract checks)."""
    rng = np.random.default_rng(0)
    img = rng.integers(90, 160, size=(h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (w // 4, h // 4), (w // 2, 3 * h // 4), (200, 40, 40), -1)
    cv2.circle(img, (3 * w // 4, h // 2), h // 6, (40, 40, 200), -1)
    return img


def assert_result_contract(result: dict, image: np.ndarray, model_name: str) -> None:
    h, w = image.shape[:2]
    assert result["model_name"] == model_name
    assert result["image_width"] == w and result["image_height"] == h
    n = result["num_detections"]
    assert n == len(result["boxes"]) == len(result["labels"]) == len(result["scores"])
    assert result["inference_time"] > 0
    for (x1, y1, x2, y2), score in zip(result["boxes"], result["scores"]):
        assert 0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h
        assert 0.0 <= score <= 1.0


def make_video(path, frames: int = 10, size=(320, 240), fps: int = 10) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV could not open a video writer (mp4v)"
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), 100, dtype=np.uint8)
        cv2.rectangle(frame, (10 + 5 * i, 40), (90 + 5 * i, 140), (0, 180, 255), -1)
        writer.write(frame)
    writer.release()
