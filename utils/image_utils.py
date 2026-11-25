"""Image loading and resizing helpers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageLoadError(ValueError):
    """Raised when an image cannot be read or decoded."""


def validate_extension(filename: str, allowed: list[str] | tuple[str, ...]) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise ImageLoadError(f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}")


def load_image_rgb(source) -> np.ndarray:
    """Load an image from a path, bytes, or file-like object as an RGB uint8 array."""
    try:
        if isinstance(source, (str, Path)):
            data = np.fromfile(str(source), dtype=np.uint8)  # unicode-safe on Windows
        elif isinstance(source, (bytes, bytearray)):
            data = np.frombuffer(source, dtype=np.uint8)
        elif hasattr(source, "getvalue"):
            data = np.frombuffer(source.getvalue(), dtype=np.uint8)
        elif hasattr(source, "read"):
            data = np.frombuffer(source.read(), dtype=np.uint8)
        else:
            raise ImageLoadError(f"Unsupported image source type: {type(source).__name__}")
    except OSError as exc:
        raise ImageLoadError(f"Could not read image file: {exc}") from exc
    if data.size == 0:
        raise ImageLoadError("The image file is empty.")
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageLoadError("Could not decode the image - the file is corrupt or not a supported image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_long_side(image: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
    """Downscale so the long side is at most max_side. Returns (image, scale). 0/None disables."""
    if not max_side or max_side <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale
