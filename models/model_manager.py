"""Common entry point that selects and runs a detector by name."""
from __future__ import annotations

import re
import threading

import numpy as np

from models.grounding_dino import GroundingDINODetector
from models.yolo_world import YOLOWorldDetector
from utils.config_loader import load_config
from utils.device import resolve_device
from utils.logger import get_logger

logger = get_logger("model_manager")

MODEL_REGISTRY = {
    "Grounding DINO": GroundingDINODetector,
    "YOLO-World": YOLOWorldDetector,
}
MODEL_NAMES = list(MODEL_REGISTRY)


def normalize_model_name(name: str) -> str:
    """Accept 'grounding-dino', 'yolo_world', ... and return the canonical display name."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    for canonical in MODEL_REGISTRY:
        if re.sub(r"[^a-z0-9]", "", canonical.lower()) == key:
            return canonical
    raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_NAMES)}")


def model_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_model_name(name).lower()).strip("_")


class ModelManager:
    """Loads detectors lazily, keeps them in memory and exposes one detect() call."""

    def __init__(self):
        self._cfg = load_config()
        self._detectors: dict = {}
        self._lock = threading.RLock()

    @staticmethod
    def available_models() -> list[str]:
        return list(MODEL_NAMES)

    def default_conf(self, model_name: str) -> float:
        det = self._cfg["detection"]
        return det["dino_box_threshold"] if model_name == "Grounding DINO" else det["yolo_conf"]

    def get_detector(self, model_name: str, device: str = "auto"):
        model_name = normalize_model_name(model_name)
        device = resolve_device(device)
        with self._lock:
            detector = self._detectors.get(model_name)
            fresh = detector is None
            if fresh:
                detector = MODEL_REGISTRY[model_name]()
            detector.load(device)  # raises ModelLoadError on failure
            self._detectors[model_name] = detector
            if fresh:
                self._warmup(detector)
        return detector

    @staticmethod
    def _warmup(detector) -> None:
        """One untimed pass so CUDA kernels / lazy init do not pollute the first real timing."""
        try:
            detector.detect(np.full((480, 640, 3), 127, dtype=np.uint8), ["object"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warm-up pass failed (ignored): %s", exc)

    def detect(self, model_name: str, image_rgb: np.ndarray, prompts: list[str], device: str = "auto",
               conf: float | None = None, text_threshold: float | None = None,
               imgsz: int | None = None) -> dict:
        """Run the selected model. Returns the common result dict (boxes in image_rgb coordinates)."""
        model_name = normalize_model_name(model_name)
        detector = self.get_detector(model_name, device)
        conf = self.default_conf(model_name) if conf is None else conf
        if text_threshold is None:
            text_threshold = self._cfg["detection"]["dino_text_threshold"]
        with self._lock:
            return detector.detect(image_rgb, prompts, conf=conf, text_threshold=text_threshold, imgsz=imgsz)

    def warmup(self, model_name: str, image_rgb: np.ndarray, prompts: list[str], device: str = "auto",
               runs: int = 3, **kwargs) -> None:
        """Run untimed inferences on real data (used before benchmarking)."""
        for _ in range(max(0, runs)):
            self.detect(model_name, image_rgb, prompts, device=device, **kwargs)
