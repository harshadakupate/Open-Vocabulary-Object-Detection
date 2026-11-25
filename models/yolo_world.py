"""YOLO-World (Ultralytics) wrapper."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from detection.prompt_manager import prepare_for_yolo_world
from models import DetectionError, ModelLoadError, build_result
from utils.config_loader import load_config, resolve_path
from utils.device import cuda_sync, is_cuda_oom, resolve_device
from utils.logger import get_logger

logger = get_logger("yolo_world")


class YOLOWorldDetector:
    """yolov8s-worldv2.pt through the Ultralytics YOLOWorld API."""

    name = "YOLO-World"

    def __init__(self, checkpoint: str | None = None, weights_dir: str | None = None):
        cfg = load_config()["models"]["yolo_world"]
        self.checkpoint = checkpoint or cfg["checkpoint"]
        self.weights_dir = resolve_path(weights_dir or cfg["weights_dir"])
        self.model = None
        self.device: str | None = None
        self.current_classes: tuple[str, ...] = ()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _ensure_weights(self) -> str:
        """Return a path/name for the checkpoint, downloading into weights/yolo_world when possible."""
        target = self.weights_dir / self.checkpoint
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return str(target)
        try:
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(str(target))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Direct download into %s failed (%s); letting Ultralytics download it.", target, exc)
        return str(target) if target.exists() else self.checkpoint

    def load(self, device: str = "auto") -> None:
        device = resolve_device(device)
        if self.is_loaded:
            if device != self.device:
                self.model.to(device)
                self.device = device
            return
        try:
            from ultralytics import YOLOWorld

            weights = self._ensure_weights()
            logger.info("Loading YOLO-World weights '%s' on %s...", weights, device)
            model = YOLOWorld(weights)
            model.to(device)
            self.model = model
            self.device = device
            self.current_classes = ()
        except Exception as exc:  # noqa: BLE001
            self.model = None
            hint = "Check your internet connection (needed to download the weights) and that 'ultralytics' is installed."
            if is_cuda_oom(exc):
                hint = "GPU ran out of memory while loading; try the CPU device."
            raise ModelLoadError(f"Could not load YOLO-World '{self.checkpoint}': {exc}. {hint}") from exc

    def _apply_classes(self, classes: list[str]) -> None:
        """Call set_classes() when the vocabulary changed, then verify it was applied."""
        wanted = tuple(classes)
        if wanted == self.current_classes:
            return
        try:
            self.model.set_classes(list(wanted))
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(
                f"set_classes() failed: {exc}. If the error mentions 'clip', run: "
                "pip install git+https://github.com/ultralytics/CLIP.git") from exc
        names = getattr(self.model.model, "names", None)
        applied = list(names.values()) if isinstance(names, dict) else list(names or [])
        if applied != list(wanted):
            raise DetectionError(f"Custom classes were not applied (model has {applied}, expected {list(wanted)}).")
        self.current_classes = wanted
        logger.info("YOLO-World vocabulary set: %s", list(wanted))

    def detect(self, image_rgb: np.ndarray, prompts: list[str], conf: float = 0.25,
               imgsz: int | None = None, **_ignored) -> dict:
        """Run detection with the custom vocabulary. Boxes are in image_rgb coordinates."""
        if not self.is_loaded:
            self.load("auto")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise DetectionError("Expected an RGB image of shape (H, W, 3).")
        imgsz = imgsz or load_config()["image"]["yolo_imgsz"]
        classes = prepare_for_yolo_world(prompts)
        self._apply_classes(classes)

        height, width = image_rgb.shape[:2]
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Ultralytics treats numpy input as BGR
        try:
            cuda_sync(self.device)
            t0 = time.perf_counter()
            results = self.model.predict(source=bgr, conf=float(conf), imgsz=int(imgsz),
                                         device=0 if self.device == "cuda" else "cpu", verbose=False)
            cuda_sync(self.device)
            wall = time.perf_counter() - t0
        except RuntimeError as exc:
            if is_cuda_oom(exc):
                import torch

                torch.cuda.empty_cache()
                raise DetectionError("GPU out of memory. Use a smaller image size or the CPU device.") from exc
            raise DetectionError(f"YOLO-World inference failed: {exc}") from exc

        r = results[0]
        speed = getattr(r, "speed", None) or {}
        # Ultralytics times preprocess / inference / postprocess (ms); report model inference + NMS.
        if "inference" in speed:
            inference_time = (speed["inference"] + speed.get("postprocess", 0.0)) / 1000.0
            preprocess_time = speed.get("preprocess", 0.0) / 1000.0
        else:
            inference_time, preprocess_time = wall, 0.0

        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        scores = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,))
        cls_ids = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) else np.zeros((0,), dtype=int)
        labels = [classes[i] if 0 <= i < len(classes) else str(i) for i in cls_ids]

        return build_result(self.name, boxes, labels, scores, inference_time, (height, width),
                            preprocess_time=preprocess_time, wall_time=wall, device=self.device)
