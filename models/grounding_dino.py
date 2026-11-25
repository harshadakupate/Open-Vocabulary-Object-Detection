"""Grounding DINO (Hugging Face Transformers) wrapper."""
from __future__ import annotations

import inspect
import time

import numpy as np
from PIL import Image

from detection.prompt_manager import match_label_to_prompt, prepare_for_grounding_dino
from models import DetectionError, ModelLoadError, build_result
from utils.config_loader import load_config, resolve_path
from utils.device import cuda_sync, is_cuda_oom, resolve_device
from utils.logger import get_logger

logger = get_logger("grounding_dino")


class GroundingDINODetector:
    """IDEA-Research/grounding-dino-tiny via AutoModelForZeroShotObjectDetection."""

    name = "Grounding DINO"

    def __init__(self, checkpoint: str | None = None, cache_dir: str | None = None):
        cfg = load_config()["models"]["grounding_dino"]
        self.checkpoint = checkpoint or cfg["checkpoint"]
        self.cache_dir = resolve_path(cache_dir or cfg["cache_dir"])
        self.processor = None
        self.model = None
        self.device: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, device: str = "auto") -> None:
        """Load (once) and place the model on the requested device."""
        device = resolve_device(device)
        if self.is_loaded:
            if device != self.device:
                self.model.to(device)
                self.device = device
            return
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Loading %s on %s (first run downloads the weights)...", self.checkpoint, device)
            processor = AutoProcessor.from_pretrained(self.checkpoint, cache_dir=str(self.cache_dir))
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.checkpoint, cache_dir=str(self.cache_dir))
            self.processor = processor
            self.model = model.to(device).eval()
            self.device = device
        except Exception as exc:  # noqa: BLE001 - convert every failure to a readable message
            self.model = self.processor = None
            hint = "Check your internet connection (needed for the first download) and that 'transformers' is installed."
            if is_cuda_oom(exc):
                hint = "GPU ran out of memory while loading; try the CPU device."
            raise ModelLoadError(f"Could not load Grounding DINO '{self.checkpoint}': {exc}. {hint}") from exc

    def _post_process(self, outputs, input_ids, conf: float, text_threshold: float, size: tuple[int, int]):
        """Call post_process_grounded_object_detection with whichever argument names this version has."""
        fn = self.processor.post_process_grounded_object_detection
        params = inspect.signature(fn).parameters
        kwargs: dict = {"target_sizes": [size]}
        if "threshold" in params:
            kwargs["threshold"] = conf
        elif "box_threshold" in params:
            kwargs["box_threshold"] = conf
        else:
            raise DetectionError("Unsupported transformers version: no box threshold argument found.")
        if "text_threshold" in params:
            kwargs["text_threshold"] = text_threshold
        return fn(outputs, input_ids, **kwargs)

    def detect(self, image_rgb: np.ndarray, prompts: list[str], conf: float = 0.35,
               text_threshold: float = 0.25, **_ignored) -> dict:
        """Run open-vocabulary detection. conf = box threshold. Boxes are in image_rgb coordinates."""
        if not self.is_loaded:
            self.load("auto")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise DetectionError("Expected an RGB image of shape (H, W, 3).")
        import torch

        height, width = image_rgb.shape[:2]
        text = prepare_for_grounding_dino(prompts)
        try:
            t_pre = time.perf_counter()
            inputs = self.processor(images=Image.fromarray(image_rgb), text=text, return_tensors="pt")
            inputs = inputs.to(self.device)
            preprocess_time = time.perf_counter() - t_pre

            cuda_sync(self.device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                outputs = self.model(**inputs)
                results = self._post_process(outputs, inputs["input_ids"], conf, text_threshold, (height, width))
            cuda_sync(self.device)
            inference_time = time.perf_counter() - t0
        except RuntimeError as exc:
            if is_cuda_oom(exc):
                torch.cuda.empty_cache()
                raise DetectionError("GPU out of memory. Use a smaller image resolution or the CPU device.") from exc
            raise DetectionError(f"Grounding DINO inference failed: {exc}") from exc

        res = results[0]
        label_key = "text_labels" if "text_labels" in res else "labels"
        boxes = res["boxes"].detach().cpu().numpy()
        scores = res["scores"].detach().cpu().numpy()

        kept_boxes, kept_labels, kept_scores = [], [], []
        for box, score, raw in zip(boxes, scores, res[label_key]):
            if not isinstance(raw, str) and hasattr(raw, "item") or isinstance(raw, (int, np.integer)):
                idx = int(raw)
                text_label = prompts[idx] if 0 <= idx < len(prompts) else ""
            else:
                text_label = str(raw)
            text_label = text_label.strip()
            if not text_label:
                continue  # no prompt tokens matched above the text threshold
            kept_boxes.append(box)
            kept_labels.append(match_label_to_prompt(text_label, prompts))
            kept_scores.append(score)

        return build_result(self.name, kept_boxes, kept_labels, kept_scores, inference_time,
                            (height, width), preprocess_time=preprocess_time, device=self.device)
