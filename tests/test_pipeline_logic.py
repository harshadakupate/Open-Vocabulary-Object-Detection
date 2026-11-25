"""Pipeline logic tests with a STUB manager (no model weights needed).

These verify box rescaling, frame skipping, FPS definitions and video writing only.
Real inference is covered by the model tests.
"""
import numpy as np

from detection.image_detection import run_image_detection
from detection.video_detection import process_video
from models import build_result
from tests.helpers import make_video, synthetic_image


class StubManager:
    """Returns one fixed box (in the coordinates of the image it receives)."""

    def __init__(self):
        self.calls = 0

    def detect(self, model_name, image, prompts, device="auto", conf=None, text_threshold=None, imgsz=None):
        self.calls += 1
        h, w = image.shape[:2]
        return build_result(model_name, [[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], [prompts[0]], [0.9], 0.01, (h, w))


def test_boxes_mapped_back_to_original_coordinates():
    image = synthetic_image(1200, 1600)
    res = run_image_detection(StubManager(), "YOLO-World", image, ["thing"], max_side=800)
    assert (res["image_width"], res["image_height"]) == (1600, 1200)
    x1, y1, x2, y2 = res["boxes"][0]
    assert abs(x1 - 400) < 2 and abs(x2 - 1200) < 2 and abs(y1 - 300) < 2 and abs(y2 - 900) < 2


def test_video_frame_skip_and_fps_definitions(tmp_path):
    src, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src, frames=10, fps=10)
    stats = process_video(StubManager(), "YOLO-World", src, out, ["thing"], frame_skip=2, warmup=False)
    assert stats["processed_frames"] == 5 and stats["frame_skip_enabled"]
    assert out.exists() and out.stat().st_size > 0
    assert abs(stats["avg_inference_ms"] - 10.0) < 1e-6          # stub reports 0.01 s per frame
    assert abs(stats["inference_fps"] - 100.0) < 1e-6           # 5 frames / 0.05 s of model time
    assert stats["input_fps"] == 10.0 and stats["inference_fps"] != stats["input_fps"]
    assert stats["end_to_end_fps"] > 0 and "Frame skipping enabled" in stats["note"]
