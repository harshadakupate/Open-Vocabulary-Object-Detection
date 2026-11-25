"""End-to-end video pipeline with a real model."""
import pytest

from detection.video_detection import process_video
from models import ModelLoadError
from models.model_manager import ModelManager
from tests.helpers import make_video


def test_video_pipeline_real_model(tmp_path):
    manager = ModelManager()
    try:
        manager.get_detector("YOLO-World", "auto")
    except ModelLoadError as exc:
        pytest.skip(str(exc))
    src, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(src, frames=8, fps=8)
    stats = process_video(manager, "YOLO-World", src, out, ["person", "laptop"], conf=0.05, frame_skip=2)
    assert stats["processed_frames"] == 4
    assert out.exists() and out.stat().st_size > 0
    assert stats["inference_fps"] > 0 and stats["end_to_end_fps"] > 0
    assert stats["end_to_end_fps"] <= stats["inference_fps"] + 1e-6  # e2e includes extra work
