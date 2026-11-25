"""End-to-end image pipeline with a real model (YOLO-World, the faster one)."""
import pytest

from detection.image_detection import run_image_detection
from detection.visualization import draw_detections, result_to_dataframe, save_annotated_image
from models import ModelLoadError
from models.model_manager import ModelManager
from tests.helpers import assert_result_contract, synthetic_image


def test_image_pipeline_original_coordinates_and_saving(tmp_path):
    manager = ModelManager()
    try:
        manager.get_detector("YOLO-World", "auto")
    except ModelLoadError as exc:
        pytest.skip(str(exc))
    image = synthetic_image(900, 1400)
    result = run_image_detection(manager, "YOLO-World", image, ["person", "laptop"], conf=0.05, max_side=640)
    assert_result_contract(result, image, "YOLO-World")  # boxes within the ORIGINAL 1400x900 frame
    annotated = draw_detections(image, result)
    assert annotated.shape == image.shape
    assert len(result_to_dataframe(result)) == result["num_detections"]
    assert save_annotated_image(annotated, tmp_path, "t").exists()
