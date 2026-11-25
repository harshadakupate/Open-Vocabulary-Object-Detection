"""Real YOLO-World inference (downloads weights on first run)."""
import pytest

from models import ModelLoadError
from models.model_manager import ModelManager
from tests.helpers import assert_result_contract, synthetic_image


@pytest.fixture(scope="module")
def manager():
    return ModelManager()


def test_yolo_world_result_contract_and_custom_classes(manager):
    try:
        detector = manager.get_detector("YOLO-World", "auto")
    except ModelLoadError as exc:
        pytest.skip(f"YOLO-World unavailable: {exc}")
    image = synthetic_image()
    result = manager.detect("YOLO-World", image, ["person", "safety helmet"], conf=0.05)
    assert_result_contract(result, image, "YOLO-World")
    assert detector.current_classes == ("person", "safety helmet")
    assert set(result["labels"]) <= {"person", "safety helmet"}


def test_yolo_world_vocabulary_can_change(manager):
    try:
        detector = manager.get_detector("YOLO-World", "auto")
    except ModelLoadError as exc:
        pytest.skip(str(exc))
    image = synthetic_image()
    manager.detect("YOLO-World", image, ["laptop"], conf=0.05)
    assert detector.current_classes == ("laptop",)
    manager.detect("YOLO-World", image, ["gas cylinder", "ear protection"], conf=0.05)
    assert detector.current_classes == ("gas cylinder", "ear protection")
