"""Real Grounding DINO inference (downloads weights on first run)."""
import pytest

from models import ModelLoadError
from models.model_manager import ModelManager
from tests.helpers import assert_result_contract, synthetic_image


@pytest.fixture(scope="module")
def manager():
    return ModelManager()


def test_grounding_dino_result_contract(manager):
    try:
        manager.get_detector("Grounding DINO", "auto")
    except ModelLoadError as exc:
        pytest.skip(f"Grounding DINO unavailable: {exc}")
    image = synthetic_image()
    result = manager.detect("Grounding DINO", image, ["person", "safety helmet", "red box"], conf=0.2)
    assert_result_contract(result, image, "Grounding DINO")


def test_grounding_dino_prompt_change_does_not_crash(manager):
    try:
        manager.get_detector("Grounding DINO", "auto")
    except ModelLoadError as exc:
        pytest.skip(str(exc))
    image = synthetic_image()
    for prompts in (["laptop"], ["gas cylinder", "fire blanket"]):
        assert_result_contract(manager.detect("Grounding DINO", image, prompts, conf=0.2), image, "Grounding DINO")
