import pytest

from detection.prompt_manager import (PromptError, match_label_to_prompt, parse_prompts,
                                      prepare_for_grounding_dino, prepare_for_yolo_world)


def test_parse_commas_newlines_duplicates_whitespace():
    raw = "person,  Safety   helmet\nperson\n\n safety helmet ,fire extinguisher;"
    assert parse_prompts(raw) == ["person", "Safety helmet", "fire extinguisher"]


def test_parse_list_input():
    assert parse_prompts(["a", " b ", "A", ""]) == ["a", "b"]


def test_grounding_dino_format():
    assert prepare_for_grounding_dino(["Person", "Safety Helmet"]) == "person . safety helmet ."


def test_yolo_world_format():
    assert prepare_for_yolo_world("person, laptop") == ["person", "laptop"]


def test_empty_prompts_raise():
    with pytest.raises(PromptError):
        prepare_for_grounding_dino([" ", ""])


def test_match_label_to_prompt():
    prompts = ["safety helmet", "safety vest", "person"]
    assert match_label_to_prompt("Person", prompts) == "person"
    assert match_label_to_prompt("safety vest .", prompts) == "safety vest"
    assert match_label_to_prompt("unknown thing", prompts) == "unknown thing"
