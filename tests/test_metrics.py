import math

import numpy as np

from benchmarking.metrics import average_precision, evaluate_detections, iou_xyxy, pair_boxes


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    assert iou_xyxy(a, a)[0, 0] == 1.0
    assert iou_xyxy(a, np.array([[20, 20, 30, 30]]))[0, 0] == 0.0
    assert abs(iou_xyxy(a, np.array([[5, 0, 15, 10]]))[0, 0] - (50 / 150)) < 1e-9


def test_perfect_detection():
    gt = {"i1": {"boxes": [[0, 0, 10, 10]], "labels": ["cat"]}}
    pr = {"i1": {"boxes": [[0, 0, 10, 10]], "labels": ["cat"], "scores": [0.9]}}
    o = evaluate_detections(pr, gt, ["cat"], 0.5, 0.25)["overall"]
    assert (o["precision"], o["recall"], o["f1"]) == (1.0, 1.0, 1.0)
    assert o["map50"] == 1.0 and o["map50_95"] == 1.0


def test_duplicate_prediction_matches_once():
    gt = {"i1": {"boxes": [[0, 0, 10, 10]], "labels": ["cat"]}}
    pr = {"i1": {"boxes": [[0, 0, 10, 10], [0, 0, 10, 10]], "labels": ["cat", "cat"], "scores": [0.9, 0.8]}}
    o = evaluate_detections(pr, gt, ["cat"], 0.5, 0.25)["overall"]
    assert (o["tp"], o["fp"], o["fn"]) == (1, 1, 0)
    assert o["precision"] == 0.5 and o["recall"] == 1.0


def test_missed_object_and_wrong_class():
    gt = {"i1": {"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "labels": ["cat", "dog"]}}
    pr = {"i1": {"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]], "labels": ["cat", "cat"], "scores": [0.9, 0.9]}}
    res = evaluate_detections(pr, gt, ["cat", "dog"], 0.5, 0.25)
    rows = {r["category"]: r for r in res["per_class"]}
    assert rows["cat"]["tp"] == 1 and rows["cat"]["fp"] == 1
    assert rows["dog"]["fn"] == 1 and rows["dog"]["recall"] == 0.0


def test_confidence_threshold_filters_pr_but_not_ap():
    gt = {"i1": {"boxes": [[0, 0, 10, 10]], "labels": ["cat"]}}
    pr = {"i1": {"boxes": [[0, 0, 10, 10]], "labels": ["cat"], "scores": [0.1]}}
    o = evaluate_detections(pr, gt, ["cat"], 0.5, 0.25)["overall"]
    assert o["recall"] == 0.0 and o["map50"] == 1.0  # AP sees the low-score detection


def test_undefined_precision_is_nan_not_zero():
    gt = {"i1": {"boxes": [[0, 0, 10, 10]], "labels": ["cat"]}}
    o = evaluate_detections({}, gt, ["cat"], 0.5, 0.25)["overall"]
    assert math.isnan(o["precision"]) and o["recall"] == 0.0 and o["map50"] == 0.0


def test_average_precision_ranking():
    # TP, FP, TP with 2 GT -> precision envelope 1.0 up to recall .5, then 2/3
    ap = average_precision([(0.9, True), (0.8, False), (0.7, True)], 2)
    assert 0.8 < ap < 0.9


def test_pair_boxes_one_to_one():
    a = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    b = np.array([[0, 0, 10, 10]])
    assert pair_boxes(a, b, 0.5) == [1.0]
