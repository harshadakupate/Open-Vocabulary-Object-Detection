import numpy as np
import pandas as pd

from benchmarking.pdf_report import create_pdf_report
from benchmarking.report_generator import build_category_metric_tables


def test_create_pdf_report_renders_tables_and_images():
    report = create_pdf_report(
        "Detection report",
        {"Model": "YOLO-World", "Detections": 1},
        {"Results": pd.DataFrame([{"label": "person", "confidence": 0.9}] * 27)},
        {"Annotated image": np.zeros((8, 8, 3), dtype=np.uint8)},
        {"Detections by category": ("bar", pd.DataFrame({"Detections": [2, 1]}, index=["person", "helmet"]),
                                   "Detections")},
    )

    assert report.startswith(b"%PDF")
    assert len(report) > 1000


def test_category_metric_tables_include_model_values_and_uncommon_f1():
    per_category = pd.DataFrame([
        {"model": "Grounding DINO", "category": "person", "precision": 0.8, "recall": 0.6, "f1": 0.685},
        {"model": "YOLO-World", "category": "person", "precision": 0.9, "recall": 0.5, "f1": 0.643},
        {"model": "Grounding DINO", "category": "ambulance", "precision": float("nan"), "recall": 0.0, "f1": 0.0},
        {"model": "YOLO-World", "category": "ambulance", "precision": 0.5, "recall": 1.0, "f1": 0.667},
    ])

    tables = build_category_metric_tables(per_category, ["Ambulance"])

    assert tables["Per-category precision and recall"].iloc[0].to_dict() == {
        "Category": "Person",
        "Grounding DINO precision": "0.800",
        "Grounding DINO recall": "0.600",
        "YOLO-World precision": "0.900",
        "YOLO-World recall": "0.500",
    }
    assert tables["Uncommon category F1"].iloc[0].to_dict() == {
        "Category": "Ambulance",
        "Grounding DINO F1": "0.000",
        "YOLO-World F1": "0.667",
    }
