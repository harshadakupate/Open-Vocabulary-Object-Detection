"""Build comparison tables, CSV files, charts and a Markdown report from evaluation summaries."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detection.prompt_manager import load_test_prompts  # noqa: E402
from utils.config_loader import load_config, resolve_path  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("report_generator")
MODELS = ["Grounding DINO", "YOLO-World"]
SLUGS = {"Grounding DINO": "grounding_dino", "YOLO-World": "yolo_world"}
NOT_EVALUATED = "Not evaluated"

# (row label, getter path) - getter looks into summary["overall"] or summary["performance"]
TABLE_ROWS = [
    ("Precision", "overall", "precision"),
    ("Recall", "overall", "recall"),
    ("F1 Score", "overall", "f1"),
    ("mAP@50", "overall", "map50"),
    ("mAP@50-95", "overall", "map50_95"),
    ("Average inference time (ms)", "performance", "avg_inference_ms"),
    ("Inference FPS", "performance", "inference_fps"),
    ("End-to-end FPS", "performance", "end_to_end_fps"),
    ("Peak GPU memory (MB)", "performance", "gpu_peak_allocated_mb"),
    ("Peak RAM (MB)", "performance", "ram_peak_mb"),
    ("Average confidence", "overall", "avg_confidence"),
]


def reports_dir() -> Path:
    path = resolve_path(load_config()["paths"]["reports_out"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summaries(directory: Path | None = None) -> dict[str, dict]:
    directory = directory or reports_dir()
    found = {}
    for model in MODELS:
        path = directory / f"summary_{SLUGS[model]}.json"
        if path.exists():
            found[model] = json.loads(path.read_text(encoding="utf-8"))
    return found


def _value(summary: dict | None, section: str, key: str):
    if not summary:
        return None
    return summary.get(section, {}).get(key)


def build_comparison_table(summaries: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for label, section, key in TABLE_ROWS:
        row = {"Metric": label}
        for model in MODELS:
            val = _value(summaries.get(model), section, key)
            row[model] = NOT_EVALUATED if val is None else round(float(val), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_category_table(summaries: dict[str, dict]) -> pd.DataFrame:
    frames = []
    for model, summary in summaries.items():
        df = pd.DataFrame(summary.get("per_class", []))
        if df.empty:
            continue
        df.insert(0, "model", model)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_category_metric_tables(per_category: pd.DataFrame,
                                 uncommon_categories: list[str]) -> dict[str, pd.DataFrame]:
    """Build readable per-category precision/recall and uncommon F1 tables."""
    models_by_category = {
        (str(row["category"]).strip().lower(), str(row["model"])): row
        for _, row in per_category.iterrows()
    }
    categories = list(dict.fromkeys(per_category.get("category", pd.Series(dtype=str)).astype(str)))

    def metric_value(category: str, model: str, metric: str) -> str:
        row = models_by_category.get((category.lower(), model))
        value = row.get(metric) if row is not None else None
        return "N/A" if value is None or pd.isna(value) else f"{float(value):.3f}"

    precision_recall = [
        {
            "Category": category.title(),
            **{
                f"{model} {metric}": metric_value(category, model, metric)
                for model in MODELS
                for metric in ("precision", "recall")
            },
        }
        for category in categories
    ]
    uncommon = {category.strip().lower() for category in uncommon_categories}
    uncommon_f1 = [
        {
            "Category": category.title(),
            **{f"{model} F1": metric_value(category, model, "f1") for model in MODELS},
        }
        for category in categories if category.lower() in uncommon
    ]
    return {
        "Per-category precision and recall": pd.DataFrame(precision_recall),
        "Uncommon category F1": pd.DataFrame(uncommon_f1),
    }


def _num(value) -> float:
    return float("nan") if value is None else float(value)


def _grouped_bar(path: Path, title: str, ylabel: str, labels: list[str], values: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(labels) + 3), 4.2))
    width = 0.8 / max(1, len(values))
    x = np.arange(len(labels))
    for i, (model, vals) in enumerate(values.items()):
        arr = np.array([0.0 if (v is None or math.isnan(v)) else v for v in vals])
        bars = ax.bar(x + i * width - 0.4 + width / 2, arr, width, label=model)
        for bar, raw in zip(bars, vals):
            txt = "n/a" if (raw is None or math.isnan(raw)) else f"{raw:.2f}"
            ax.annotate(txt, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _metric_chart(path, title, ylabel, summaries, items):
    labels = [lbl for lbl, _, _ in items]
    values = {m: [_num(_value(summaries.get(m), sec, key)) for _, sec, key in items] for m in summaries}
    _grouped_bar(path, title, ylabel, labels, values)


def generate_report(directory: Path | None = None) -> dict | None:
    """Create CSVs, charts and report.md from the saved summaries. Returns paths, or None if no data."""
    directory = directory or reports_dir()
    summaries = load_summaries(directory)
    if not summaries:
        logger.warning("No evaluation summaries found in %s - evaluation has not been completed.", directory)
        return None

    outputs: dict[str, str] = {}
    table = build_comparison_table(summaries)
    table.to_csv(directory / "comparison_table.csv", index=False)
    outputs["comparison_table"] = str(directory / "comparison_table.csv")
    per_cat = build_per_category_table(summaries)
    if not per_cat.empty:
        per_cat.to_csv(directory / "per_category_comparison.csv", index=False)
        outputs["per_category"] = str(directory / "per_category_comparison.csv")

    charts = {
        "accuracy.png": ("Accuracy metrics (IoU 0.50 matching)", "score",
                         [("Precision", "overall", "precision"), ("Recall", "overall", "recall"),
                          ("F1", "overall", "f1"), ("mAP@50", "overall", "map50"),
                          ("mAP@50-95", "overall", "map50_95")]),
        "inference_time.png": ("Average inference time", "ms / image",
                               [("avg", "performance", "avg_inference_ms"),
                                ("min", "performance", "min_inference_ms"),
                                ("max", "performance", "max_inference_ms")]),
        "fps.png": ("Throughput", "frames (images) per second",
                    [("Inference FPS", "performance", "inference_fps"),
                     ("End-to-end FPS", "performance", "end_to_end_fps")]),
        "gpu_memory.png": ("GPU memory (PyTorch allocated)", "MB",
                           [("Allocated (end)", "performance", "gpu_allocated_mb"),
                            ("Peak allocated", "performance", "gpu_peak_allocated_mb"),
                            ("Total GPU", "performance", "gpu_total_mb")]),
        "ram.png": ("Process RAM", "MB",
                    [("Initial", "performance", "ram_initial_mb"), ("Peak", "performance", "ram_peak_mb"),
                     ("Final", "performance", "ram_final_mb")]),
    }
    for fname, (title, ylabel, items) in charts.items():
        _metric_chart(directory / fname, title, ylabel, summaries, items)
        outputs[fname] = str(directory / fname)

    if not per_cat.empty:
        cats = list(dict.fromkeys(per_cat["category"]))
        for metric in ("precision", "recall"):
            vals = {}
            for model in summaries:
                sub = per_cat[per_cat["model"] == model].set_index("category")
                vals[model] = [_num(sub[metric].get(c)) if c in sub.index else float("nan") for c in cats]
            _grouped_bar(directory / f"per_category_{metric}.png", f"Per-category {metric}", metric, cats, vals)
            outputs[f"per_category_{metric}.png"] = str(directory / f"per_category_{metric}.png")

        uncommon = [u.lower() for u in load_test_prompts().get("uncommon", [])]
        unc = [c for c in cats if c.lower() in uncommon]
        if unc:
            vals = {}
            for model in summaries:
                sub = per_cat[per_cat["model"] == model].set_index("category")
                vals[model] = [_num(sub["f1"].get(c)) if c in sub.index else float("nan") for c in unc]
            _grouped_bar(directory / "uncommon_categories.png",
                         "Uncommon categories - F1 (rare in typical datasets; not a claim about pretraining data)",
                         "F1", unc, vals)
            outputs["uncommon_categories.png"] = str(directory / "uncommon_categories.png")

    lines = ["# Open-Vocabulary Detection - Benchmark Report", ""]
    for model, s in summaries.items():
        lines.append(f"- **{model}**: device `{s['device']}`, {s['num_images_evaluated']} images, "
                     f"operating confidence {s['operating_conf']}, run {s['timestamp']}")
    lines += ["", "## Comparison", "", table.to_markdown(index=False) if _has_tabulate() else table.to_string(index=False), "",
              "Missing models are shown as 'Not evaluated'. Confidence is not accuracy. "
              "Both models were evaluated in separate processes on the same hardware, images and annotations.", ""]
    (directory / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    outputs["report"] = str(directory / "benchmark_report.md")
    return outputs


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
