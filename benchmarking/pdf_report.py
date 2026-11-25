"""Create downloadable PDF reports for detection and benchmark results."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


def _format_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "N/A"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def create_pdf_report(title: str, details: dict[str, object],
                      tables: dict[str, pd.DataFrame] | None = None,
                      images: dict[str, np.ndarray | str | Path] | None = None,
                      charts: dict[str, tuple[str, pd.DataFrame, str]] | None = None) -> bytes:
    """Render summary, chart, table, and image pages into a downloadable PDF."""
    output = BytesIO()
    with PdfPages(output) as pdf:
        detail_items = list(details.items()) or [("Status", "No summary details available")]
        for page_no, start in enumerate(range(0, len(detail_items), 14)):
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.suptitle(title, fontsize=18, fontweight="bold", x=0.08, ha="left", y=0.94)
            fig.text(0.08, 0.88, "Processing report", fontsize=10, color="#0f766e")
            y = 0.82
            for key, value in detail_items[start:start + 14]:
                fig.text(0.08, y, f"{key}: {_format_value(value)}", fontsize=10, va="top", wrap=True)
                y -= 0.052
            if page_no:
                fig.text(0.92, 0.04, f"Summary page {page_no + 1}", fontsize=8, ha="right", color="#64748b")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for caption, (kind, frame, ylabel) in (charts or {}).items():
            if frame.empty:
                continue
            if kind not in {"bar", "line"}:
                raise ValueError(f"Unsupported PDF chart type: {kind}")
            chart_data = frame.apply(pd.to_numeric, errors="coerce")
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            plot_options = {"marker": "o"} if kind == "line" else {}
            chart_data.plot(kind=kind, ax=ax, legend=len(chart_data.columns) > 1, **plot_options)
            ax.set_title(caption, loc="left", fontsize=15, fontweight="bold", pad=18)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", labelrotation=40 if len(chart_data.index) > 8 else 0)
            if kind == "bar":
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.2g", padding=2, fontsize=7)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        for section, frame in (tables or {}).items():
            if frame.empty:
                continue
            page_size = 24
            for start in range(0, len(frame), page_size):
                page = frame.iloc[start:start + page_size].fillna("").astype(str)
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.axis("off")
                ax.set_title(section, loc="left", fontsize=15, fontweight="bold", pad=18)
                table = ax.table(cellText=page.values, colLabels=page.columns, loc="center",
                                 cellLoc="left", colLoc="left")
                table.auto_set_font_size(False)
                table.set_fontsize(7 if len(page.columns) > 7 else 8)
                table.scale(1, 1.25)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        for caption, image in (images or {}).items():
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.imshow(plt.imread(str(image)) if isinstance(image, (str, Path)) else image)
            ax.set_title(caption, loc="left", fontsize=15, fontweight="bold", pad=18)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return output.getvalue()
