"""Open-Vocabulary Object Detection - Streamlit dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from benchmarking.metrics import pair_boxes
from benchmarking.performance_monitor import PerformanceMonitor
from benchmarking.pdf_report import create_pdf_report
from benchmarking.report_generator import (
    NOT_EVALUATED,
    build_category_metric_tables,
    generate_report,
    load_summaries,
    reports_dir,
)
from detection.image_detection import run_image_detection
from detection.prompt_manager import (
    PromptError,
    get_all_presets,
    load_test_prompts,
    parse_prompts,
    save_prompt_preset,
)
from detection.video_detection import process_video
from detection.visualization import draw_detections, result_to_dataframe, save_annotated_image
from models import DetectionError, ModelLoadError
from models.model_manager import MODEL_NAMES, ModelManager
from utils.config_loader import PROJECT_ROOT, ensure_project_dirs, load_config, resolve_path
from utils.device import device_info, resolve_device
from utils.image_utils import ImageLoadError, load_image_rgb, validate_extension
from utils.logger import get_logger
from utils.video_utils import VideoError, get_video_info

logger = get_logger("app")
CFG = load_config()
ensure_project_dirs()
KNOWN_ERRORS = (ImageLoadError, VideoError, PromptError, ModelLoadError, DetectionError, ValueError, FileNotFoundError)
DEFAULT_PROMPTS = (
    "person\nsafety helmet\nsafety vest\nfire extinguisher\nmobile phone\n"
    "laptop\ntraffic cone\ntoolbox\nwelding machine\nindustrial safety equipment\n"
    "Vending Machine\nAmbulance\nTelescope\nStretcher\nSuitcase"
)

st.set_page_config(page_title=CFG["app"]["name"], page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 86% 0%, rgba(20, 184, 166, .10), transparent 34rem),
            linear-gradient(180deg, rgba(15, 23, 42, .025), transparent 28rem);
    }
    .block-container {padding-top: 1.2rem; max-width: 1540px; padding-bottom: 3rem;}
    [data-testid="stHeader"] {background: transparent;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,.18);}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {letter-spacing: -.02em;}
    .ovd-hero {
        isolation: isolate; position: relative; overflow: hidden; min-height: 225px;
        display: flex; align-items: center; padding: 2rem 2.3rem; margin: .2rem 0 1.5rem;
        border: 1px solid rgba(148,163,184,.2); border-radius: 24px;
        background: linear-gradient(112deg, #0b1725 0%, #103a43 55%, #0f766e 100%);
        box-shadow: 0 20px 55px rgba(2, 12, 27, .22);
    }
    .ovd-hero:before, .ovd-hero:after {
        content: ""; position: absolute; z-index: -1; border-radius: 50%; pointer-events: none;
    }
    .ovd-hero:before {
        width: 340px; height: 340px; right: 3%; top: -225px;
        border: 1px solid rgba(153,246,228,.22);
        box-shadow: 0 0 0 28px rgba(153,246,228,.035), 0 0 0 58px rgba(153,246,228,.025);
    }
    .ovd-hero:after {
        width: 250px; height: 250px; right: 10%; bottom: -205px;
        background: rgba(45,212,191,.18); filter: blur(35px);
    }
    .ovd-hero-copy {max-width: 820px;}
    .ovd-eyebrow {
        display: flex; align-items: center; gap: .55rem; margin-bottom: .65rem;
        color: #99f6e4; font-size: .72rem; font-weight: 750; letter-spacing: .16em;
    }
    .ovd-pulse {
        width: 8px; height: 8px; border-radius: 50%; background: #5eead4;
        box-shadow: 0 0 14px rgba(94,234,212,.9);
    }
    .ovd-hero h1 {
        margin: 0; color: #f8fafc; font-size: clamp(2rem, 4vw, 3.15rem);
        font-weight: 760; line-height: 1.08; letter-spacing: -.045em;
    }
    .ovd-hero h1 span {color: #5eead4;}
    .ovd-hero p {max-width: 740px; margin: .7rem 0 1rem; color: #d1dbe6; font-size: 1rem;}
    .ovd-tags {display: flex; flex-wrap: wrap; gap: .5rem;}
    .ovd-tags span {
        padding: .28rem .65rem; border: 1px solid rgba(204,251,241,.22); border-radius: 999px;
        background: rgba(15,23,42,.22); color: #e2e8f0; font-size: .74rem; font-weight: 600;
    }
    .ovd-quick-card {
        height: 100%; min-height: 145px; padding: 1.1rem 1.15rem; border-radius: 16px;
        border: 1px solid rgba(148,163,184,.2);
        background: linear-gradient(145deg, rgba(148,163,184,.09), rgba(148,163,184,.025));
        box-shadow: 0 8px 24px rgba(15,23,42,.045); transition: transform .18s ease, border-color .18s ease;
    }
    .ovd-quick-card:hover {transform: translateY(-3px); border-color: rgba(20,184,166,.48);}
    .ovd-step {
        display: inline-block; margin-bottom: .65rem; padding: .18rem .48rem; border-radius: 7px;
        background: rgba(20,184,166,.13); color: #0f766e; font-size: .68rem;
        font-weight: 800; letter-spacing: .08em;
    }
    .ovd-quick-card h4 {margin: 0 0 .35rem; font-size: 1rem; letter-spacing: -.015em;}
    .ovd-quick-card p {margin: 0; color: rgba(100,116,139,.95); font-size: .88rem; line-height: 1.5;}
    .stTabs [data-baseweb="tab-list"] {
        gap: .35rem; padding: .4rem; border: 1px solid rgba(148,163,184,.18);
        border-radius: 15px; background: rgba(148,163,184,.055);
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.8rem; padding: 0 1rem; border-radius: 10px; transition: background .18s ease;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 0; background: rgba(20,184,166,.14); box-shadow: inset 0 0 0 1px rgba(20,184,166,.22);
    }
    div[data-testid="stMetric"] {
        min-height: 106px; padding: .9rem 1rem; border: 1px solid rgba(148,163,184,.2);
        border-radius: 16px; background: linear-gradient(145deg, rgba(148,163,184,.10), rgba(148,163,184,.035));
        box-shadow: 0 8px 22px rgba(15,23,42,.06); transition: transform .18s ease, box-shadow .18s ease;
    }
    div[data-testid="stMetric"]:hover {transform: translateY(-2px); box-shadow: 0 12px 28px rgba(15,23,42,.10);}
    div[data-testid="stMetricLabel"] {font-size: .8rem; font-weight: 650;}
    div[data-testid="stMetricValue"] {font-size: 1.65rem; font-weight: 720;}
    .stButton > button, .stDownloadButton > button {border-radius: 10px; font-weight: 650; transition: transform .15s ease;}
    .stButton > button:hover, .stDownloadButton > button:hover {transform: translateY(-1px);}
    .stButton > button[kind="primary"] {
        border: 0; background: linear-gradient(100deg, #0f766e, #0d9488); color: white;
        box-shadow: 0 6px 16px rgba(13,148,136,.2);
    }
    div[data-testid="stFileUploader"] {
        padding: .25rem; border: 1px dashed rgba(20,184,166,.32); border-radius: 14px;
        background: rgba(20,184,166,.025);
    }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {border: 1px solid rgba(148,163,184,.16); border-radius: 12px;}
    div[data-testid="stExpander"] {border-color: rgba(148,163,184,.22); border-radius: 12px;}
    div[data-testid="stAlert"] {border-radius: 12px;}
    h2, h3 {letter-spacing: -.025em;}
    @media (max-width: 700px) {
        .ovd-hero {min-height: 195px; padding: 1.4rem; border-radius: 18px;}
        .ovd-hero h1 {font-size: 2rem;}
        .ovd-hero p {font-size: .9rem;}
        .stTabs [data-baseweb="tab"] {padding: 0 .55rem;}
    }
    </style>
    <section class="ovd-hero" aria-label="Open-vocabulary object detection workspace">
      <div class="ovd-hero-copy">
        <div class="ovd-eyebrow"><span class="ovd-pulse"></span> AI VISION WORKSPACE</div>
        <h1>Open-Vocabulary<br><span>Object Detection</span></h1>
        <p>Explore images and video with flexible, natural-language prompts. Compare two powerful zero-shot models without training or fine-tuning.</p>
        <div class="ovd-tags"><span>Grounding DINO</span><span>YOLO-World</span><span>Custom prompts</span><span>Image · Video · Benchmarks</span></div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_manager() -> ModelManager:
    """One manager for the whole server: model weights stay in memory across reruns/sessions."""
    return ModelManager()


manager = get_manager()


# ------------------------------------------------------------------ helpers
def show_error(exc: Exception) -> None:
    if isinstance(exc, KNOWN_ERRORS):
        st.error(str(exc))
    else:
        logger.exception("Unexpected error")
        st.error(f"Unexpected error: {exc}")


def png_bytes(image_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


def _load_preset(key: str) -> None:
    choice = st.session_state.get(f"{key}_preset")
    presets = get_all_presets()
    if choice in presets:
        st.session_state[f"{key}_text"] = "\n".join(presets[choice])


def prompt_input(key: str) -> list[str]:
    """Prompt editor with presets. Edits take effect immediately (no restart needed)."""
    st.session_state.setdefault(f"{key}_text", DEFAULT_PROMPTS)
    presets = get_all_presets()
    st.selectbox("Load a prompt set", ["(keep current text)"] + list(presets), key=f"{key}_preset",
                 on_change=_load_preset, args=(key,))
    st.text_area("Object categories (one per line or comma-separated)", key=f"{key}_text", height=150,
                 help="Examples: person, safety helmet, fire extinguisher, Vending Machine")
    prompts = parse_prompts(st.session_state[f"{key}_text"])
    c1, c2 = st.columns([3, 1])
    name = c1.text_input("Save as prompt set", key=f"{key}_save_name", placeholder="e.g. workshop PPE",
                         label_visibility="collapsed")
    if c2.button("Save set", key=f"{key}_save_btn", use_container_width=True):
        try:
            save_prompt_preset(name, prompts)
            st.success(f"Saved '{name}'.")
        except PromptError as exc:
            st.warning(str(exc))
    st.caption(f"{len(prompts)} categories after cleaning (duplicates and blanks removed).")
    return prompts


def record_run(kind: str, filename: str, result: dict) -> None:
    """Keep lightweight inference metadata for the current Streamlit session."""
    history = st.session_state.setdefault("run_history", [])
    history.insert(0, {
        "time": time.strftime("%H:%M:%S"),
        "type": kind,
        "file": filename,
        "model": result["model_name"],
        "detections": result["num_detections"],
        "inference_ms": round(result["inference_time"] * 1000, 1),
        "device": str(result.get("device", "-")).upper(),
    })
    del history[100:]


def run_with_performance(callback, device: str) -> tuple[dict, dict]:
    """Run one inference while sampling its latency and system resource usage."""
    monitor = PerformanceMonitor(device, interval=float(CFG["benchmark"]["monitor_interval_s"]))
    monitor.start()
    started = time.perf_counter()
    try:
        result = callback()
        monitor.record_inference(result["inference_time"])
        monitor.record_item(time.perf_counter() - started)
    finally:
        monitor.stop()
    return result, monitor.summary()


def _performance_number(performance: dict, key: str) -> float | None:
    value = performance.get(key)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _format_performance(performance: dict, key: str, unit: str, multiplier: float = 1.0) -> str:
    value = _performance_number(performance, key)
    return "N/A" if value is None else f"{value * multiplier:.1f}{unit}"


def show_runtime_benchmark(performance: dict, time_label: str = "Detection pipeline time") -> None:
    st.markdown("**Runtime benchmark**")
    metrics = st.columns(5)
    metrics[0].metric(time_label, _format_performance(performance, "e2e_total_s", " ms", 1000))
    metrics[1].metric("Peak GPU memory", _format_performance(performance, "gpu_peak_allocated_mb", " MB"))
    metrics[2].metric("Peak GPU utilization", _format_performance(performance, "gpu_util_max_pct", "%"))
    metrics[3].metric("Peak process RAM", _format_performance(performance, "ram_peak_mb", " MB"))
    metrics[4].metric("Peak CPU utilization", _format_performance(performance, "cpu_peak_pct", "%"))
    st.caption("Runtime measurements do not measure accuracy. Precision, recall, and mAP require annotated ground truth.")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Open-Vocabulary Object Detection")
    model = st.selectbox("Model", MODEL_NAMES)
    device_choice = st.selectbox("Device", ["auto", "cuda", "cpu"], index=["auto", "cuda", "cpu"].index(CFG["device"]["default"]))
    resolved = resolve_device(device_choice)
    info = device_info()
    st.caption(f"Running on **{resolved.upper()}**" + (f" - {info.get('gpu_name')}" if resolved == "cuda" else ""))
    if device_choice != "cpu" and not info["cuda_available"]:
        st.warning(
            f"PyTorch ({info['torch']}) cannot access CUDA, so inference is running on CPU. "
            "Install a CUDA-enabled PyTorch build in this environment and restart the app to use the GPU."
        )

    st.subheader("Thresholds")
    is_dino = model == "Grounding DINO"
    conf = st.slider("Detection threshold" + (" (box)" if is_dino else " (confidence)"), 0.01, 0.95,
                     float(manager.default_conf(model)), 0.01, key=f"conf_{model}")
    text_thr = st.slider("Text threshold (Grounding DINO)", 0.01, 0.95, float(CFG["detection"]["dino_text_threshold"]),
                         0.01, disabled=not is_dino)

    st.subheader("Resolution")
    max_side = st.select_slider("Image: max long side before inference (0 = original)",
                                options=[0, 320, 480, 640, 800, 1024, 1280, 1600, 1920, 2560],
                                value=CFG["image"]["inference_max_side"])
    imgsz = st.select_slider("YOLO-World inference size", options=[320, 480, 640, 800, 1024, 1280],
                             value=CFG["image"]["yolo_imgsz"], disabled=is_dino)

    st.subheader("Video")
    frame_skip = st.number_input("Frame skip (1 = every frame)", 1, 60, int(CFG["video"]["frame_skip"]))
    video_side = st.select_slider("Video: resize long side (0 = original)",
                                  options=[0, 320, 480, 640, 800, 1024, 1280, 1920],
                                  value=CFG["video"]["resize_long_side"])
    max_frames = st.number_input("Max frames to process (0 = all)", 0, 100000, int(CFG["video"]["max_frames"]))
    st.caption("First use of a model downloads its weights and runs a warm-up pass.")

tab_home, tab_img, tab_vid, tab_cmp, tab_bench, tab_info = st.tabs(
    ["🏠 Overview", "🖼️ Image Detection", "🎬 Video Detection", "⚖️ Model Comparison",
     "📊 Benchmark Dashboard", "ℹ️ Project Information"])


# ------------------------------------------------------------------ overview tab
with tab_home:
    st.subheader("Workspace overview")
    st.caption("A quick view of your inference activity in this browser session.")
    history = st.session_state.get("run_history", [])
    if history:
        history_df = pd.DataFrame(history)
        total_detections = int(history_df["detections"].sum())
        average_ms = float(history_df["inference_ms"].mean())
        model_count = int(history_df["model"].nunique())
        summary = st.columns(4)
        summary[0].metric("Inference runs", len(history_df))
        summary[1].metric("Detections", total_detections)
        summary[2].metric("Avg recorded inference", f"{average_ms:.1f} ms")
        summary[3].metric("Models used", model_count)

        model_activity, type_activity = st.columns(2)
        with model_activity:
            st.markdown("**Runs by model**")
            st.bar_chart(history_df["model"].value_counts().rename("Runs"))
        with type_activity:
            st.markdown("**Runs by workflow**")
            st.bar_chart(history_df["type"].value_counts().rename("Runs"))

        st.markdown("**Recent activity**")
        st.dataframe(history_df, hide_index=True, use_container_width=True)
        h1, h2 = st.columns([1, 5])
        h1.download_button("Export run history", history_df.to_csv(index=False).encode("utf-8"),
                           file_name="detection_run_history.csv", mime="text/csv")
        if h2.button("Clear session history", key="clear_run_history"):
            st.session_state["run_history"] = []
            st.rerun()
    else:
        st.info("No inference runs yet. Start with an image, process a video, or compare both models.")

    st.markdown("**Quick start**")
    q1, q2, q3 = st.columns(3)
    q1.markdown(
        '<div class="ovd-quick-card"><span class="ovd-step">STEP 01</span>'
        "<h4>Choose a model</h4><p>Select Grounding DINO or YOLO-World and tune thresholds in the sidebar.</p></div>",
        unsafe_allow_html=True,
    )
    q2.markdown(
        '<div class="ovd-quick-card"><span class="ovd-step">STEP 02</span>'
        "<h4>Describe objects</h4><p>Enter categories in natural language or load a reusable prompt set.</p></div>",
        unsafe_allow_html=True,
    )
    q3.markdown(
        '<div class="ovd-quick-card"><span class="ovd-step">STEP 03</span>'
        "<h4>Explore results</h4><p>Inspect detections, compare models, or review benchmark performance.</p></div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ image tab
with tab_img:
    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Upload an image", type=[e.strip(".") for e in CFG["image"]["allowed_extensions"]],
                                    key="img_upload")
    with right:
        img_prompts = prompt_input("img")

    if st.button("Detect", type="primary", disabled=uploaded is None or not img_prompts, key="img_run"):
        try:
            validate_extension(uploaded.name, CFG["image"]["allowed_extensions"])
            image = load_image_rgb(uploaded)
            with st.spinner(f"Running {model}..."):
                manager.get_detector(model, resolved)
                result, performance = run_with_performance(
                    lambda: run_image_detection(manager, model, image, img_prompts, device=resolved, conf=conf,
                                                text_threshold=text_thr, max_side=max_side, imgsz=imgsz),
                    resolved,
                )
            annotated = draw_detections(image, result)
            saved = save_annotated_image(annotated, stem=Path(uploaded.name).stem)
            st.session_state["img_out"] = {"name": uploaded.name, "original": image, "annotated": annotated,
                                           "result": result, "performance": performance, "saved": str(saved),
                                           "settings": {"prompts": img_prompts, "confidence_threshold": conf,
                                                        "text_threshold": text_thr, "max_side": max_side,
                                                        "imgsz": imgsz}}
            record_run("Image", uploaded.name, result)
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

    out = st.session_state.get("img_out")
    if out:
        res = out["result"]
        st.markdown(f"**Result for `{out['name']}`** - {res['model_name']}")
        m = st.columns(5)
        m[0].metric("Detections", res["num_detections"])
        m[1].metric("Inference time", f"{res['inference_time'] * 1000:.1f} ms")
        m[2].metric("Avg confidence", f"{np.mean(res['scores']) * 100:.1f}%" if res["scores"] else "-")
        m[3].metric("Image size", f"{res['image_width']}x{res['image_height']}")
        m[4].metric("Device", str(res.get("device", "-")).upper())
        if "performance" in out:
            show_runtime_benchmark(out["performance"])
        c1, c2 = st.columns(2)
        c1.image(out["original"], caption="Original", use_container_width=True)
        c2.image(out["annotated"], caption="Annotated", use_container_width=True)
        if res["num_detections"] == 0:
            st.info("No objects found. Try lowering the threshold or rephrasing the categories.")
        else:
            df = result_to_dataframe(res)
            min_display_conf = st.slider("Minimum confidence shown in insights", 0, 100, 0, 1,
                                         format="%d%%", key="img_min_display_conf")
            visible_df = df[df["confidence_%"] >= min_display_conf]
            st.caption(f"Showing {len(visible_df)} of {len(df)} detections in the insights below. "
                       "The annotated image includes all detections from inference.")
            if visible_df.empty:
                st.info("No detections meet this display filter. Lower the slider to see more results.")
            else:
                insights_left, insights_right = st.columns(2)
                with insights_left:
                    st.markdown("**Detections by category**")
                    st.bar_chart(visible_df["label"].value_counts().rename("Detections"))
                with insights_right:
                    st.markdown("**Mean confidence by category**")
                    confidence_by_label = visible_df.groupby("label")["confidence_%"].mean().sort_values(
                        ascending=False).rename("Mean confidence (%)")
                    st.bar_chart(confidence_by_label)
                st.dataframe(visible_df, use_container_width=True, hide_index=True)
                st.download_button("Download detection data (CSV)", visible_df.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{Path(out['name']).stem}_detections.csv", mime="text/csv")
        st.download_button("Download annotated image (PNG)", png_bytes(out["annotated"]),
                           file_name=Path(out["saved"]).name, mime="image/png")
        report_details = {
            "Image": out["name"],
            "Model": res["model_name"],
            "Device": res.get("device", "-"),
            "Image size": f"{res['image_width']}x{res['image_height']}",
            "Detections": res["num_detections"],
            "Inference time (ms)": round(res["inference_time"] * 1000, 2),
            "Average confidence": round(float(np.mean(res["scores"])), 4) if res["scores"] else "N/A",
            **out.get("settings", {}),
        }
        for key, value in out.get("performance", {}).items():
            report_details[f"Runtime {key}"] = value
        image_report_detections = result_to_dataframe(res)
        display_confidence = st.session_state.get("img_min_display_conf", 0)
        visible_report_detections = image_report_detections[
            image_report_detections["confidence_%"] >= display_confidence
        ]
        image_charts = {}
        if not visible_report_detections.empty:
            image_charts = {
                "Detections by category": (
                    "bar",
                    visible_report_detections["label"].value_counts().rename("Detections").to_frame(),
                    "Detections",
                ),
                "Mean confidence by category": (
                    "bar",
                    visible_report_detections.groupby("label")["confidence_%"].mean().sort_values(
                        ascending=False
                    ).rename("Mean confidence (%)").to_frame(),
                    "Mean confidence (%)",
                ),
            }
        image_report = out.get("pdf_report")
        if image_report is None or out.get("pdf_report_filter") != display_confidence:
            image_report = create_pdf_report(
                f"Image detection - {out['name']}",
                report_details,
                {"Detections": image_report_detections},
                {"Annotated image": out["annotated"]},
                image_charts,
            )
            out["pdf_report"] = image_report
            out["pdf_report_filter"] = display_confidence
        st.download_button(
            "Download image report (PDF)",
            image_report,
            file_name=f"{Path(out['name']).stem}_detection_report.pdf",
            mime="application/pdf",
        )
        st.caption(f"Saved to {out['saved']}. Time covers model inference + post-processing; "
                   "image decoding, resizing and drawing are excluded.")


# ------------------------------------------------------------------ video tab
def _save_upload(upload) -> str:
    """Write the upload to a temp file once per file; remove the previous temp file."""
    key = f"{upload.name}-{upload.size}"
    if st.session_state.get("vid_src_key") != key:
        old = st.session_state.get("vid_src_path")
        if old:
            Path(old).unlink(missing_ok=True)
        suffix = Path(upload.name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
            fh.write(upload.getvalue())
        st.session_state["vid_src_key"] = key
        st.session_state["vid_src_path"] = fh.name
    return st.session_state["vid_src_path"]


with tab_vid:
    left, right = st.columns(2)
    with left:
        vid = st.file_uploader("Upload a video", type=[e.strip(".") for e in CFG["video"]["allowed_extensions"]],
                               key="vid_upload")
        if vid is not None:
            try:
                src_path = _save_upload(vid)
                vinfo = get_video_info(src_path)
                st.video(vid.getvalue())
                a = st.columns(4)
                a[0].metric("Resolution", f"{vinfo['width']}x{vinfo['height']}")
                a[1].metric("Input FPS", f"{vinfo['fps']:.2f}")
                a[2].metric("Total frames", vinfo["total_frames"])
                a[3].metric("Duration", f"{vinfo['duration_s']:.1f} s")
            except Exception as exc:  # noqa: BLE001
                show_error(exc)
                vid = None
    with right:
        vid_prompts = prompt_input("vid")
    if frame_skip > 1:
        st.warning(f"Frame skipping is ON: only every {frame_skip}th frame is processed and written.")

    if st.button("Process Video", type="primary", disabled=vid is None or not vid_prompts, key="vid_run"):
        progress = st.progress(0.0, text="Starting...")
        preview = st.empty()
        try:
            validate_extension(vid.name, CFG["video"]["allowed_extensions"])
            out_path = resolve_path(CFG["paths"]["videos_out"]) / f"{Path(vid.name).stem}_{int(time.time())}_annotated.mp4"
            manager.get_detector(model, resolved)
            monitor = PerformanceMonitor(resolved, interval=float(CFG["benchmark"]["monitor_interval_s"]))
            monitor.start()

            def on_progress(frame_no, total, processed, frame):
                if total:
                    progress.progress(min(frame_no / total, 1.0), text=f"Frame {frame_no}/{total}")
                else:
                    progress.progress(0.0, text=f"Frame {frame_no}")
                if processed % 3 == 0:
                    preview.image(frame, caption="Live preview", use_container_width=True)

            try:
                stats = process_video(manager, model, st.session_state["vid_src_path"], out_path, vid_prompts,
                                      device=resolved, conf=conf, text_threshold=text_thr,
                                      frame_skip=int(frame_skip), resize_side=video_side, imgsz=imgsz,
                                      max_frames=int(max_frames), progress_callback=on_progress)
                e2e_per_frame = stats["total_processing_time_s"] / stats["processed_frames"]
                for frame in stats["per_frame"]:
                    monitor.record_inference(frame["inference_ms"] / 1000)
                    monitor.record_item(e2e_per_frame)
            finally:
                monitor.stop()
            stats["performance"] = monitor.summary()
            stats.update({"input_name": vid.name, "prompts": list(vid_prompts), "device": resolved,
                          "confidence_threshold": conf, "text_threshold": text_thr})
            progress.progress(1.0, text="Done")
            st.session_state["vid_out"] = stats
            record_run("Video", vid.name, {
                "model_name": model,
                "num_detections": stats["total_detections"],
                "inference_time": stats["avg_inference_ms"] / 1000,
                "device": resolved,
            })
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
        finally:
            preview.empty()

    stats = st.session_state.get("vid_out")
    if stats:
        st.subheader("Results")
        r1 = st.columns(4)
        r1[0].metric("Processed frames", stats["processed_frames"])
        r1[1].metric("Avg inference time", f"{stats['avg_inference_ms']:.1f} ms")
        r1[2].metric("Inference FPS (model only)", f"{stats['inference_fps']:.2f}")
        r1[3].metric("End-to-end FPS", f"{stats['end_to_end_fps']:.2f}")
        r2 = st.columns(4)
        r2[0].metric("Total processing time", f"{stats['total_processing_time_s']:.1f} s")
        r2[1].metric("Input FPS (not a model FPS)", f"{stats['input_fps']:.2f}")
        r2[2].metric("Input frames", stats["input_total_frames"])
        r2[3].metric("Total detections", stats["total_detections"])
        st.caption(stats["note"])
        if "performance" in stats:
            show_runtime_benchmark(stats["performance"], "Video processing time")
        out_file = Path(stats["output_path"])
        if out_file.exists():
            data = out_file.read_bytes()
            if stats["browser_playable"]:
                st.video(data)
            else:
                st.info("The output uses the OpenCV 'mp4v' codec, which browsers often cannot play. "
                        "Download it, or install ffmpeg so it is converted to H.264 automatically.")
            st.download_button("Download annotated video", data, file_name=out_file.name, mime="video/mp4")
            st.caption(f"Saved to {out_file}")
        per_frame = pd.DataFrame(stats["per_frame"])
        if not per_frame.empty:
            st.line_chart(per_frame.set_index("frame")[["inference_ms"]])
        report_details = {
            "Video": stats.get("input_name", Path(stats["output_path"]).name),
            "Model": stats["model_name"],
            "Device": stats.get("device", "-"),
            "Input resolution": stats["input_resolution"],
            "Input duration (seconds)": round(stats["input_duration_s"], 2),
            "Input frames": stats["input_total_frames"],
            "Processed frames": stats["processed_frames"],
            "Frame skip": stats["frame_skip"],
            "Total detections": stats["total_detections"],
            "Average inference time (ms)": round(stats["avg_inference_ms"], 2),
            "Inference FPS": round(stats["inference_fps"], 2),
            "End-to-end FPS": round(stats["end_to_end_fps"], 2),
            "Total processing time (seconds)": round(stats["total_processing_time_s"], 2),
            "Prompts": stats.get("prompts", []),
            "Confidence threshold": stats.get("confidence_threshold"),
            "Text threshold": stats.get("text_threshold"),
            "Annotated video": stats["output_path"],
        }
        for key, value in stats.get("performance", {}).items():
            report_details[f"Runtime {key}"] = value
        video_report = stats.get("pdf_report")
        if video_report is None:
            video_report = create_pdf_report(
                f"Video detection - {report_details['Video']}",
                report_details,
                {"Per-frame results": per_frame},
                charts={
                    "Inference time by frame": (
                        "line",
                        per_frame.set_index("frame")[["inference_ms"]],
                        "Inference time (ms)",
                    ),
                } if not per_frame.empty else None,
            )
            stats["pdf_report"] = video_report
        st.download_button(
            "Download video report (PDF)",
            video_report,
            file_name=f"{Path(report_details['Video']).stem}_video_report.pdf",
            mime="application/pdf",
        )


# ------------------------------------------------------------------ comparison tab
def _agreement(a: dict, b: dict, thr: float = 0.5) -> pd.DataFrame:
    rows = []
    for label in sorted(set(a["labels"]) | set(b["labels"])):
        ba = np.array([x for x, l in zip(a["boxes"], a["labels"]) if l == label]).reshape(-1, 4)
        bb = np.array([x for x, l in zip(b["boxes"], b["labels"]) if l == label]).reshape(-1, 4)
        ious = pair_boxes(ba, bb, thr)
        rows.append({"label": label, "Grounding DINO": len(ba), "YOLO-World": len(bb),
                     f"matched (IoU>={thr})": len(ious),
                     "mean IoU of matches": round(float(np.mean(ious)), 3) if ious else None})
    return pd.DataFrame(rows)


with tab_cmp:
    st.markdown("Run the **same image and prompts** through both models.")
    left, right = st.columns(2)
    with left:
        cmp_up = st.file_uploader("Test image", type=[e.strip(".") for e in CFG["image"]["allowed_extensions"]],
                                  key="cmp_upload")
        t1, t2 = st.columns(2)
        dino_conf = t1.slider("Grounding DINO box threshold", 0.01, 0.95, float(CFG["detection"]["dino_box_threshold"]), 0.01)
        yolo_conf = t2.slider("YOLO-World confidence", 0.01, 0.95, float(CFG["detection"]["yolo_conf"]), 0.01)
    with right:
        cmp_prompts = prompt_input("cmp")

    if st.button("Run both models", type="primary", disabled=cmp_up is None or not cmp_prompts, key="cmp_run"):
        try:
            image = load_image_rgb(cmp_up)
            results = {}
            performances = {}
            for name, c in (("Grounding DINO", dino_conf), ("YOLO-World", yolo_conf)):
                with st.spinner(f"Running {name}..."):
                    manager.get_detector(name, resolved)
                    results[name], performances[name] = run_with_performance(
                        lambda: run_image_detection(manager, name, image, cmp_prompts, device=resolved,
                                                    conf=c, text_threshold=text_thr, max_side=max_side,
                                                    imgsz=imgsz),
                        resolved,
                    )
            st.session_state["cmp_out"] = {"name": cmp_up.name, "image": image, "results": results,
                                           "performances": performances,
                                           "annotated": {n: draw_detections(image, r) for n, r in results.items()},
                                           "prompts": list(cmp_prompts),
                                           "settings": {"dino_conf": dino_conf, "yolo_conf": yolo_conf,
                                                        "text_threshold": text_thr, "max_side": max_side,
                                                        "imgsz": imgsz}}
            for name, result in results.items():
                record_run("Comparison", cmp_up.name, result)
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

    cmp_out = st.session_state.get("cmp_out")
    if cmp_out:
        cols = st.columns(2)
        rows = []
        for col, name in zip(cols, MODEL_NAMES):
            r = cmp_out["results"][name]
            performance = cmp_out.get("performances", {}).get(name, {})
            col.image(cmp_out["annotated"][name], caption=name, use_container_width=True)
            end_to_end_s = _performance_number(performance, "e2e_total_s")
            rows.append({"Model": name, "Detections": r["num_detections"],
                         "Avg confidence %": round(float(np.mean(r["scores"])) * 100, 1) if r["scores"] else None,
                         "Max confidence %": round(max(r["scores"]) * 100, 1) if r["scores"] else None,
                         "Inference ms": round(r["inference_time"] * 1000, 1),
                         "Detection pipeline ms": round(end_to_end_s * 1000, 1) if end_to_end_s is not None else None,
                         "Peak GPU MB": _performance_number(performance, "gpu_peak_allocated_mb"),
                         "Peak GPU util %": _performance_number(performance, "gpu_util_max_pct"),
                         "Peak RAM MB": _performance_number(performance, "ram_peak_mb")})
        comparison_summary = pd.DataFrame(rows)
        st.dataframe(comparison_summary, hide_index=True, use_container_width=True)
        st.bar_chart(comparison_summary.set_index("Model")[["Inference ms"]])
        st.download_button("Download model comparison (CSV)", comparison_summary.to_csv(index=False).encode("utf-8"),
                           file_name=f"{Path(cmp_out['name']).stem}_model_comparison.csv",
                           mime="text/csv")
        st.markdown("**Box agreement between the two models** (greedy one-to-one matching per category)")
        agreement_df = _agreement(*(cmp_out["results"][n] for n in MODEL_NAMES))
        st.dataframe(agreement_df, hide_index=True, use_container_width=True)
        st.download_button("Download box agreement (CSV)", agreement_df.to_csv(index=False).encode("utf-8"),
                           file_name="model_box_agreement.csv", mime="text/csv")
        comparison_detections = []
        for model_name, result in cmp_out["results"].items():
            detections = result_to_dataframe(result)
            detections.insert(0, "model", model_name)
            comparison_detections.append(detections)
        comparison_detection_df = pd.concat(comparison_detections, ignore_index=True)
        report_details = {
            "Image": cmp_out["name"],
            "Models": ", ".join(MODEL_NAMES),
            "Prompts": cmp_out.get("prompts", []),
            "Grounding DINO threshold": cmp_out.get("settings", {}).get("dino_conf", "-"),
            "YOLO-World threshold": cmp_out.get("settings", {}).get("yolo_conf", "-"),
        }
        for model_name, performance in cmp_out.get("performances", {}).items():
            for key, value in performance.items():
                report_details[f"{model_name} runtime {key}"] = value
        comparison_report = cmp_out.get("pdf_report")
        if comparison_report is None:
            comparison_report = create_pdf_report(
                f"Model comparison - {cmp_out['name']}",
                report_details,
                {"Model summary": comparison_summary, "Box agreement": agreement_df,
                 "Detections": comparison_detection_df},
                {name: cmp_out["annotated"][name] for name in MODEL_NAMES},
                {"Inference time by model": (
                    "bar",
                    comparison_summary.set_index("Model")[["Inference ms"]],
                    "Inference time (ms)",
                )},
            )
            cmp_out["pdf_report"] = comparison_report
        st.download_button(
            "Download model comparison report (PDF)",
            comparison_report,
            file_name=f"{Path(cmp_out['name']).stem}_comparison_report.pdf",
            mime="application/pdf",
        )
        st.caption("Agreement is not accuracy - without ground truth neither model is the reference. "
                   "Thresholds differ in meaning between the two models.")


# ------------------------------------------------------------------ benchmark tab
with tab_bench:
    success_message = st.session_state.pop("benchmark_success_message", None)
    if success_message:
        st.success(success_message)
    benchmark_pdf = st.session_state.get("benchmark_pdf")
    if benchmark_pdf:
        st.download_button(
            "Download dataset benchmark report (PDF)",
            benchmark_pdf["data"],
            file_name=benchmark_pdf["filename"],
            mime="application/pdf",
        )
    default_annotations = resolve_path(CFG["benchmark"]["annotations"])
    st.caption("Runtime timing and resource benchmarks appear with image, video, and comparison results. "
               "Precision, recall, and mAP require a COCO-annotated image dataset.")
    if not default_annotations.is_file():
        st.warning(
            f"Dataset accuracy benchmarking needs a COCO annotation file. The configured file is missing: "
            f"`{default_annotations}`. `annotation_format_example.json` is only a template, not ground truth."
        )

    summaries = load_summaries()
    rdir = reports_dir()
    table_path = rdir / "comparison_table.csv"
    if not summaries:
        st.warning("Evaluation has not been completed yet - no benchmark results exist. "
                   "Add images and COCO-style annotations, then run the benchmark below or from the command line.")
    else:
        for name, s in summaries.items():
            st.caption(f"{name}: {s['num_images_evaluated']} images on {s['device']} - {s['timestamp']}")
        if table_path.exists():
            st.subheader("Model comparison")
            st.dataframe(pd.read_csv(table_path), hide_index=True, use_container_width=True)
        acc = pd.DataFrame({m: {k: s["overall"].get(k) for k in ("precision", "recall", "f1", "map50", "map50_95")}
                            for m, s in summaries.items()})
        perf = pd.DataFrame({m: {"Avg inference (ms)": s["performance"].get("avg_inference_ms"),
                                 "Inference FPS": s["performance"].get("inference_fps"),
                                 "End-to-end FPS": s["performance"].get("end_to_end_fps"),
                                 "Peak GPU MB": s["performance"].get("gpu_peak_allocated_mb"),
                                 "Peak RAM MB": s["performance"].get("ram_peak_mb")} for m, s in summaries.items()})
        c1, c2 = st.columns(2)
        c1.markdown("**Accuracy (IoU 0.50)**")
        c1.bar_chart(acc.fillna(0.0))
        c2.markdown("**Speed and memory**")
        c2.dataframe(perf, use_container_width=True)
        for fname in ("inference_time.png", "fps.png", "gpu_memory.png", "ram.png", "per_category_precision.png",
                      "per_category_recall.png", "uncommon_categories.png"):
            if (rdir / fname).exists():
                st.image(str(rdir / fname), use_container_width=True)
        per_cat = rdir / "per_category_comparison.csv"
        if per_cat.exists():
            st.subheader("Per-category results")
            st.dataframe(pd.read_csv(per_cat), hide_index=True, use_container_width=True)

    with st.expander("Run a dataset benchmark"):
        benchmark_target = st.selectbox(
            "Models to evaluate",
            ["Both models", *MODEL_NAMES],
            help="A single-model run is useful for checking one detector; both models builds the comparison report.",
        )
        b1, b2 = st.columns(2)
        images_dir = b1.text_input("Images folder", CFG["benchmark"]["images_dir"])
        ann_path = b2.text_input("COCO annotation JSON", CFG["benchmark"]["annotations"])
        b3, b4, b5 = st.columns(3)
        b_dino = b3.number_input("Grounding DINO box threshold", 0.01, 0.95, float(CFG["detection"]["dino_box_threshold"]), 0.01)
        b_yolo = b4.number_input("YOLO-World confidence", 0.01, 0.95, float(CFG["detection"]["yolo_conf"]), 0.01)
        b_limit = b5.number_input("Limit images (0 = all)", 0, 100000, 0)
        if st.button("Run benchmark", key="bench_run"):
            image_path = resolve_path(images_dir)
            annotation_path = resolve_path(ann_path)
            if not image_path.is_dir():
                st.error(f"Image folder does not exist: {image_path}")
            elif not any(path.is_file() for path in image_path.iterdir()):
                st.error(f"No image files found in: {image_path}")
            elif not annotation_path.is_file():
                st.error(
                    f"COCO annotation file not found: {annotation_path}. "
                    "Create real ground-truth annotations; the example JSON is only a format template."
                )
            else:
                if benchmark_target == "Both models":
                    cmd = [sys.executable, "-m", "benchmarking.compare_models", "--images-dir", str(image_path),
                           "--annotations", str(annotation_path), "--device", resolved, "--dino-conf", str(b_dino),
                           "--yolo-conf", str(b_yolo), "--text-threshold", str(text_thr)]
                else:
                    selected_conf = b_dino if benchmark_target == "Grounding DINO" else b_yolo
                    cmd = [sys.executable, "-m", "benchmarking.evaluate", "--model", benchmark_target,
                           "--images-dir", str(image_path), "--annotations", str(annotation_path),
                           "--device", resolved, "--conf", str(selected_conf), "--text-threshold", str(text_thr),
                           "--max-side", str(max_side), "--imgsz", str(imgsz)]
                if b_limit:
                    cmd += ["--limit", str(int(b_limit))]
                with st.spinner("Benchmarking - this can take several minutes..."):
                    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
                if proc.returncode == 0:
                    if benchmark_target != "Both models":
                        generate_report()
                    report_summaries = load_summaries()
                    if not report_summaries:
                        st.error("Benchmark completed, but no evaluation summary was found to build the PDF report.")
                    else:
                        per_category = pd.DataFrame([
                            {"model": name, **row}
                            for name, summary in report_summaries.items()
                            for row in summary.get("per_class", [])
                        ])
                        category_tables = build_category_metric_tables(
                            per_category,
                            load_test_prompts().get("uncommon", []),
                        )
                        accuracy_chart = pd.DataFrame({
                            name: {key: summary["overall"].get(key)
                                   for key in ("precision", "recall", "f1", "map50", "map50_95")}
                            for name, summary in report_summaries.items()
                        }).fillna(0.0)
                        benchmark_chart_files = {
                            "Inference time": "inference_time.png",
                            "Throughput": "fps.png",
                            "GPU memory": "gpu_memory.png",
                            "Process RAM": "ram.png",
                            "Per-category precision": "per_category_precision.png",
                            "Per-category recall": "per_category_recall.png",
                            "Uncommon category F1": "uncommon_categories.png",
                        }
                        benchmark_images = {
                            title: rdir / filename
                            for title, filename in benchmark_chart_files.items()
                            if (rdir / filename).is_file()
                        }
                        report_details = {
                            "Models shown": ", ".join(report_summaries),
                            "Run target": benchmark_target,
                            "Images folder": str(image_path),
                            "Annotations": str(annotation_path),
                            "Image limit": int(b_limit) or "All",
                        }
                        for name, summary in report_summaries.items():
                            report_details[f"{name} evaluated images"] = summary["num_images_evaluated"]
                            report_details[f"{name} device"] = summary["device"]
                            report_details[f"{name} timestamp"] = summary["timestamp"]
                        st.session_state["benchmark_pdf"] = {
                            "data": create_pdf_report(
                                "Dataset benchmark report",
                                report_details,
                                {
                                    "Overall metrics": pd.DataFrame([
                                        {"Model": name, "Metric": metric, "Value": value}
                                        for name, summary in report_summaries.items()
                                        for section in ("overall", "performance")
                                        for metric, value in summary.get(section, {}).items()
                                    ]),
                                    **category_tables,
                                },
                                images=benchmark_images,
                                charts={
                                    "Accuracy (IoU 0.50)": ("bar", accuracy_chart, "Score"),
                                },
                            ),
                            "filename": f"dataset_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                        }
                    st.session_state["benchmark_success_message"] = (
                        f"Benchmark complete for {benchmark_target}. Updated results are shown below."
                    )
                    st.rerun()
                else:
                    st.error("Benchmark failed. See the command output below.")
                    st.code((proc.stdout + "\n" + proc.stderr)[-4000:])


# ------------------------------------------------------------------ info tab
with tab_info:
    st.markdown(
        """
### Objective
Evaluate **open-vocabulary object detection** for industrial computer-vision use: detect objects named
by free-text prompts with pretrained models only - **no training and no fine-tuning**.

### Grounding DINO (`IDEA-Research/grounding-dino-tiny`)
A DINO transformer detector fused with a text encoder. Image and text features are combined inside the network
(feature enhancer, language-guided query selection, cross-modality decoder), so the prompt directly shapes every
detection. Prompts are one string with categories separated by periods. Strong with phrases and attributes, but
heavier and slower.

### YOLO-World (`yolov8s-worldv2.pt`)
A YOLOv8-based detector with a CLIP text encoder. The vocabulary is encoded once with `set_classes()` and reused,
so inference runs like a normal YOLO model - much faster and well suited to video. Changing categories means
calling `set_classes()` again.

### Technologies
Python, PyTorch, Hugging Face Transformers, Ultralytics, OpenCV, Streamlit, NumPy, Pillow, Pandas, Matplotlib,
Psutil, NVIDIA CUDA / NVML.

### Limitations
- Confidence scores are not accuracy; precision/recall/mAP need ground-truth annotations.
- Thresholds mean different things in the two models, so equal sliders are not equal operating points.
- Rare or ambiguous prompts can be missed or confused; wording matters.
- Grounding DINO may return merged labels for adjacent prompts; labels are mapped back to the closest prompt.
- Peak GPU memory shown is PyTorch-allocated memory, not the total reserved by the driver.
- Uncommon categories here means rare in typical datasets - not a claim they were unseen during pretraining.
        """
    )
