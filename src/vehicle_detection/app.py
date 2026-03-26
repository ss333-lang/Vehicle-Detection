"""Vehicle Detection & Tracking System — Streamlit UI.

Provides two processing modes:

- **Batch**: Upload a video, process all frames, save an annotated MP4,
  and display a summary with download link.
- **Real-time**: Upload a video and display each frame immediately as
  detection, tracking, and colour recognition run on it.
"""

import json
import os
import pathlib
import subprocess
import tempfile

import streamlit as st

# set_page_config must be the first Streamlit call in the module.
st.set_page_config(
    page_title="Vehicle Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

from vehicle_detection.utils.video_processor import (  # noqa: E402
    CLASS_COLORS,
    TRACKER_MAP,
    VEHICLE_CLASSES,
    VehicleDetector,
)

# ── Module-level constants ─────────────────────────────────────────────────────

# Resolve project root regardless of where the process is launched from.
_PROJECT_ROOT: pathlib.Path = (
    pathlib.Path(__file__).resolve().parent.parent.parent
)
TEMP_DIR: str = str(_PROJECT_ROOT / "temp")

# Update the batch-mode preview image every N processed frames.
_PREVIEW_INTERVAL: int = 15

# ffmpeg H.264 encoding parameters for browser-compatible playback.
_FFMPEG_TIMEOUT: int = 600
_H264_CRF: int = 23

os.makedirs(TEMP_DIR, exist_ok=True)


# ── Session-state initialisation ──────────────────────────────────────────────

for _key, _default in {"detector": None, "model_loaded": False}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ── Shared UI helpers ─────────────────────────────────────────────────────────

def render_stats(
    stats: dict,
    unique_tracks: set | None = None,
    fps: float | None = None,
) -> None:
    """Render detection and tracking statistics as Streamlit metric cards.

    Args:
        stats (dict): Detection stats with keys ``total``, ``classes``,
            and ``colors``.
        unique_tracks (set | None): Set of unique track IDs seen so far.
            Displayed as a metric when provided.
        fps (float | None): Average processing FPS. Displayed when
            provided.
    """
    total = stats.get("total", 0)
    classes = stats.get("classes", {})
    colors = stats.get("colors", {})

    # Build summary row with a variable number of columns.
    n_cols = (
        2
        + (1 if unique_tracks is not None else 0)
        + (1 if fps is not None else 0)
    )
    cols = st.columns(n_cols)
    cols[0].metric("Detections", total)
    idx = 1
    if unique_tracks is not None:
        cols[idx].metric("Unique Tracks", len(unique_tracks))
        idx += 1
    if fps is not None:
        cols[idx].metric("Avg FPS", f"{fps:.1f}")

    if classes:
        st.caption("Vehicle Types")
        cls_items = sorted(classes.items(), key=lambda x: -x[1])
        ccols = st.columns(max(len(cls_items), 1))
        for i, (cls, cnt) in enumerate(cls_items):
            ccols[i].metric(cls.capitalize(), cnt)

    if colors:
        st.caption("Colors Detected")
        top = sorted(colors.items(), key=lambda x: -x[1])[:6]
        vcols = st.columns(max(len(top), 1))
        for i, (color, cnt) in enumerate(top):
            vcols[i].metric(color, cnt)


def save_upload(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to the temporary directory.

    Args:
        uploaded_file: Streamlit UploadedFile from ``st.file_uploader``.

    Returns:
        str: Absolute path to the saved temporary file.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=TEMP_DIR
    )
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def try_reencode_h264(src: str) -> str:
    """Re-encode *src* with H.264 via ffmpeg for browser compatibility.

    Falls back to returning *src* unchanged when ffmpeg is not installed
    or the re-encode fails, so the download button still works.

    Args:
        src (str): Path to the source video file (mp4v codec).

    Returns:
        str: Path to the H.264-encoded file, or *src* on failure.
    """
    dst = src.replace(".mp4", "_h264.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", src,
                "-c:v", "libx264", "-crf", str(_H264_CRF),
                "-preset", "fast", "-an", dst,
            ],
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT,
        )
        if result.returncode == 0 and os.path.exists(dst):
            return dst
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return src


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚗 Vehicle Detection")
    st.divider()

    st.markdown("### Model")
    if not st.session_state.model_loaded:
        with st.spinner("Loading model …"):
            try:
                st.session_state.detector = VehicleDetector("models/best.pt")
                st.session_state.model_loaded = True
            except OSError as exc:
                st.error(f"Model load failed: {exc}")
    if st.session_state.model_loaded:
        st.success("Model ready")
    st.divider()

    st.markdown("### Detection")
    conf = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.15, 0.05, key="conf"
    )
    iou = st.slider(
        "IoU threshold", 0.10, 0.95, 0.60, 0.05, key="iou"
    )
    st.divider()

    st.markdown("### Tracker")
    tracker_name = st.radio("Algorithm", list(TRACKER_MAP.keys()), index=0)
    tracker_yaml = TRACKER_MAP[tracker_name]
    st.divider()

    st.markdown("### Options")
    detect_colors = st.toggle("Color Detection", value=True)
    show_labels = st.toggle("Show Labels", value=True)
    augment = st.toggle("Augment (better distant detection)", value=False)
    show_trails = st.toggle("Show Track Trails", value=True)
    frame_skip = st.slider(
        "Frame Skip (real-time speed)", 1, 10, 1,
        help="Run detection every N frames. Use 1 for smooth tracking. Increase on slow hardware.",
    )
    st.divider()

    st.markdown("### Speed Estimation")
    speed_enabled = st.toggle("Estimate Speed (KPH)", value=False)
    meters_per_pixel = st.slider(
        "Scale Factor (m/px)", 0.01, 0.20, 0.05, 0.005,
        help="Metres per pixel. Requires camera calibration for accuracy. "
             "Default 0.05 is a rough estimate for a typical traffic camera.",
        disabled=not speed_enabled,
    )
    if not speed_enabled:
        meters_per_pixel = 0.0
    st.divider()

    st.markdown("### Counting Line")
    line_enabled = st.toggle("Enable Counting Line", value=False)
    line_position = st.slider(
        "Line Position (% from top)", 10, 90, 50,
        help="Horizontal line position. Vehicles crossing this line are counted. Only vehicles below this line are detected.",
        disabled=not line_enabled,
    )
    line_y = line_position / 100.0 if line_enabled else None
    count_direction = st.radio(
        "Count Direction",
        ["down", "up", "both"],
        index=0,
        help="down: vehicles moving top→bottom  |  up: bottom→top  |  both: either",
        disabled=not line_enabled,
        horizontal=True,
    )

    # When the counting line is active, use its position as the ROI top
    # so only vehicles below (in front of) the line are detected.
    if line_enabled:
        roi_top = line_position / 100.0
    else:
        roi_top_pct = st.slider(
            "Mask top % (ignore sky/bridge)", 0, 60, 0,
            help="Black out the top N% of the frame before detection.",
        )
        roi_top = roi_top_pct / 100.0
    st.divider()

    st.markdown("### Vehicle Types")
    class_id_map = {name: cid for cid, name in VEHICLE_CLASSES.items()}
    vehicle_checks = {
        name: st.checkbox(name.capitalize(), value=True)
        for name in VEHICLE_CLASSES.values()
    }
    selected_ids = [
        class_id_map[n]
        for n, checked in vehicle_checks.items()
        if checked
    ]
    if not selected_ids:
        st.warning("Select at least one class.")
        selected_ids = list(VEHICLE_CLASSES.keys())

    st.divider()
    st.markdown("### Box Colors")
    for cls, bgr in CLASS_COLORS.items():
        if cls == "unknown":
            continue
        r, g, b = bgr[2], bgr[1], bgr[0]
        st.markdown(
            f'<span style="background-color:rgb({r},{g},{b});'
            f'padding:2px 8px;border-radius:4px;'
            f'color:white;font-size:0.8em">'
            f"{cls.capitalize()}</span>",
            unsafe_allow_html=True,
        )


# ── Main content ──────────────────────────────────────────────────────────────

st.title("Vehicle Detection & Tracking System")
st.caption(
    f"**Model:** Indian Vehicle YOLO11m  |  **Tracker:** {tracker_name}  |  "
    f"**Conf:** {conf}  |  **IoU:** {iou}  |  "
    f"**Color Detection:** {'On' if detect_colors else 'Off'}"
)

tab_batch, tab_rt = st.tabs(
    ["📁  Batch Processing", "▶️  Real-time Stream"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Batch Processing
# ═══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.markdown(
        "Upload a video file. Every frame is processed and written to an "
        "annotated output video, which you can preview and download."
    )

    uploaded_batch = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="batch_file",
    )

    if uploaded_batch:
        input_path = save_upload(uploaded_batch)
        output_path = os.path.join(TEMP_DIR, "output_batch.mp4")

        if st.button("⚙️  Process Video", type="primary"):
            if not st.session_state.model_loaded:
                st.error("Model is not loaded yet — please wait.")
            else:
                progress_bar = st.progress(0, text="Initialising …")
                prev_col, stats_col = st.columns([3, 2])

                with prev_col:
                    preview_ph = st.empty()
                with stats_col:
                    live_stats_ph = st.empty()

                cum_stats: dict = {}
                last_fps = 0.0

                processor = st.session_state.detector.process_video(
                    input_path,
                    output_path,
                    tracker_yaml=tracker_yaml,
                    conf=conf,
                    iou=iou,
                    classes=selected_ids,
                    detect_colors=detect_colors,
                    show_labels=show_labels,
                    meters_per_pixel=meters_per_pixel,
                    show_trails=show_trails,
                )

                for rgb, cum, fn, total_f, fps in processor:
                    cum_stats = cum
                    last_fps = fps
                    pct = fn / max(total_f, 1)
                    progress_bar.progress(
                        pct,
                        text=f"Frame {fn} / {total_f}  |  {fps:.1f} FPS",
                    )

                    if fn % _PREVIEW_INTERVAL == 0 or fn == 1:
                        preview_ph.image(
                            rgb,
                            caption=f"Preview — frame {fn}",
                            width="stretch",
                        )

                    with live_stats_ph.container():
                        st.markdown("**Live Stats**")
                        render_stats(
                            cum_stats,
                            unique_tracks=cum_stats.get("unique_tracks"),
                            fps=fps,
                        )

                progress_bar.progress(1.0, text="Processing complete!")
                preview_ph.empty()

                # ── JSON detection export ───────────────────────────────────
                all_detections = cum_stats.get("all_detections", [])
                if all_detections:
                    json_payload = {
                        "total_frames_processed": fn,
                        "unique_vehicles_tracked": len(
                            cum_stats.get("unique_tracks", set())
                        ),
                        "detections": all_detections,
                    }
                    st.download_button(
                        label="⬇️  Download Detection JSON",
                        data=json.dumps(json_payload, indent=2),
                        file_name="vehicle_detections.json",
                        mime="application/json",
                    )

                st.subheader("Summary")
                render_stats(
                    cum_stats,
                    unique_tracks=cum_stats.get("unique_tracks"),
                    fps=last_fps,
                )

                st.divider()
                st.subheader("Output Video")
                playback_path = try_reencode_h264(output_path)
                st.video(playback_path)

                with open(output_path, "rb") as fh:
                    st.download_button(
                        label="⬇️  Download Processed Video",
                        data=fh.read(),
                        file_name="vehicle_detection_output.mp4",
                        mime="video/mp4",
                    )

                try:
                    os.unlink(input_path)
                except OSError:
                    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Real-time Stream
# ═══════════════════════════════════════════════════════════════════════════════

with tab_rt:
    st.markdown(
        "Upload a video file. Detection, tracking, and colour recognition "
        "run on every frame and are displayed immediately as it plays."
    )

    uploaded_rt = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        key="rt_file",
    )

    if uploaded_rt:
        rt_path = save_upload(uploaded_rt)

        if st.button("▶️  Start Real-time Detection", type="primary"):
            if not st.session_state.model_loaded:
                st.error("Model is not loaded yet — please wait.")
            else:
                col_video, col_stats = st.columns([3, 1])

                with col_video:
                    frame_ph = st.empty()
                    caption_ph = st.empty()

                with col_stats:
                    st.markdown("**Live Stats**")
                    stats_ph = st.empty()

                rt_progress = st.progress(0, text="Starting stream …")

                rt_cum: dict = {
                    "total": 0,
                    "classes": {},
                    "colors": {},
                    "unique_tracks": set(),
                }

                streamer = st.session_state.detector.stream_video(
                    rt_path,
                    tracker_yaml=tracker_yaml,
                    conf=conf,
                    iou=iou,
                    classes=selected_ids,
                    detect_colors=detect_colors,
                    show_labels=show_labels,
                    frame_skip=frame_skip,
                    augment=augment,
                    line_y=line_y,
                    imgsz=640,
                    roi_top=roi_top,
                    count_direction=count_direction,
                    meters_per_pixel=meters_per_pixel,
                    show_trails=show_trails,
                )

                for rgb, stats, fn, total_f, fps in streamer:
                    rt_cum["total"] += stats["total"]
                    for k, v in stats["classes"].items():
                        rt_cum["classes"][k] = (
                            rt_cum["classes"].get(k, 0) + v
                        )
                    for k, v in stats["colors"].items():
                        rt_cum["colors"][k] = (
                            rt_cum["colors"].get(k, 0) + v
                        )
                    rt_cum["unique_tracks"].update(
                        stats.get("track_ids", [])
                    )

                    frame_ph.image(rgb, width="stretch")
                    caption_ph.caption(
                        f"Frame {fn} / {total_f}"
                        f"  |  FPS: {fps:.1f}"
                        f"  |  Vehicles: {stats['total']}"
                    )

                    with stats_ph.container():
                        st.metric("Frame", f"{fn}/{total_f}")
                        st.metric("Vehicles", stats["total"])
                        st.metric("Cum. Detections", rt_cum["total"])
                        st.metric(
                            "Unique Tracks",
                            len(rt_cum["unique_tracks"]),
                        )
                        if line_enabled:
                            st.metric(
                                "Line Crossings",
                                stats.get("line_crossings", 0),
                            )
                        if stats["classes"]:
                            st.markdown("**Types**")
                            for cls, cnt in stats["classes"].items():
                                st.metric(cls.capitalize(), cnt)
                        if stats["colors"]:
                            st.markdown("**Colors**")
                            for color, cnt in list(
                                stats["colors"].items()
                            )[:4]:
                                st.metric(color, cnt)
                        vehicles_with_speed = [
                            v for v in stats.get("detected_vehicles", [])
                            if v.get("speed_info") is not None
                        ]
                        if vehicles_with_speed:
                            st.markdown("**Speed**")
                            for v in vehicles_with_speed[:4]:
                                si = v["speed_info"]
                                st.metric(
                                    f"#{v['vehicle_id']} {v['vehicle_type']}",
                                    f"{si['kph']:.0f} km/h",
                                    help=f"{si['direction_label']} · reliability {si['reliability']}",
                                )

                    rt_progress.progress(
                        fn / max(total_f, 1),
                        text=f"Frame {fn} / {total_f}  |  {fps:.1f} FPS",
                    )

                rt_progress.progress(1.0, text="Stream complete!")
                st.success("Real-time detection finished.")

                st.divider()
                st.subheader("Session Summary")
                render_stats(
                    rt_cum, unique_tracks=rt_cum["unique_tracks"]
                )

                try:
                    os.unlink(rt_path)
                except OSError:
                    pass
