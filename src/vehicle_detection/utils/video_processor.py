"""Video processing pipeline.

Combines YOLO11m object detection with ByteTrack / BoT-SORT tracking
and optional per-vehicle colour detection into a single generator-based
pipeline that supports both batch output and real-time frame streaming.
"""

import pathlib
import time
from collections import Counter
from typing import Generator

import cv2
import numpy as np
from ultralytics import YOLO

from .color_detection import detect_vehicle_color


# ── Constants ──────────────────────────────────────────────────────────────────

# Custom Indian vehicle class IDs from trained best.pt.
VEHICLE_CLASSES: dict[int, str] = {
    0: "Auto Rickshaw",
    1: "Bus -2 axle",
    2: "Bus -3 axle",
    3: "Car",
    4: "E-Rickshaw",
    5: "Mini Loading",
    6: "Motor Cycle",
    7: "Scooty",
    8: "Truck",
}

# BGR bounding-box stroke colours, one per vehicle class.
CLASS_COLORS: dict[str, tuple] = {
    "Auto Rickshaw": (  0, 215, 255),
    "Bus -2 axle":   ( 50,  50, 255),
    "Bus -3 axle":   (  0,  80, 255),
    "Car":           ( 50, 205,  50),
    "E-Rickshaw":    (255, 180,   0),
    "Mini Loading":  (255,   0, 180),
    "Motor Cycle":   (255,   0, 255),
    "Scooty":        (180, 105, 255),
    "Truck":         (  0, 140, 255),
    "unknown":       (128, 128, 128),
}

import pathlib as _pl
_PROJECT_ROOT = _pl.Path(__file__).resolve().parents[3]

TRACKER_MAP: dict[str, str] = {
    "ByteSORT (ByteTrack)": str(_PROJECT_ROOT / "bytetrack.yaml"),
    "BoT-SORT":             str(_PROJECT_ROOT / "botsort.yaml"),
}

# Guard against zero-duration frames when computing FPS.
_FPS_EPSILON: float = 1e-6

# Used when a video file reports 0 FPS in its metadata.
_DEFAULT_FPS: float = 25.0

# OpenCV text rendering parameters for detection labels.
_LABEL_FONT_SCALE: float = 0.48
_LABEL_THICKNESS: int = 1
_LABEL_PAD_H: int = 4   # Horizontal padding inside label background.
_LABEL_PAD_V: int = 6   # Vertical clearance above the bounding box.


# ── Internal helpers ──────────────────────────────────────────────────────────

def _draw_label(
    img: np.ndarray,
    text: str,
    x1: int,
    y1: int,
    color: tuple,
) -> None:
    """Draw a filled label box with white text above a bounding-box corner.

    Args:
        img (np.ndarray): BGR image to draw onto (modified in place).
        text (str): Label string to render.
        x1 (int): Left x-coordinate of the bounding box.
        y1 (int): Top y-coordinate of the bounding box.
        color (tuple): BGR fill colour for the label background.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(
        text, font, _LABEL_FONT_SCALE, _LABEL_THICKNESS
    )
    cv2.rectangle(
        img,
        (x1, y1 - th - _LABEL_PAD_V),
        (x1 + tw + _LABEL_PAD_H, y1),
        color, -1,
    )
    cv2.putText(
        img, text,
        (x1 + 2, y1 - 4),
        font, _LABEL_FONT_SCALE,
        (255, 255, 255), _LABEL_THICKNESS, cv2.LINE_AA,
    )


# ── Public class ──────────────────────────────────────────────────────────────

class VehicleDetector:
    """YOLO11m detector with ByteTrack / BoT-SORT and colour recognition.

    Args:
        model_path (str): Path to YOLO weights. Defaults to
            ``"yolo11m.pt"``, which ultralytics downloads automatically.
    """

    def __init__(self, model_path: str = "models/best.pt") -> None:
        # Prefer ONNX over .pt for ~2-3x faster CPU inference.
        # Export is done once automatically if ONNX does not yet exist.
        pt_path = pathlib.Path(model_path)
        onnx_path = pt_path.with_suffix(".onnx")
        if onnx_path.exists():
            self.model = YOLO(str(onnx_path), task="detect")
        else:
            self.model = YOLO(str(pt_path), task="detect")
            try:
                self.model.export(format="onnx", imgsz=640, dynamic=False)
                if onnx_path.exists():
                    self.model = YOLO(str(onnx_path), task="detect")
            except Exception:
                pass  # If export fails, continue with .pt

        self._vehicle_positions: dict[int, float] = {}
        self._line_crossings: int = 0
        # Majority-vote colour buffer: accumulate up to _COLOR_VOTE_FRAMES
        # detections per track, then lock in the most common result.
        # Prevents the first (possibly shadowed) frame from locking a wrong colour.
        self._vehicle_color_votes: dict[int, list[str]] = {}
        self._vehicle_colors: dict[int, str] = {}
        # Smoothed bounding boxes per track: EMA keeps boxes stable as vehicles move.
        self._smooth_boxes: dict[int, list[float]] = {}
        # Track IDs already counted — prevents double-counting at the line.
        self._counted_ids: set[int] = set()
        # Last detection draw data — used to overlay boxes on skipped frames.
        self._last_draw_data: list[dict] = []

    # Number of per-track colour votes before locking in the final colour.
    _COLOR_VOTE_FRAMES: int = 5
    # EMA smoothing factor for bounding boxes (0=frozen, 1=no smoothing).
    _BOX_SMOOTH_ALPHA: float = 0.6

    def _reset_tracker(self) -> None:
        """Clear tracker state so track IDs restart for a new video."""
        try:
            self.model.predictor = None
        except AttributeError:
            pass
        self._vehicle_positions.clear()
        self._line_crossings = 0
        self._vehicle_color_votes.clear()
        self._vehicle_colors.clear()
        self._smooth_boxes.clear()
        self._counted_ids.clear()
        self._last_draw_data.clear()

    def process_frame(
        self,
        frame: np.ndarray,
        tracker_yaml: str = "bytetrack.yaml",
        conf: float = 0.50,
        iou: float = 0.45,
        classes: list[int] | None = None,
        detect_colors: bool = True,
        show_labels: bool = True,
        augment: bool = False,
        line_y: float | None = None,
        imgsz: int = 1280,
        roi_top: float = 0.0,
        count_direction: str = "down",
    ) -> tuple[np.ndarray, dict]:
        """Run detection, tracking, and optional colour detection on *frame*.

        Args:
            frame (np.ndarray): BGR image from the video capture.
            tracker_yaml (str): Tracker config filename, e.g.
                ``"bytetrack.yaml"`` or ``"botsort.yaml"``.
            conf (float): Minimum detection confidence threshold.
            iou (float): IoU threshold for non-maximum suppression.
            classes (list[int] | None): COCO class IDs to detect.
                Defaults to all entries in ``VEHICLE_CLASSES``.
            detect_colors (bool): Run per-vehicle colour detection when
                ``True``.
            show_labels (bool): Draw text labels on the annotated frame
                when ``True``.

        Returns:
            tuple: A pair ``(annotated_bgr, stats)`` where:

                - ``annotated_bgr`` (np.ndarray): Frame with bounding
                  boxes and labels drawn (BGR colour space).
                - ``stats`` (dict): Keys are ``total`` (int),
                  ``classes`` (dict[str, int]),
                  ``colors`` (dict[str, int]),
                  ``track_ids`` (list[int]).
        """
        if classes is None:
            classes = list(VEHICLE_CLASSES.keys())

        fh, fw = frame.shape[:2]

        # Apply ROI mask: black out the top fraction of the frame so YOLO
        # never detects vehicles on overhead bridges or in the background.
        if roi_top > 0.0:
            inference_frame = frame.copy()
            mask_h = int(fh * roi_top)
            inference_frame[:mask_h, :] = 0
        else:
            inference_frame = frame

        results = self.model.track(
            source=inference_frame,
            persist=True,
            tracker=tracker_yaml,
            conf=conf,
            iou=iou,
            classes=classes,
            imgsz=imgsz,
            augment=augment,
            verbose=False,
        )

        stats: dict = {
            "total": 0, "classes": {}, "colors": {}, "track_ids": [],
            "line_crossings": self._line_crossings,
        }
        annotated = frame.copy()

        # Draw ROI boundary line so the operator can see the masked zone.
        if roi_top > 0.0:
            mask_h = int(fh * roi_top)
            cv2.line(annotated, (0, mask_h), (fw, mask_h), (80, 80, 80), 1)

        # Precompute line pixel position for crossing detection.
        line_px: int | None = int(line_y * fh) if line_y is not None else None

        self._last_draw_data.clear()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                track_id = int(box.id[0]) if box.id is not None else -1

                cls_name = VEHICLE_CLASSES.get(cls_id, "unknown")
                box_color = CLASS_COLORS.get(cls_name, CLASS_COLORS["unknown"])

                v_color = "N/A"
                if detect_colors:
                    if track_id >= 0 and track_id in self._vehicle_colors:
                        # Colour already locked in by majority vote.
                        v_color = self._vehicle_colors[track_id]
                    else:
                        raw = detect_vehicle_color(frame, xyxy)
                        if track_id >= 0 and raw not in ("N/A", "Unknown"):
                            votes = self._vehicle_color_votes.setdefault(track_id, [])
                            votes.append(raw)
                            if len(votes) >= self._COLOR_VOTE_FRAMES:
                                # Lock in the majority colour across the vote window.
                                winner = Counter(votes).most_common(1)[0][0]
                                self._vehicle_colors[track_id] = winner
                                v_color = winner
                            else:
                                # Still collecting votes; show current best guess.
                                v_color = Counter(votes).most_common(1)[0][0]
                        else:
                            v_color = raw

                # Accumulate per-frame statistics.
                stats["total"] += 1
                cls_counts = stats["classes"]
                cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1

                if v_color not in ("N/A", "Unknown"):
                    clr_counts = stats["colors"]
                    clr_counts[v_color] = clr_counts.get(v_color, 0) + 1

                x1, y1, x2, y2 = map(int, xyxy)

                # Tighten box: shrink each side by 3 % of box dimensions
                # to remove the Kalman-filter prediction padding.
                _bw = x2 - x1
                _bh = y2 - y1
                _pad = 0.03
                x1 = int(x1 + _bw * _pad)
                y1 = int(y1 + _bh * _pad)
                x2 = int(x2 - _bw * _pad)
                y2 = int(y2 - _bh * _pad)

                # Cap box size: Kalman filter can produce unrealistically large
                # predicted boxes for fast-moving or partially-visible vehicles.
                # Clamp each dimension to 35 % of the frame to prevent this.
                _max_w = int(fw * 0.35)
                _max_h = int(fh * 0.35)
                _cx = (x1 + x2) // 2
                _cy = (y1 + y2) // 2
                _bw = min(x2 - x1, _max_w)
                _bh = min(y2 - y1, _max_h)
                x1 = _cx - _bw // 2
                y1 = _cy - _bh // 2
                x2 = _cx + _bw // 2
                y2 = _cy + _bh // 2

                # EMA box smoothing: blend current box with previous smoothed box
                # so the rectangle adapts gradually rather than jumping each frame.
                if track_id >= 0:
                    raw_box = [float(x1), float(y1), float(x2), float(y2)]
                    if track_id in self._smooth_boxes:
                        prev = self._smooth_boxes[track_id]
                        a = self._BOX_SMOOTH_ALPHA
                        smoothed = [a * r + (1 - a) * p for r, p in zip(raw_box, prev)]
                    else:
                        smoothed = raw_box
                    self._smooth_boxes[track_id] = smoothed
                    x1, y1, x2, y2 = (int(v) for v in smoothed)

                # Line crossing detection using track centre Y.
                if line_px is not None and track_id >= 0:
                    center_y = (y1 + y2) / 2.0
                    prev_y = self._vehicle_positions.get(track_id)
                    if prev_y is not None and track_id not in self._counted_ids:
                        went_down = prev_y < line_px <= center_y
                        went_up   = prev_y > line_px >= center_y
                        crossed = (
                            (count_direction == "down" and went_down) or
                            (count_direction == "up"   and went_up)   or
                            (count_direction == "both" and (went_down or went_up))
                        )
                        if crossed:
                            self._counted_ids.add(track_id)
                            self._line_crossings += 1
                            stats["line_crossings"] = self._line_crossings
                    self._vehicle_positions[track_id] = center_y

                if track_id >= 0:
                    stats["track_ids"].append(track_id)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                label = ""
                if show_labels:
                    id_part = f"#{track_id} " if track_id >= 0 else ""
                    color_part = (
                        f" | {v_color}"
                        if v_color not in ("N/A", "Unknown")
                        else ""
                    )
                    label = (
                        f"{id_part}{cls_name.upper()}"
                        f" {conf_val:.2f}{color_part}"
                    )
                    _draw_label(annotated, label, x1, y1, box_color)

                # Store box info so skipped frames can re-draw on the live frame.
                self._last_draw_data.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "box_color": box_color, "label": label,
                })

        # Draw counting line over all bounding boxes.
        if line_px is not None:
            cv2.line(annotated, (0, line_px), (fw, line_px), (0, 255, 255), 3)
            count_label = f"Count: {self._line_crossings}"
            (tw, th), _ = cv2.getTextSize(
                count_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.rectangle(
                annotated, (8, line_px - th - 12), (16 + tw, line_px - 2),
                (0, 200, 200), -1,
            )
            cv2.putText(
                annotated, count_label, (12, line_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA,
            )

        return annotated, stats

    def _overlay_last_boxes(
        self,
        frame: np.ndarray,
        line_px: int | None,
        show_labels: bool,
        roi_top: float,
    ) -> np.ndarray:
        """Draw last known boxes onto *frame* for smooth skipped-frame display."""
        fh, fw = frame.shape[:2]
        out = frame.copy()
        if roi_top > 0.0:
            cv2.line(out, (0, int(fh * roi_top)), (fw, int(fh * roi_top)), (80, 80, 80), 1)
        for d in self._last_draw_data:
            cv2.rectangle(out, (d["x1"], d["y1"]), (d["x2"], d["y2"]), d["box_color"], 2)
            if show_labels and d["label"]:
                _draw_label(out, d["label"], d["x1"], d["y1"], d["box_color"])
        if line_px is not None:
            cv2.line(out, (0, line_px), (fw, line_px), (0, 255, 255), 3)
            count_label = f"Count: {self._line_crossings}"
            (tw, th), _ = cv2.getTextSize(count_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(out, (8, line_px - th - 12), (16 + tw, line_px - 2), (0, 200, 200), -1)
            cv2.putText(out, count_label, (12, line_px - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        return out

    def stream_video(
        self,
        video_path: str,
        tracker_yaml: str = "bytetrack.yaml",
        conf: float = 0.50,
        iou: float = 0.45,
        classes: list[int] | None = None,
        detect_colors: bool = True,
        show_labels: bool = True,
        frame_skip: int = 1,
        augment: bool = False,
        line_y: float | None = None,
        imgsz: int = 640,
        roi_top: float = 0.0,
        count_direction: str = "down",
    ) -> Generator[tuple, None, None]:
        """Yield processed frames for real-time display in Streamlit.

        Runs detection on every frame for smooth tracking. Display is
        throttled to the video's native FPS so playback feels normal.
        ``frame_skip`` skips detection on intermediate frames on very
        slow hardware, trading tracking smoothness for speed.

        Yields:
            tuple: ``(rgb_frame, stats, frame_num, total_frames, fps)``
                where ``rgb_frame`` is ready for ``st.image()``.
        """
        self._reset_tracker()

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or _DEFAULT_FPS
        # Minimum wall-clock interval per frame to match native video speed.
        min_interval = 1.0 / src_fps

        fn = 0
        last_annotated: np.ndarray | None = None
        last_stats: dict = {
            "total": 0, "classes": {}, "colors": {},
            "track_ids": [], "line_crossings": 0,
        }

        try:
            while cap.isOpened():
                frame_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break
                fn += 1

                if fn % max(frame_skip, 1) == 1 or frame_skip <= 1 or last_annotated is None:
                    annotated, stats = self.process_frame(
                        frame, tracker_yaml, conf, iou,
                        classes, detect_colors, show_labels,
                        augment, line_y, imgsz, roi_top, count_direction,
                    )
                    last_annotated = annotated
                    last_stats = stats
                else:
                    # Overlay last known boxes on the CURRENT raw frame so the
                    # background video plays smoothly while detection catches up.
                    line_px = int(line_y * frame.shape[0]) if line_y is not None else None
                    annotated = self._overlay_last_boxes(frame, line_px, show_labels, roi_top)
                    stats = last_stats

                elapsed = time.perf_counter() - frame_start
                fps = 1.0 / max(elapsed, _FPS_EPSILON)

                # Throttle to native video FPS when processing is faster.
                remaining = min_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                yield rgb, stats, fn, total, fps
        finally:
            cap.release()

    def process_video(
        self,
        input_path: str,
        output_path: str,
        tracker_yaml: str = "bytetrack.yaml",
        conf: float = 0.50,
        iou: float = 0.45,
        classes: list[int] | None = None,
        detect_colors: bool = True,
        show_labels: bool = True,
        imgsz: int = 640,
    ) -> Generator[tuple, None, None]:
        """Process a full video, write annotated output, and yield progress.

        Saves the annotated video to *output_path* while yielding cumulative
        statistics so the caller can update a progress bar during processing.

        Args:
            input_path (str): Path to the source video file.
            output_path (str): Destination path for the annotated MP4.
            tracker_yaml (str): Tracker config filename.
            conf (float): Detection confidence threshold.
            iou (float): IoU threshold for NMS.
            classes (list[int] | None): COCO class IDs to detect.
            detect_colors (bool): Enable per-vehicle colour detection.
            show_labels (bool): Draw labels on each frame.

        Yields:
            tuple: ``(rgb_frame, cumulative_stats, frame_num,
                total_frames, fps)`` on every processed frame.
        """
        self._reset_tracker()

        cap = cv2.VideoCapture(input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or _DEFAULT_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

        fn = 0
        t0 = time.perf_counter()
        cum: dict = {
            "total": 0, "classes": {}, "colors": {}, "unique_tracks": set()
        }

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                fn += 1

                annotated, stats = self.process_frame(
                    frame, tracker_yaml, conf, iou,
                    classes, detect_colors, show_labels,
                    imgsz=imgsz,
                )
                writer.write(annotated)

                t1 = time.perf_counter()
                fps = 1.0 / max(t1 - t0, _FPS_EPSILON)
                t0 = t1

                cum["total"] += stats["total"]
                for k, v in stats["classes"].items():
                    cum["classes"][k] = cum["classes"].get(k, 0) + v
                for k, v in stats["colors"].items():
                    cum["colors"][k] = cum["colors"].get(k, 0) + v
                cum["unique_tracks"].update(stats["track_ids"])

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                yield rgb, cum, fn, total, fps
        finally:
            cap.release()
            writer.release()
