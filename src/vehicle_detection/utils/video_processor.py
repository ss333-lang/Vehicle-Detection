"""Video processing pipeline.

Combines YOLO11m object detection with ByteTrack / BoT-SORT tracking,
optional per-vehicle colour detection, speed estimation (KPH), track
trail visualisation, and structured JSON output into a single
generator-based pipeline that supports both batch output and real-time
frame streaming.
"""

import math
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
_LABEL_FONT_SCALE: float = 0.62
_LABEL_THICKNESS: int = 2
_LABEL_PAD_H: int = 6
_LABEL_PAD_V: int = 8

# 8-directional labels indexed clockwise from East (Right).
_DIRECTION_LABELS: list[str] = [
    "Right", "Bottom-Right", "Bottom", "Bottom-Left",
    "Left", "Top-Left", "Top", "Top-Right",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _draw_label(
    img: np.ndarray,
    text: str,
    x1: int,
    y1: int,
    color: tuple,
) -> None:
    """Draw a filled label box with white text above a bounding-box corner."""
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


def _direction_label(angle_rad: float) -> str:
    """Map an atan2 angle in radians to one of 8 compass direction labels."""
    angle_deg = math.degrees(angle_rad) % 360
    idx = round(angle_deg / 45) % 8
    return _DIRECTION_LABELS[idx]


# ── Public class ──────────────────────────────────────────────────────────────

class VehicleDetector:
    """YOLO11m detector with ByteTrack / BoT-SORT, colour recognition,
    speed estimation (KPH), and track trail visualisation.

    Args:
        model_path (str): Path to YOLO weights. Defaults to
            ``"models/best.pt"``.
    """

    def __init__(self, model_path: str = "models/best.pt") -> None:
        import torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = YOLO(str(model_path), task="detect")

        self._vehicle_positions: dict[int, float] = {}
        self._line_crossings: int = 0
        # Majority-vote colour buffer per track ID.
        self._vehicle_color_votes: dict[int, list[str]] = {}
        self._vehicle_colors: dict[int, str] = {}
        # EMA-smoothed bounding boxes per track ID.
        self._smooth_boxes: dict[int, list[float]] = {}
        # Track IDs already counted at the line (prevents double-counting).
        self._counted_ids: set[int] = set()
        # Last detection draw data — used to overlay boxes on skipped frames.
        self._last_draw_data: list[dict] = []
        # Centroid history per track for speed & direction: (cx, cy, timestamp).
        self._track_history: dict[int, list[tuple[float, float, float]]] = {}

    _COLOR_VOTE_FRAMES: int = 5
    _BOX_SMOOTH_ALPHA: float = 0.2
    # Max positions kept per track (mirrors sergio11's 30-point trail).
    _TRACK_HISTORY_MAX: int = 30
    # Minimum history points required before a speed estimate is returned.
    _SPEED_MIN_SAMPLES: int = 3

    def _reset_tracker(self) -> None:
        """Clear all tracker state so track IDs restart for a new video."""
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
        self._track_history.clear()

    def _compute_speed_direction(
        self,
        track_id: int,
        cx: float,
        cy: float,
        t: float,
        meters_per_pixel: float,
    ) -> dict | None:
        """Append centroid to history, then compute KPH speed and direction.

        Args:
            track_id: Unique tracker ID.
            cx, cy: Current centroid in pixels.
            t: Timestamp in seconds (use ``frame_number / video_fps``).
            meters_per_pixel: Real-world scale (metres per pixel).

        Returns:
            dict with keys ``kph``, ``reliability``, ``direction_label``,
            ``direction`` (radians), or ``None`` if history is too short.
        """
        history = self._track_history.setdefault(track_id, [])
        history.append((cx, cy, t))
        if len(history) > self._TRACK_HISTORY_MAX:
            history.pop(0)

        if len(history) < self._SPEED_MIN_SAMPLES:
            return None

        # Direction: vector from oldest to newest centroid.
        x0, y0, _ = history[0]
        xl, yl, _ = history[-1]
        angle_rad = math.atan2(yl - y0, xl - x0)

        # Speed: mean of per-interval pixel distances converted to m/s → KPH.
        speeds_mps: list[float] = []
        for i in range(1, len(history)):
            px, py, pt = history[i]
            qx, qy, qt = history[i - 1]
            dt = pt - qt
            if dt <= 0:
                continue
            dist_m = math.hypot(px - qx, py - qy) * meters_per_pixel
            speeds_mps.append(dist_m / dt)

        if not speeds_mps:
            return None

        kph = (sum(speeds_mps) / len(speeds_mps)) * 3.6

        n = len(history)
        reliability = 0.5 if n < 5 else (0.7 if n < 10 else 1.0)

        return {
            "kph": round(kph, 1),
            "reliability": reliability,
            "direction_label": _direction_label(angle_rad),
            "direction": round(angle_rad, 4),
        }

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
        meters_per_pixel: float = 0.05,
        show_trails: bool = True,
        frame_time: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Run detection, tracking, colour detection, and speed estimation on *frame*.

        Args:
            frame: BGR image from the video capture.
            tracker_yaml: Tracker config filename.
            conf: Minimum detection confidence threshold.
            iou: IoU threshold for non-maximum suppression.
            classes: Class IDs to detect. Defaults to all VEHICLE_CLASSES.
            detect_colors: Run per-vehicle colour detection when True.
            show_labels: Draw text labels on the annotated frame when True.
            augment: Use test-time augmentation for distant detection.
            line_y: Counting line position as fraction of frame height (0–1).
            imgsz: Inference image size in pixels.
            roi_top: Fraction of frame top to mask out (0–1).
            count_direction: "down", "up", or "both" for counting line.
            meters_per_pixel: Real-world scale (m/px) for speed estimation.
                Requires camera calibration for accuracy. Default 0.05 is a
                rough estimate for a typical traffic camera.
            show_trails: Draw 30-point centroid trail polylines when True.
            frame_time: Timestamp for this frame in seconds. Pass
                ``frame_number / video_fps`` for accurate speed in batch mode.
                Falls back to ``time.time()`` when None.

        Returns:
            tuple: ``(annotated_bgr, stats)`` where stats includes
                ``detected_vehicles`` (list of per-vehicle JSON dicts).
        """
        if classes is None:
            classes = list(VEHICLE_CLASSES.keys())

        t_now = frame_time if frame_time is not None else time.time()
        fh, fw = frame.shape[:2]

        # Black out the top fraction of the frame before YOLO inference.
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
            device=self.device,
            half=False,
            verbose=False,
        )

        stats: dict = {
            "total": 0,
            "classes": {},
            "colors": {},
            "track_ids": [],
            "line_crossings": self._line_crossings,
            "detected_vehicles": [],
        }
        annotated = frame.copy()

        # Draw ROI boundary line so the operator can see the masked zone.
        if roi_top > 0.0:
            mask_h = int(fh * roi_top)
            cv2.line(annotated, (0, mask_h), (fw, mask_h), (80, 80, 80), 1)

        line_px: int | None = int(line_y * fh) if line_y is not None else None

        self._last_draw_data.clear()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id   = int(box.cls[0])
                conf_val = float(box.conf[0])
                xyxy     = box.xyxy[0].cpu().numpy()
                track_id = int(box.id[0]) if box.id is not None else -1

                cls_name  = VEHICLE_CLASSES.get(cls_id, "unknown")
                box_color = CLASS_COLORS.get(cls_name, CLASS_COLORS["unknown"])

                # ── Colour detection ────────────────────────────────────────
                v_color = "N/A"
                if detect_colors:
                    if track_id >= 0 and track_id in self._vehicle_colors:
                        v_color = self._vehicle_colors[track_id]
                    else:
                        raw = detect_vehicle_color(frame, xyxy)
                        if track_id >= 0 and raw not in ("N/A", "Unknown"):
                            votes = self._vehicle_color_votes.setdefault(track_id, [])
                            votes.append(raw)
                            if len(votes) >= self._COLOR_VOTE_FRAMES:
                                winner = Counter(votes).most_common(1)[0][0]
                                self._vehicle_colors[track_id] = winner
                                v_color = winner
                            else:
                                v_color = Counter(votes).most_common(1)[0][0]
                        else:
                            v_color = raw

                stats["total"] += 1
                cls_counts = stats["classes"]
                cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1

                if v_color not in ("N/A", "Unknown"):
                    clr_counts = stats["colors"]
                    clr_counts[v_color] = clr_counts.get(v_color, 0) + 1

                x1, y1, x2, y2 = map(int, xyxy)

                # ── Bounding box refinement ─────────────────────────────────
                # Shrink 3 % per side to remove Kalman-filter padding.
                _bw = x2 - x1
                _bh = y2 - y1
                _pad = 0.03
                x1 = int(x1 + _bw * _pad)
                y1 = int(y1 + _bh * _pad)
                x2 = int(x2 - _bw * _pad)
                y2 = int(y2 - _bh * _pad)

                # Cap to 35 % of frame to prevent unrealistically large boxes.
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

                # ── EMA box smoothing ───────────────────────────────────────
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

                # ── Speed & direction estimation ────────────────────────────
                cx_f = (x1 + x2) / 2.0
                cy_f = (y1 + y2) / 2.0
                speed_info: dict | None = None
                if track_id >= 0:
                    speed_info = self._compute_speed_direction(
                        track_id, cx_f, cy_f, t_now, meters_per_pixel
                    )

                # ── Track history trail (30-point polyline) ─────────────────
                if show_trails and track_id >= 0 and track_id in self._track_history:
                    history = self._track_history[track_id]
                    if len(history) >= 2:
                        pts = np.array(
                            [(int(h[0]), int(h[1])) for h in history],
                            dtype=np.int32,
                        )
                        cv2.polylines(
                            annotated, [pts], False, box_color, 2, cv2.LINE_AA
                        )

                # ── Line crossing detection ─────────────────────────────────
                if line_px is not None and track_id >= 0:
                    prev_y = self._vehicle_positions.get(track_id)
                    if prev_y is not None and track_id not in self._counted_ids:
                        went_down = prev_y < line_px <= cy_f
                        went_up   = prev_y > line_px >= cy_f
                        crossed = (
                            (count_direction == "down" and went_down) or
                            (count_direction == "up"   and went_up)   or
                            (count_direction == "both" and (went_down or went_up))
                        )
                        if crossed:
                            self._counted_ids.add(track_id)
                            self._line_crossings += 1
                            stats["line_crossings"] = self._line_crossings
                    self._vehicle_positions[track_id] = cy_f

                if track_id >= 0:
                    stats["track_ids"].append(track_id)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

                # ── Label (type · conf · colour · KPH) ─────────────────────
                label = ""
                if show_labels:
                    id_part    = f"#{track_id} " if track_id >= 0 else ""
                    color_part = (
                        f" | {v_color}" if v_color not in ("N/A", "Unknown") else ""
                    )
                    speed_part = (
                        f" | {speed_info['kph']:.0f} km/h"
                        if speed_info is not None else ""
                    )
                    label = (
                        f"{id_part}{cls_name.upper()}"
                        f" {conf_val:.2f}{color_part}{speed_part}"
                    )
                    _draw_label(annotated, label, x1, y1, box_color)

                # ── Structured JSON record for this vehicle ─────────────────
                vehicle_record: dict = {
                    "vehicle_id": track_id,
                    "vehicle_type": cls_name,
                    "detection_confidence": round(conf_val, 4),
                    "color": v_color,
                    "speed_info": speed_info,
                    "vehicle_coordinates": {
                        "x": int(cx_f),
                        "y": int(cy_f),
                        "width":  x2 - x1,
                        "height": y2 - y1,
                    },
                }
                stats["detected_vehicles"].append(vehicle_record)

                self._last_draw_data.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "box_color": box_color, "label": label,
                    "track_id": track_id,
                })

        # ── Counting line overlay ───────────────────────────────────────────
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
        show_trails: bool = True,
    ) -> np.ndarray:
        """Draw last known boxes and trails onto *frame* for smooth skipped-frame display."""
        fh, fw = frame.shape[:2]
        out = frame.copy()
        if roi_top > 0.0:
            cv2.line(out, (0, int(fh * roi_top)), (fw, int(fh * roi_top)), (80, 80, 80), 1)
        for d in self._last_draw_data:
            tid = d.get("track_id", -1)
            if show_trails and tid >= 0 and tid in self._track_history:
                history = self._track_history[tid]
                if len(history) >= 2:
                    pts = np.array(
                        [(int(h[0]), int(h[1])) for h in history],
                        dtype=np.int32,
                    )
                    cv2.polylines(out, [pts], False, d["box_color"], 2, cv2.LINE_AA)
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
        meters_per_pixel: float = 0.05,
        show_trails: bool = True,
    ) -> Generator[tuple, None, None]:
        """Yield processed frames for real-time display in Streamlit.

        Yields:
            tuple: ``(rgb_frame, stats, frame_num, total_frames, fps)``
        """
        self._reset_tracker()

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or _DEFAULT_FPS
        min_interval = 1.0 / src_fps

        fn = 0
        last_annotated: np.ndarray | None = None
        last_stats: dict = {
            "total": 0, "classes": {}, "colors": {},
            "track_ids": [], "line_crossings": 0, "detected_vehicles": [],
        }

        try:
            while cap.isOpened():
                frame_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break
                fn += 1
                frame_time = fn / src_fps

                if fn % max(frame_skip, 1) == 1 or frame_skip <= 1 or last_annotated is None:
                    annotated, stats = self.process_frame(
                        frame, tracker_yaml, conf, iou,
                        classes, detect_colors, show_labels,
                        augment, line_y, imgsz, roi_top, count_direction,
                        meters_per_pixel, show_trails, frame_time,
                    )
                    last_annotated = annotated
                    last_stats = stats
                else:
                    line_px = int(line_y * frame.shape[0]) if line_y is not None else None
                    annotated = self._overlay_last_boxes(
                        frame, line_px, show_labels, roi_top, show_trails
                    )
                    stats = last_stats

                elapsed = time.perf_counter() - frame_start
                fps = 1.0 / max(elapsed, _FPS_EPSILON)

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
        meters_per_pixel: float = 0.05,
        show_trails: bool = True,
    ) -> Generator[tuple, None, None]:
        """Process a full video, write annotated output, and yield progress.

        Accumulates per-vehicle JSON records in ``cumulative_stats["all_detections"]``
        so the caller can offer a JSON download after processing completes.

        Yields:
            tuple: ``(rgb_frame, cumulative_stats, frame_num, total_frames, fps)``
        """
        self._reset_tracker()

        cap = cv2.VideoCapture(input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or _DEFAULT_FPS
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

        fn = 0
        t0 = time.perf_counter()
        cum: dict = {
            "total": 0, "classes": {}, "colors": {}, "unique_tracks": set(),
            "all_detections": [],
        }

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                fn += 1
                frame_time = fn / src_fps

                annotated, stats = self.process_frame(
                    frame, tracker_yaml, conf, iou,
                    classes, detect_colors, show_labels,
                    imgsz=imgsz,
                    meters_per_pixel=meters_per_pixel,
                    show_trails=show_trails,
                    frame_time=frame_time,
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
                if stats["detected_vehicles"]:
                    cum["all_detections"].extend(stats["detected_vehicles"])

                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                yield rgb, cum, fn, total, fps
        finally:
            cap.release()
            writer.release()
