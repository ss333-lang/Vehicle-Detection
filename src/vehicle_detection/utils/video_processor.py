"""Video processing pipeline.

Combines YOLO11m object detection with ByteTrack / BoT-SORT tracking
and optional per-vehicle colour detection into a single generator-based
pipeline that supports both batch output and real-time frame streaming.
"""

import time
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
        self.model = YOLO(model_path)

    def _reset_tracker(self) -> None:
        """Clear tracker state so track IDs restart for a new video."""
        try:
            self.model.predictor = None
        except AttributeError:
            pass

    def process_frame(
        self,
        frame: np.ndarray,
        tracker_yaml: str = "bytetrack.yaml",
        conf: float = 0.50,
        iou: float = 0.45,
        classes: list[int] | None = None,
        detect_colors: bool = True,
        show_labels: bool = True,
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

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=tracker_yaml,
            conf=conf,
            iou=iou,
            classes=classes,
            imgsz=1280,
            verbose=False,
        )

        stats: dict = {
            "total": 0, "classes": {}, "colors": {}, "track_ids": []
        }
        annotated = frame.copy()

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
                    v_color = detect_vehicle_color(frame, xyxy)

                # Accumulate per-frame statistics.
                stats["total"] += 1
                cls_counts = stats["classes"]
                cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1

                if v_color not in ("N/A", "Unknown"):
                    clr_counts = stats["colors"]
                    clr_counts[v_color] = clr_counts.get(v_color, 0) + 1

                if track_id >= 0:
                    stats["track_ids"].append(track_id)

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

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

        return annotated, stats

    def stream_video(
        self,
        video_path: str,
        tracker_yaml: str = "bytetrack.yaml",
        conf: float = 0.50,
        iou: float = 0.45,
        classes: list[int] | None = None,
        detect_colors: bool = True,
        show_labels: bool = True,
    ) -> Generator[tuple, None, None]:
        """Yield processed frames for real-time display in Streamlit.

        Each iteration yields a tuple so the caller can update the UI
        immediately rather than waiting for the full video to finish.

        Args:
            video_path (str): Path to the source video file.
            tracker_yaml (str): Tracker config filename.
            conf (float): Detection confidence threshold.
            iou (float): IoU threshold for NMS.
            classes (list[int] | None): COCO class IDs to detect.
            detect_colors (bool): Enable per-vehicle colour detection.
            show_labels (bool): Draw labels on each frame.

        Yields:
            tuple: ``(rgb_frame, stats, frame_num, total_frames, fps)``
                where ``rgb_frame`` is ready for ``st.image()``.
        """
        self._reset_tracker()

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fn = 0
        t0 = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                fn += 1

                annotated, stats = self.process_frame(
                    frame, tracker_yaml, conf, iou,
                    classes, detect_colors, show_labels,
                )

                t1 = time.perf_counter()
                fps = 1.0 / max(t1 - t0, _FPS_EPSILON)
                t0 = t1

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
