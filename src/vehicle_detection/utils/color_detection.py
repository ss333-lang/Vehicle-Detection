"""Vehicle color detection using HSV color-space range matching.

The region of interest (ROI) is trimmed to the vehicle body area:
top and bottom fractions are discarded to avoid roof reflections
and road/tyre pixels, leaving the painted bodywork for sampling.
"""

import cv2
import numpy as np

# (name, hsv_lower, hsv_upper)
# Red occupies both ends of the HSV hue circle, so two ranges are needed.
_COLOR_RANGES: list[tuple[str, tuple, tuple]] = [
    ("White",   (  0,   0, 185), (180,  35, 255)),
    ("Black",   (  0,   0,   0), (180, 255,  50)),
    ("Silver",  (  0,   0,  55), (180,  35, 185)),
    ("Red",     (  0, 110,  80), ( 10, 255, 255)),
    ("Red",     (170, 110,  80), (180, 255, 255)),
    ("Orange",  ( 11,  90,  90), ( 25, 255, 255)),
    ("Yellow",  ( 26,  90,  90), ( 34, 255, 255)),
    ("Green",   ( 35,  80,  80), ( 85, 255, 255)),
    ("Blue",    ( 86,  80,  80), (130, 255, 255)),
    ("Purple",  (131,  80,  80), (160, 255, 255)),
    ("Brown",   ( 10, 100,  20), ( 20, 200, 150)),
]

# ROI crop margins as fractions of bounding-box height / width.
# Skipping the roof (top) and tyres/road (bottom) isolates the
# painted body panels where the true vehicle colour is visible.
_ROI_TOP_MARGIN: float = 0.15
_ROI_BOTTOM_MARGIN: float = 0.30
_ROI_SIDE_MARGIN: float = 0.08

# Reject ROIs too small for reliable colour statistics.
_MIN_ROI_SIZE: int = 4

# Minimum fraction of ROI pixels that must match a colour before
# it is accepted; prevents noise from dominating the result.
_MIN_COVERAGE: float = 0.08


def detect_vehicle_color(frame: np.ndarray, bbox: tuple) -> str:
    """Return the dominant colour name of the vehicle inside *bbox*.

    Args:
        frame (np.ndarray): Full BGR image from the video frame.
        bbox (tuple): Bounding-box coordinates as ``(x1, y1, x2, y2)``.

    Returns:
        str: Colour name (e.g. ``"Blue"``), or ``"Unknown"`` when pixel
            coverage is below the minimum threshold.
    """
    x1, y1, x2, y2 = map(int, bbox)
    fh, fw = frame.shape[:2]

    # Clamp coordinates to valid frame bounds before slicing.
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)

    if x2 <= x1 or y2 <= y1:
        return "Unknown"

    bh = y2 - y1
    bw = x2 - x1

    ry1 = y1 + int(bh * _ROI_TOP_MARGIN)
    ry2 = y2 - int(bh * _ROI_BOTTOM_MARGIN)
    rx1 = x1 + int(bw * _ROI_SIDE_MARGIN)
    rx2 = x2 - int(bw * _ROI_SIDE_MARGIN)

    # Fall back to the full box if margins collapse the ROI.
    if ry2 <= ry1 or rx2 <= rx1:
        ry1, ry2, rx1, rx2 = y1, y2, x1, x2

    roi = frame[ry1:ry2, rx1:rx2]

    if (
        roi.size == 0
        or roi.shape[0] < _MIN_ROI_SIZE
        or roi.shape[1] < _MIN_ROI_SIZE
    ):
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = roi.shape[0] * roi.shape[1]

    scores: dict[str, int] = {}
    for name, lo, hi in _COLOR_RANGES:
        mask = cv2.inRange(
            hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)
        )
        cnt = int(cv2.countNonZero(mask))
        # Both "Red" entries map to the same key so their pixel counts merge.
        scores[name] = scores.get(name, 0) + cnt

    best = max(scores, key=scores.get)
    coverage = scores[best] / max(total, 1)

    return best if coverage >= _MIN_COVERAGE else "Unknown"
