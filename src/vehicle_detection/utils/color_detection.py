"""Vehicle color detection using HSV color-space range matching.

The region of interest (ROI) is trimmed to the vehicle body area:
top and bottom fractions are discarded to avoid roof reflections
and road/tyre pixels, leaving the painted bodywork for sampling.

CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied
to the Value channel before matching so that colours detected under
different lighting conditions (shadows, glare, overcast) remain stable.
"""

import cv2
import numpy as np

# (name, hsv_lower, hsv_upper)
# Ranges tuned for real Indian road footage under mixed lighting.
# Red wraps the HSV hue circle so two ranges are required.
# White / Silver / Black use only Saturation + Value (hue-independent).
_COLOR_RANGES: list[tuple[str, tuple, tuple]] = [
    # Achromatic colours — hue is irrelevant, only S and V matter.
    ("White",   (  0,   0, 180), (180,  40, 255)),  # bright & desaturated
    ("Black",   (  0,   0,   0), (180, 255,  55)),  # very dark
    ("Silver",  (  0,   0,  56), (180,  40, 179)),  # mid-brightness, desaturated

    # Chromatic colours — tight hue bands + minimum saturation & brightness.
    ("Red",     (  0, 120,  70), ( 10, 255, 255)),  # lower hue red
    ("Red",     (165, 120,  70), (180, 255, 255)),  # upper hue red (wrap)
    ("Orange",  ( 11, 120,  80), ( 22, 255, 255)),
    ("Yellow",  ( 23, 120, 100), ( 35, 255, 255)),
    ("Green",   ( 36,  80,  50), ( 85, 255, 255)),
    ("Blue",    ( 86,  90,  50), (130, 255, 255)),
    ("Purple",  (131,  70,  50), (165, 255, 255)),
    ("Brown",   ( 10, 100,  30), ( 22, 210, 140)),
]

# ROI crop margins as fractions of bounding-box height / width.
# Skipping the roof (top) and tyres/road (bottom) isolates the
# painted body panels where the true vehicle colour is visible.
_ROI_TOP_MARGIN: float    = 0.18   # skip roof / windscreen reflection
_ROI_BOTTOM_MARGIN: float = 0.28   # skip tyres, road surface
_ROI_SIDE_MARGIN: float   = 0.10   # skip shadow edges

# Reject ROIs too small for reliable colour statistics.
_MIN_ROI_SIZE: int = 8

# Minimum fraction of ROI pixels that must match a colour before
# it is accepted; prevents noise from dominating the result.
_MIN_COVERAGE: float = 0.12

# CLAHE parameters for Value-channel lighting normalisation.
_CLAHE_CLIP: float    = 2.0
_CLAHE_TILE: tuple    = (4, 4)


def detect_vehicle_color(frame: np.ndarray, bbox: tuple) -> str:
    """Return the dominant colour name of the vehicle inside *bbox*.

    Applies CLAHE on the HSV Value channel before range matching so
    that detections remain stable across varying lighting conditions
    (shadows, direct sunlight, overcast sky).

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

    # Convert to HSV then apply CLAHE on the Value channel only.
    # This normalises brightness without altering hue or saturation,
    # making colour matching robust to shadows and glare.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILE)
    v_eq = clahe.apply(v)
    hsv_eq = cv2.merge([h, s, v_eq])

    total = roi.shape[0] * roi.shape[1]

    scores: dict[str, int] = {}
    for name, lo, hi in _COLOR_RANGES:
        mask = cv2.inRange(
            hsv_eq, np.array(lo, np.uint8), np.array(hi, np.uint8)
        )
        cnt = int(cv2.countNonZero(mask))
        # Both "Red" entries map to the same key so their counts merge.
        scores[name] = scores.get(name, 0) + cnt

    best = max(scores, key=scores.get)
    coverage = scores[best] / max(total, 1)

    return best if coverage >= _MIN_COVERAGE else "Unknown"
