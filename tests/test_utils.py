"""Unit tests for vehicle_detection utility modules."""

import numpy as np
import pytest

from vehicle_detection.utils.color_detection import detect_vehicle_color
from vehicle_detection.utils.video_processor import (
    CLASS_COLORS,
    TRACKER_MAP,
    VEHICLE_CLASSES,
)


class TestVehicleClasses:
    """Tests for VEHICLE_CLASSES and CLASS_COLORS constants."""

    def test_car_is_defined(self) -> None:
        assert 3 in VEHICLE_CLASSES
        assert VEHICLE_CLASSES[3] == "Car"

    def test_all_classes_have_box_color(self) -> None:
        # Every detected class must have a corresponding render colour.
        for cls_name in VEHICLE_CLASSES.values():
            assert cls_name in CLASS_COLORS

    def test_tracker_map_yields_yaml_paths(self) -> None:
        for yaml_path in TRACKER_MAP.values():
            assert yaml_path.endswith(".yaml")


class TestDetectVehicleColor:
    """Tests for the HSV-based colour detection function."""

    def test_black_image_returns_black(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        color = detect_vehicle_color(frame, (10, 10, 90, 90))
        assert color == "Black"

    def test_white_image_returns_white(self) -> None:
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        color = detect_vehicle_color(frame, (10, 10, 90, 90))
        assert color == "White"

    def test_blue_image_returns_blue(self) -> None:
        # Pure blue in BGR: (255, 0, 0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 200   # Blue channel
        color = detect_vehicle_color(frame, (10, 10, 90, 90))
        assert color == "Blue"

    def test_inverted_bbox_returns_unknown(self) -> None:
        # x1 > x2 is an invalid box; function must not raise.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        color = detect_vehicle_color(frame, (90, 10, 10, 90))
        assert color == "Unknown"

    def test_zero_size_bbox_returns_unknown(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        color = detect_vehicle_color(frame, (50, 50, 50, 50))
        assert color == "Unknown"

    def test_out_of_bounds_bbox_is_clamped(self) -> None:
        # Coordinates beyond frame dimensions must not raise an exception.
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        color = detect_vehicle_color(frame, (-10, -10, 200, 200))
        assert isinstance(color, str)
