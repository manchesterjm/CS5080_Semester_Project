"""Tests for shared module."""

import json

import numpy as np
import pytest

from shared import upscale, NumpyEncoder
from constants import DISPLAY_SIZE


class TestUpscale:
    def test_output_size(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = upscale(frame)
        assert result.shape == (DISPLAY_SIZE, DISPLAY_SIZE, 3)

    def test_nearest_neighbor_preserves_colors(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[0, 0] = [255, 0, 0]
        result = upscale(frame)
        # The top-left pixel block should be red
        assert np.array_equal(result[0, 0], [255, 0, 0])

    def test_dtype_preserved(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = upscale(frame)
        assert result.dtype == np.uint8


class TestNumpyEncoder:
    def test_encodes_numpy_int(self):
        result = json.dumps({"val": np.int64(42)}, cls=NumpyEncoder)
        assert json.loads(result)["val"] == 42

    def test_encodes_numpy_float(self):
        result = json.dumps({"val": np.float64(3.14)}, cls=NumpyEncoder)
        assert abs(json.loads(result)["val"] - 3.14) < 1e-6

    def test_encodes_numpy_bool(self):
        result = json.dumps({"val": np.bool_(True)}, cls=NumpyEncoder)
        assert json.loads(result)["val"] is True

    def test_encodes_numpy_array(self):
        result = json.dumps({"val": np.array([1, 2, 3])}, cls=NumpyEncoder)
        assert json.loads(result)["val"] == [1, 2, 3]

    def test_raises_for_unknown_type(self):
        with pytest.raises(TypeError):
            json.dumps({"val": object()}, cls=NumpyEncoder)
