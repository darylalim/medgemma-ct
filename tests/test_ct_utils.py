# tests/test_ct_utils.py
import numpy as np
import pytest
from ct_utils import window_ct_slice, sample_slices


class TestWindowCtSlice:
    def test_output_shape_is_rgb(self):
        ct_slice = np.zeros((512, 512), dtype=np.int16)
        result = window_ct_slice(ct_slice)
        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8

    def test_clipping_wide_window(self):
        # Values outside -1024..1024 should be clipped for red channel
        ct_slice = np.array([[2000, -2000]], dtype=np.int16)
        result = window_ct_slice(ct_slice)
        assert result[0, 0, 0] == 255  # 2000 clipped to 1024 -> max
        assert result[0, 1, 0] == 0    # -2000 clipped to -1024 -> min

    def test_zero_hu_maps_correctly(self):
        # 0 HU in wide window (-1024..1024): (0 - -1024) / 2048 * 255 = 127.5 -> 128
        ct_slice = np.array([[0]], dtype=np.int16)
        result = window_ct_slice(ct_slice)
        assert result[0, 0, 0] == 128  # red channel (wide)

    def test_brain_window_range(self):
        # Blue channel: 0..80
        # 40 HU -> (40-0)/80 * 255 = 127.5 -> 128
        ct_slice = np.array([[40]], dtype=np.int16)
        result = window_ct_slice(ct_slice)
        assert result[0, 0, 2] == 128  # blue channel (brain)


class TestSampleSlices:
    def test_no_sampling_when_under_max(self):
        slices = list(range(10))
        result = sample_slices(slices, max_slices=85)
        assert result == slices

    def test_samples_to_max(self):
        slices = list(range(200))
        result = sample_slices(slices, max_slices=85)
        assert len(result) == 85

    def test_preserves_order(self):
        slices = list(range(200))
        result = sample_slices(slices, max_slices=85)
        assert result == sorted(result)

    def test_single_slice(self):
        slices = [42]
        result = sample_slices(slices, max_slices=85)
        assert result == [42]
