# tests/test_ct_utils.py
import io
import pathlib
from contextlib import ExitStack

import numpy as np
import PIL.Image
import pytest

from ct_utils import (
    build_messages,
    parse_dicom_files,
    sample_slices,
    slices_to_gif_bytes,
    window_ct_slice,
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "samples"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_rgb():
    """4x4 random uint8 RGB array."""
    return np.random.default_rng(0).integers(0, 256, (4, 4, 3), dtype=np.uint8)


@pytest.fixture
def dicom_paths():
    """Sorted list of real DICOM file paths in the test data directory."""
    paths = sorted(DATA_DIR.glob("*.dcm"))
    if not paths:
        pytest.skip("No DICOM test data found")
    return paths


# ---------------------------------------------------------------------------
# window_ct_slice
# ---------------------------------------------------------------------------


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
        assert result[0, 1, 0] == 0  # -2000 clipped to -1024 -> min

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

    def test_soft_tissue_window(self):
        # Green channel: -135..215 (range=350)
        # 40 HU -> (40 - -135) / 350 * 255 = 175/350*255 = 127.5 -> 128
        ct_slice = np.array([[40]], dtype=np.int16)
        result = window_ct_slice(ct_slice)
        assert result[0, 0, 1] == 128  # green channel (soft tissue)

    def test_all_channels_min_at_low_hu(self):
        ct_slice = np.array([[-2000]], dtype=np.int16)
        result = window_ct_slice(ct_slice)
        assert result[0, 0, 0] == 0  # wide min
        assert result[0, 0, 1] == 0  # soft tissue min
        assert result[0, 0, 2] == 0  # brain min

    def test_all_channels_max_at_high_hu(self):
        ct_slice = np.array([[2000]], dtype=np.int16)
        result = window_ct_slice(ct_slice)
        assert result[0, 0, 0] == 255  # wide max
        assert result[0, 0, 1] == 255  # soft tissue max
        assert result[0, 0, 2] == 255  # brain max

    def test_accepts_float_input(self):
        ct_slice = np.array([[40.0]], dtype=np.float64)
        result = window_ct_slice(ct_slice)
        assert result.dtype == np.uint8
        assert result.shape == (1, 1, 3)


# ---------------------------------------------------------------------------
# sample_slices
# ---------------------------------------------------------------------------


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

    def test_exact_max(self):
        slices = list(range(85))
        result = sample_slices(slices, max_slices=85)
        assert result == slices

    def test_includes_first_element(self):
        slices = list(range(200))
        result = sample_slices(slices, max_slices=10)
        assert result[0] == 0

    def test_includes_last_element(self):
        slices = list(range(200))
        result = sample_slices(slices, max_slices=10)
        assert result[-1] == 199

    def test_max_slices_one_returns_first(self):
        slices = list(range(200))
        result = sample_slices(slices, max_slices=1)
        assert result == [0]


# ---------------------------------------------------------------------------
# parse_dicom_files
# ---------------------------------------------------------------------------


class TestParseDicomFiles:
    def test_parses_real_dicoms(self, dicom_paths):
        with ExitStack() as stack:
            files = [stack.enter_context(open(p, "rb")) for p in dicom_paths]
            result = parse_dicom_files(files)
        assert len(result) > 0
        assert all(isinstance(s, np.ndarray) for s in result)
        assert all(s.ndim == 2 for s in result)

    def test_returns_sorted_by_instance(self, dicom_paths):
        # Parse in reversed file order — result should still be instance-sorted
        with ExitStack() as stack:
            files = [stack.enter_context(open(p, "rb")) for p in reversed(dicom_paths)]
            result_reversed = parse_dicom_files(files)

        with ExitStack() as stack:
            files = [stack.enter_context(open(p, "rb")) for p in dicom_paths]
            result_normal = parse_dicom_files(files)

        for a, b in zip(result_reversed, result_normal):
            np.testing.assert_array_equal(a, b)

    def test_empty_list(self):
        result = parse_dicom_files([])
        assert result == []

    def test_bytesio_input(self, dicom_paths):
        buffers = [io.BytesIO(p.read_bytes()) for p in dicom_paths[:2]]
        result = parse_dicom_files(buffers)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# slices_to_gif_bytes
# ---------------------------------------------------------------------------


class TestSlicesToGifBytes:
    def test_returns_gif_bytes(self, small_rgb):
        result = slices_to_gif_bytes([small_rgb])
        assert isinstance(result, bytes)
        assert result[:6] in (b"GIF87a", b"GIF89a")

    def test_multiple_frames(self, small_rgb):
        frames = [small_rgb, small_rgb, small_rgb]
        result = slices_to_gif_bytes(frames)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one slice"):
            slices_to_gif_bytes([])


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_structure_single_slice(self, small_rgb):
        msgs, images = build_messages([small_rgb], "instruction", "query")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        # instruction + image + SLICE 1 + query = 4 items
        assert len(content) == 4
        assert content[0] == {"type": "text", "text": "instruction"}
        assert content[1] == {"type": "image"}
        assert content[2] == {"type": "text", "text": "SLICE 1"}
        assert content[3] == {"type": "text", "text": "query"}
        assert len(images) == 1
        assert isinstance(images[0], PIL.Image.Image)
        assert images[0].mode == "RGB"
        assert images[0].size == (4, 4)

    def test_structure_multiple_slices(self, small_rgb):
        slices = [small_rgb, small_rgb, small_rgb]
        msgs, images = build_messages(slices, "inst", "q")
        content = msgs[0]["content"]
        # inst + 3*(image + slice_label) + query = 1 + 6 + 1 = 8
        assert len(content) == 8
        text_items = [c["text"] for c in content if c["type"] == "text"]
        assert text_items == ["inst", "SLICE 1", "SLICE 2", "SLICE 3", "q"]
        assert len(images) == 3
        assert all(isinstance(img, PIL.Image.Image) for img in images)

    def test_image_entries_are_placeholders(self, small_rgb):
        slices = [small_rgb, small_rgb, small_rgb]
        msgs, images = build_messages(slices, "inst", "q")
        content = msgs[0]["content"]
        image_items = [c for c in content if c["type"] == "image"]
        assert len(image_items) == 3
        assert all(item == {"type": "image"} for item in image_items)

    def test_empty_slices(self):
        msgs, images = build_messages([], "inst", "q")
        content = msgs[0]["content"]
        # instruction + query only
        assert len(content) == 2
        assert content[0]["text"] == "inst"
        assert content[1]["text"] == "q"
        assert images == []
