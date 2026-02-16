# ct_utils.py
from __future__ import annotations

import io
import base64

import numpy as np
import PIL.Image
import pydicom


# MedGemma 1.5 CT windowing: (min_hu, max_hu) per RGB channel
WINDOW_CLIPS = [(-1024, 1024), (-135, 215), (0, 80)]


def window_ct_slice(ct_slice: np.ndarray) -> np.ndarray:
    """Apply 3-channel RGB windowing to a CT slice in Hounsfield Units.

    Red: wide (-1024..1024), Green: soft tissue (-135..215), Blue: brain (0..80).
    Returns uint8 RGB array.
    """
    channels = []
    for min_hu, max_hu in WINDOW_CLIPS:
        ch = np.clip(ct_slice.astype(np.float32), min_hu, max_hu)
        ch = (ch - min_hu) / (max_hu - min_hu) * 255.0
        channels.append(np.round(ch).astype(np.uint8))
    return np.stack(channels, axis=-1)


def sample_slices(slices: list, max_slices: int) -> list:
    """Uniformly sample up to max_slices from an ordered list."""
    if len(slices) <= max_slices:
        return slices
    n = len(slices)
    indices = [int(round(i / max_slices * (n - 1))) for i in range(1, max_slices + 1)]
    return [slices[i] for i in indices]


def parse_dicom_files(uploaded_files: list) -> list[np.ndarray]:
    """Read uploaded DICOM files, sort by InstanceNumber, return HU pixel arrays."""
    instances = []
    for f in uploaded_files:
        dcm = pydicom.dcmread(f)
        if not hasattr(dcm, "pixel_array"):
            continue
        instance_num = int(getattr(dcm, "InstanceNumber", 0))
        hu_pixels = pydicom.pixels.apply_rescale(dcm.pixel_array, dcm)
        instances.append((instance_num, hu_pixels))
    instances.sort(key=lambda x: x[0])
    return [pixels for _, pixels in instances]


def slices_to_gif_bytes(windowed_slices: list[np.ndarray]) -> bytes:
    """Convert list of windowed uint8 RGB arrays to an animated GIF."""
    if not windowed_slices:
        raise ValueError("At least one slice is required to create a GIF")
    images = [PIL.Image.fromarray(s) for s in windowed_slices]
    buf = io.BytesIO()
    images[0].save(
        buf,
        format="GIF",
        loop=0,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=len(images) * 3,
    )
    return buf.getvalue()


def encode_slice_base64(data: np.ndarray, fmt: str = "jpeg") -> str:
    """Base64-encode a windowed CT slice as a data URI."""
    buf = io.BytesIO()
    PIL.Image.fromarray(data).save(buf, format=fmt)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt};base64,{encoded}"


def build_messages(
    windowed_slices: list[np.ndarray],
    instruction: str,
    query: str,
) -> list[dict]:
    """Build the chat-completion messages list for MedGemma."""
    content = [{"type": "text", "text": instruction}]
    for i, ct_slice in enumerate(windowed_slices, 1):
        content.append({"type": "image", "image": encode_slice_base64(ct_slice)})
        content.append({"type": "text", "text": f"SLICE {i}"})
    content.append({"type": "text", "text": query})
    return [{"role": "user", "content": content}]
