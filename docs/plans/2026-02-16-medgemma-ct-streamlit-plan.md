# MedGemma CT Streamlit App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single-page Streamlit app that accepts DICOM CT uploads, previews windowed slices, and runs MedGemma 1.5 locally on MPS for CT analysis.

**Architecture:** Single file `streamlit_app.py` with pure-function helpers extracted to `ct_utils.py` for testability. Sidebar for config, main area for preview and results. Model loaded once via `@st.cache_resource`.

**Tech Stack:** Streamlit, PyTorch (MPS), Hugging Face Transformers, pydicom, Pillow, NumPy

---

### Task 1: CT utility functions — windowing and sampling

**Files:**
- Create: `ct_utils.py`
- Create: `tests/test_ct_utils.py`

**Step 1: Write failing tests for `window_ct_slice` and `sample_slices`**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/medgemma-ct" && python -m pytest tests/test_ct_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ct_utils'`

**Step 3: Implement `ct_utils.py`**

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/medgemma-ct" && python -m pytest tests/test_ct_utils.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add ct_utils.py tests/test_ct_utils.py
git commit -m "feat: add CT utility functions with windowing, sampling, and prompt building"
```

---

### Task 2: Streamlit app — sidebar and DICOM upload

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Write the Streamlit app skeleton with sidebar and upload**

```python
# streamlit_app.py
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ct_utils import (
    build_messages,
    parse_dicom_files,
    sample_slices,
    slices_to_gif_bytes,
    window_ct_slice,
)

DEFAULT_INSTRUCTION = (
    "You are an instructor teaching medical students. You are "
    "analyzing a contiguous block of CT slices from the center of "
    "the abdomen. Please review the slices provided below carefully."
)

DEFAULT_QUERY = (
    "\n\nBased on the visual evidence in the slices provided above, "
    "is this image a good teaching example of liver pathology? "
    "Comment on hypodense lesions or other hepatic irregularities. "
    "Do not comment on findings outside the liver. Please provide "
    "your reasoning and conclude with a 'Final Answer: yes' or "
    "'Final Answer: no'."
)

st.set_page_config(page_title="MedGemma CT Analysis", layout="wide")
st.title("MedGemma CT Analysis")
st.caption(
    "Upload DICOM CT files to analyze with MedGemma 1.5. "
    "For educational purposes only — not for clinical use."
)

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader(
        "Upload DICOM files (.dcm)",
        type=["dcm"],
        accept_multiple_files=True,
    )
    max_slices = st.slider("Max slices", min_value=1, max_value=200, value=85)
    instruction = st.text_area("System instruction", value=DEFAULT_INSTRUCTION, height=150)
    query = st.text_area("Clinical question", value=DEFAULT_QUERY, height=150)
    analyze_btn = st.button("Analyze", type="primary", disabled=not uploaded_files)

# --- Main area ---
if not uploaded_files:
    st.info("Upload DICOM files in the sidebar to get started.")
    st.stop()

# Parse and window slices
with st.spinner("Reading DICOM files..."):
    hu_slices = parse_dicom_files(uploaded_files)
    if not hu_slices:
        st.error("No valid DICOM files with pixel data found.")
        st.stop()
    hu_slices = sample_slices(hu_slices, max_slices)
    windowed = [window_ct_slice(s) for s in hu_slices]

st.subheader(f"CT Preview ({len(windowed)} slices)")
gif_bytes = slices_to_gif_bytes(windowed)
st.image(gif_bytes, caption="Windowed CT slices (R=wide, G=soft tissue, B=brain)")

# --- Inference ---
if analyze_btn:
    st.subheader("Analysis Results")
    messages = build_messages(windowed, instruction, query)

    with st.spinner("Loading MedGemma model (first run may take a few minutes)..."):
        model, processor = load_model()

    with st.spinner("Running inference..."):
        response = run_inference(model, processor, messages)

    st.markdown(response)
```

**Step 2: Run the app to verify it loads (no crash)**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/medgemma-ct" && timeout 5 python -c "import streamlit_app" 2>&1 || true`
Expected: Imports succeed (Streamlit won't fully run outside `streamlit run`, but no import errors)

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add Streamlit app with sidebar config, DICOM upload, and CT preview"
```

---

### Task 3: Model loading and inference

**Files:**
- Modify: `streamlit_app.py`

**Step 1: Add model loading and inference functions**

Add these functions to `streamlit_app.py` before the sidebar section:

```python
import os
import torch
import transformers


@st.cache_resource
def load_model():
    """Load MedGemma model and processor once, cached across reruns."""
    model_id = "google/medgemma-1.5-4b-it"
    processor = transformers.AutoProcessor.from_pretrained(
        model_id,
        use_fast=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    return model, processor


def run_inference(model, processor, messages: list[dict]) -> str:
    """Run MedGemma inference and return the response text."""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        continue_final_message=False,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    )
    with torch.inference_mode():
        inputs = inputs.to(model.device)
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=2000)
    response = processor.post_process_image_text_to_text(
        generated, skip_special_tokens=True
    )
    decoded_input = processor.post_process_image_text_to_text(
        inputs["input_ids"], skip_special_tokens=True
    )
    result = response[0]
    idx = result.find(decoded_input[0])
    if 0 <= idx <= 2:
        result = result[idx + len(decoded_input[0]) :]
    return result
```

**Step 2: Verify imports work**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/medgemma-ct" && python -c "from streamlit_app import load_model, run_inference; print('OK')" 2>&1 || true`
Expected: OK (or Streamlit context warning, but no import error)

**Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: add MedGemma model loading and inference for MPS"
```

---

### Task 4: End-to-end manual test

**Files:** None (manual testing)

**Step 1: Run the Streamlit app**

Run: `cd "/Users/daryl-lim/Library/Mobile Documents/com~apple~CloudDocs/GitHub/medgemma-ct" && streamlit run streamlit_app.py`

**Step 2: Test the full flow**

1. Open `http://localhost:8501` in browser
2. Upload a set of `.dcm` files from a CT series
3. Verify the CT preview GIF displays correctly (colored, not grayscale)
4. Adjust the system instruction and clinical question
5. Click "Analyze" and verify MedGemma returns a meaningful response
6. Verify no errors in the terminal

**Step 3: Fix any issues found**

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete MedGemma CT Streamlit app"
```
