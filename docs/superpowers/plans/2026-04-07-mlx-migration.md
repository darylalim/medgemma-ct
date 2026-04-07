# MLX Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CUDA/transformers inference backend with mlx-vlm for native Apple Silicon inference.

**Architecture:** Minimal swap — keep 2-file architecture (`ct_utils.py` + `streamlit_app.py`). Remove `encode_slice_base64`, update `build_messages` to return `(messages, images)` with PIL Images, rewrite model loading and inference in `streamlit_app.py` to use mlx-vlm with streaming output.

**Tech Stack:** mlx-vlm, Streamlit, numpy, PIL, pydicom

**Spec:** `docs/superpowers/specs/2026-04-07-mlx-migration-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Swap CUDA deps for mlx-vlm, simplify platform config |
| `ct_utils.py` | Modify | Remove `encode_slice_base64`, update `build_messages` return type |
| `streamlit_app.py` | Modify | Rewrite imports, `load_model()`, inference, and UI streaming |
| `tests/test_ct_utils.py` | Modify | Update `TestBuildMessages`, delete `TestEncodeSliceBase64` |
| `CLAUDE.md` | Modify | Update docs to reflect MLX/Apple Silicon |

---

### Task 1: Update dependencies in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:1-48`

- [ ] **Step 1: Replace dependencies and remove CUDA config**

Replace the full contents of `pyproject.toml` with:

```toml
[project]
name = "medgemma-ct"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "mlx-vlm",
    "numpy",
    "pillow",
    "pydicom",
    "python-dotenv",
    "streamlit",
]

[dependency-groups]
dev = [
    "pytest",
    "ruff",
    "ty",
]

[tool.uv]
required-environments = [
    "sys_platform == 'darwin' and platform_machine == 'arm64'",
]

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Install updated dependencies**

Run: `uv sync --dev`

Expected: Successful install. mlx-vlm and its transitive dependencies (mlx, transformers, etc.) are installed. torch, accelerate, flash-attn are no longer present.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: swap CUDA/torch deps for mlx-vlm"
```

---

### Task 2: Write failing tests for new `build_messages` signature

**Files:**
- Modify: `tests/test_ct_utils.py:1-279`

- [ ] **Step 1: Update imports — remove `base64` and `encode_slice_base64`**

Replace lines 1-17 of `tests/test_ct_utils.py`:

```python
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
```

- [ ] **Step 2: Delete `TestEncodeSliceBase64` class**

Delete the entire `TestEncodeSliceBase64` class (lines 213-235) and its section comment (lines 208-211).

- [ ] **Step 3: Rewrite `TestBuildMessages` class**

Replace the entire `TestBuildMessages` class (lines 243-278, including the section comment at lines 238-241) with:

```python
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
        msgs, images = build_messages([small_rgb], "inst", "q")
        content = msgs[0]["content"]
        image_items = [c for c in content if c["type"] == "image"]
        assert len(image_items) == 1
        assert image_items[0] == {"type": "image"}
        assert "image" not in image_items[0] or image_items[0]["type"] == "image"

    def test_empty_slices(self):
        msgs, images = build_messages([], "inst", "q")
        content = msgs[0]["content"]
        # instruction + query only
        assert len(content) == 2
        assert content[0]["text"] == "inst"
        assert content[1]["text"] == "q"
        assert images == []
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_ct_utils.py::TestBuildMessages -v`

Expected: FAIL — `build_messages` still returns `list[dict]`, not `tuple`. Tests that unpack `(msgs, images)` will raise `ValueError: too many values to unpack` or similar.

- [ ] **Step 5: Commit failing tests**

```bash
git add tests/test_ct_utils.py
git commit -m "test: update build_messages tests for new mlx-vlm signature"
```

---

### Task 3: Update `ct_utils.py` — remove `encode_slice_base64`, update `build_messages`

**Files:**
- Modify: `ct_utils.py:1-102`

- [ ] **Step 1: Remove `base64` import and `encode_slice_base64` function**

Remove the `import base64` from line 4.

Delete the entire `encode_slice_base64` function (lines 75-80).

- [ ] **Step 2: Rewrite `build_messages` function**

Replace the `build_messages` function (lines 83-101) with:

```python
def build_messages(
    windowed_slices: list[np.ndarray],
    instruction: str,
    query: str,
) -> tuple[list[dict], list[PIL.Image.Image]]:
    """Build the chat messages and image list for mlx-vlm inference."""
    images = [PIL.Image.fromarray(s) for s in windowed_slices]
    content = (
        [{"type": "text", "text": instruction}]
        + [
            item
            for i in range(len(windowed_slices))
            for item in (
                {"type": "image"},
                {"type": "text", "text": f"SLICE {i + 1}"},
            )
        ]
        + [{"type": "text", "text": query}]
    )
    return [{"role": "user", "content": content}], images
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_ct_utils.py -v`

Expected: All tests PASS — including the updated `TestBuildMessages` tests. `TestEncodeSliceBase64` is gone. All other test classes (`TestWindowCtSlice`, `TestSampleSlices`, `TestParseDicomFiles`, `TestSlicesToGifBytes`) still pass unchanged.

- [ ] **Step 4: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`

Expected: No errors (the unused `base64` import is gone, `encode_slice_base64` references removed).

- [ ] **Step 5: Commit**

```bash
git add ct_utils.py
git commit -m "feat: update build_messages for mlx-vlm, remove encode_slice_base64"
```

---

### Task 4: Rewrite `streamlit_app.py` for mlx-vlm

**Files:**
- Modify: `streamlit_app.py:1-152`

- [ ] **Step 1: Replace the full contents of `streamlit_app.py`**

```python
# streamlit_app.py
from __future__ import annotations

import io
import os
import platform
import sys

import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ct_utils import (  # noqa: E402
    build_messages,
    parse_dicom_files,
    sample_slices,
    slices_to_gif_bytes,
    window_ct_slice,
)


@st.cache_data(show_spinner=False)
def process_dicoms(
    file_contents: list[bytes], max_slices: int
) -> tuple[list[np.ndarray], bytes]:
    """Parse, sample, window DICOMs and build GIF. Cached on file contents."""
    files = [io.BytesIO(c) for c in file_contents]
    hu_slices = parse_dicom_files(files)
    if not hu_slices:
        return [], b""
    hu_slices = sample_slices(hu_slices, max_slices)
    windowed = [window_ct_slice(s) for s in hu_slices]
    gif = slices_to_gif_bytes(windowed)
    return windowed, gif


DEFAULT_INSTRUCTION = (
    "You are an instructor teaching medical students. You are "
    "analyzing a contiguous block of CT slices from the center of "
    "the abdomen. Please review the slices provided below carefully."
)

DEFAULT_QUERY = (
    "Based on the visual evidence in the slices provided above, "
    "is this image a good teaching example of liver pathology? "
    "Comment on hypodense lesions or other hepatic irregularities. "
    "Do not comment on findings outside the liver. Please provide "
    "your reasoning and conclude with a 'Final Answer: yes' or "
    "'Final Answer: no'."
)


@st.cache_resource
def load_model():
    """Load MedGemma model and processor once, cached across reruns."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "Apple Silicon is required. This app does not support x86 or Linux."
        )
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set. Add it to your .env file.")

    from mlx_vlm import load

    model, processor = load("mlx-community/medgemma-1.5-4b-it-bf16")
    return model, processor


def run_inference_stream(model, processor, messages, images):
    """Yield response text chunks from MedGemma via mlx-vlm streaming."""
    from mlx_vlm import stream_generate
    from mlx_vlm.prompt_utils import get_chat_template

    prompt = get_chat_template(processor, messages, add_generation_prompt=True)
    for chunk in stream_generate(
        model,
        processor,
        prompt,
        image=images,
        max_tokens=2000,
        temperature=0.0,
    ):
        yield chunk.text


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
    instruction = st.text_area(
        "System instruction", value=DEFAULT_INSTRUCTION, height=150
    )
    query = st.text_area("Clinical question", value=DEFAULT_QUERY, height=150)
    analyze_btn = st.button("Analyze", type="primary", disabled=not uploaded_files)

# --- Main area ---
if not uploaded_files:
    st.info("Upload DICOM files in the sidebar to get started.")
    st.stop()

# Parse and window slices (cached on file contents + max_slices)
with st.spinner("Reading DICOM files..."):
    file_contents = [f.getvalue() for f in uploaded_files]
    windowed, gif_bytes = process_dicoms(file_contents, max_slices)
    if not windowed:
        st.error("No valid DICOM files with pixel data found.")
        st.stop()

st.subheader(f"CT Preview ({len(windowed)} slices)")
st.image(gif_bytes, caption="Windowed CT slices (R=wide, G=soft tissue, B=brain)")

# --- Inference ---
if analyze_btn:
    st.subheader("Analysis Results")
    messages, images = build_messages(windowed, instruction, query)

    with st.spinner("Loading MedGemma model (first run may take a few minutes)..."):
        model, processor = load_model()

    output = st.empty()
    full_response = ""
    for token in run_inference_stream(model, processor, messages, images):
        full_response += token
        output.markdown(full_response)
```

- [ ] **Step 2: Run lint and format**

Run: `uv run ruff check . && uv run ruff format .`

Expected: No errors.

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `uv run pytest tests/ -v`

Expected: All tests PASS. The `streamlit_app.py` changes don't affect `ct_utils` tests.

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: rewrite inference for mlx-vlm with streaming on Apple Silicon"
```

---

### Task 5: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md:1-43`

- [ ] **Step 1: Replace the full contents of `CLAUDE.md`**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit app that analyzes CT scans using Google's MedGemma 1.5 (4B-IT) vision-language model via mlx-vlm on Apple Silicon. Users upload DICOM files, the app applies CT windowing, and MedGemma produces a streaming clinical analysis.

## Commands

```bash
uv run streamlit run streamlit_app.py                  # run app
uv run pytest tests/ -v                                # all tests
uv run pytest tests/test_ct_utils.py::TestName -v      # single test class
uv run ruff check .                                    # lint
uv run ruff format .                                   # format
uv run ty check                                        # type check
```

## Architecture

Two files: `ct_utils.py` (pure functions) and `streamlit_app.py` (UI + inference).

**`ct_utils.py`** — Stateless utility pipeline, no Streamlit dependency:
`parse_dicom_files` → `sample_slices` → `window_ct_slice` → `slices_to_gif_bytes` / `build_messages`

**`streamlit_app.py`** — Orchestrates the pipeline with two caching layers:
- `@st.cache_data` on `process_dicoms()` — caches DICOM parsing + windowing + GIF, keyed on file bytes + max_slices
- `@st.cache_resource` on `load_model()` — loads MedGemma once via mlx-vlm
- Streaming inference via `stream_generate()` with token-by-token UI updates

**CT windowing** — MedGemma expects 3-channel RGB from Hounsfield Units. Defined in `WINDOW_CLIPS`:
- Red: wide (-1024..1024), Green: soft tissue (-135..215), Blue: brain (0..80)

## Environment

- `HF_TOKEN` in `.env` — required for Hugging Face model access
- Apple Silicon (macOS ARM64) — only supported runtime
- `mlx-vlm` — MLX vision-language model inference library
- `uv sync --dev` — installs all dependencies

## Test Data

16 DICOM files in `samples/` used by `TestParseDicomFiles` integration tests.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for MLX/Apple Silicon migration"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`

Expected: All tests PASS.

- [ ] **Step 2: Run lint**

Run: `uv run ruff check .`

Expected: No errors.

- [ ] **Step 3: Run format check**

Run: `uv run ruff format --check .`

Expected: No files would be reformatted.

- [ ] **Step 4: Verify no stale imports**

Run: `grep -rn "import torch\|import transformers\|from torch\|from transformers\|import accelerate\|flash_attn\|encode_slice_base64" ct_utils.py streamlit_app.py tests/test_ct_utils.py`

Expected: No matches — all old references are gone.
