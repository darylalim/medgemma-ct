# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Streamlit app that analyzes CT scans using Google's MedGemma 1.5 (4B-IT) vision-language model. Users upload DICOM files, the app applies CT windowing, and MedGemma produces a clinical analysis.

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
`parse_dicom_files` → `sample_slices` → `window_ct_slice` → `slices_to_gif_bytes` / `encode_slice_base64` → `build_messages`

**`streamlit_app.py`** — Orchestrates the pipeline with two caching layers:
- `@st.cache_data` on `process_dicoms()` — caches DICOM parsing + windowing + GIF, keyed on file bytes + max_slices
- `@st.cache_resource` on `load_model()` — loads MedGemma once with Flash Attention 2 on CUDA

**CT windowing** — MedGemma expects 3-channel RGB from Hounsfield Units. Defined in `WINDOW_CLIPS`:
- Red: wide (-1024..1024), Green: soft tissue (-135..215), Blue: brain (0..80)

## Environment

- `HF_TOKEN` in `.env` — required for Hugging Face model access
- CUDA (Linux) — only supported runtime
- `flash-attn` — Linux-only, requires `nvcc`/`CUDA_HOME` to build
- `uv sync --dev` — installs all dependencies

## Test Data

16 DICOM files in `samples/` used by `TestParseDicomFiles` integration tests.
