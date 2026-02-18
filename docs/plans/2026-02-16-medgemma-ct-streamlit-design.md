# MedGemma CT Streamlit App Design

## Overview

Single-page Streamlit app that lets users upload DICOM CT files, preview windowed slices, and run MedGemma 1.5 locally on Apple Silicon (MPS) to analyze the CT volume.

## Architecture

Single file: `streamlit_app.py`. No multi-page routing.

## UI Layout

### Sidebar

- **File uploader** — accepts multiple `.dcm` files (`st.file_uploader` with `accept_multiple_files=True`)
- **Max slices slider** — default 85, range 1–200 (uniform sampling)
- **System instruction** — `st.text_area`, pre-filled with notebook default
- **Clinical question** — `st.text_area`, pre-filled with notebook liver pathology query
- **"Analyze" button**

### Main Area (sequential)

1. Title + brief description
2. **CT Preview** — animated GIF of windowed CT slices via `st.image`
3. **Analysis Results** — spinner during inference, then rendered markdown response

## Data Flow

1. Upload `.dcm` files → `pydicom.dcmread` → sort by `InstanceNumber`
2. Uniform sample to max slices
3. Convert to Hounsfield Units via `apply_rescale`
4. Apply 3-channel RGB windowing (R: wide -1024–1024, G: soft tissue -135–215, B: brain 0–80)
5. Display preview GIF
6. On "Analyze": base64-encode slices → build chat message → tokenize → `model.generate` on MPS
7. Post-process and display response

## Model Loading

- `@st.cache_resource` for model + processor (persist across reruns)
- Device: MPS via `device_map="auto"`
- Dtype: `torch.float32` (MPS has limited bfloat16 support)
- Model: `google/medgemma-1.5-4b-it`

## Key Differences from Notebook

| Aspect | Notebook | Streamlit App |
|--------|----------|---------------|
| Data source | Google DICOM store | Local file upload |
| GPU | CUDA (L4) | MPS (Apple Silicon) |
| Dtype | bfloat16 | float32 (MPS compat) |
| Prompt | Hardcoded | Editable text areas |
| Auth | Google Cloud + HF | HF token from `.env` only |

## Error Handling

- Validate uploaded files are valid DICOM with pixel data
- Show warning if no files uploaded
- Catch model loading/inference errors and display via `st.error`
