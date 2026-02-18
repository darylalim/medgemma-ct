# MedGemma CT Analysis

Analyze CT scans with Google's [MedGemma 1.5](https://huggingface.co/google/medgemma-1.5-4b-it) vision-language model. Upload DICOM files, preview windowed slices, and get AI-generated clinical analysis — all in a local Streamlit app.

> For educational purposes only — not for clinical use.

## Requirements

- Linux with CUDA GPU
- [uv](https://docs.astral.sh/uv/)
- Hugging Face account with MedGemma access
- `nvcc` / `CUDA_HOME` set (required to build `flash-attn`)

## Setup

1. Install dependencies:

    ```bash
    uv sync
    ```

2. Create a `.env` file with your Hugging Face token:

    ```bash
    HF_TOKEN=hf_...
    ```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

1. Upload `.dcm` files in the sidebar
2. Adjust max slices and prompts as needed
3. Preview the windowed CT slices (animated GIF)
4. Click **Analyze** to run MedGemma inference

## How It Works

DICOMs are parsed, sorted by instance number, and uniformly sampled to a configurable max. Each slice is windowed into 3-channel RGB:

| Channel | Window      | HU Range    |
|---------|-------------|-------------|
| Red     | Wide        | -1024..1024 |
| Green   | Soft tissue | -135..215   |
| Blue    | Brain       | 0..80       |

Windowed slices are base64-encoded and sent as a multi-image chat message. Inference runs on CUDA with Flash Attention 2.

## Development

```bash
uv run pytest tests/ -v     # run tests
uv run ruff check .         # lint
uv run ruff format .        # format
uv run ty check             # type check
```

## Project Structure

```
ct_utils.py          # CT processing utilities (windowing, sampling, DICOM parsing, encoding)
streamlit_app.py     # Streamlit UI and MedGemma inference
tests/
  test_ct_utils.py   # Unit and integration tests
samples/             # Sample DICOM files for testing
```
