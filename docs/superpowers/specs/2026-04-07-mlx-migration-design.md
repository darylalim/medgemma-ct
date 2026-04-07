# MLX Migration Design

Migrate the MedGemma CT analysis app from CUDA/transformers to MLX/mlx-vlm, enabling native inference on Apple Silicon.

## Decisions

- **Full replacement** — remove torch/transformers/CUDA entirely, MLX-only
- **Model:** `mlx-community/medgemma-1.5-4b-it-bf16` (full precision, ~10GB)
- **Streaming output** — use `stream_generate()` for token-by-token display
- **Minimal swap (Approach 1)** — keep 2-file architecture, smallest diff
- **Clean break on `build_messages`** — new return signature, remove `encode_slice_base64`

## Changes by File

### `ct_utils.py`

**Remove `encode_slice_base64`** — mlx-vlm accepts PIL Images directly; base64 encoding is no longer needed.

**Replace `build_messages` signature:**

```python
# Before
def build_messages(windowed_slices, instruction, query) -> list[dict]

# After
def build_messages(windowed_slices, instruction, query) -> tuple[list[dict], list[PIL.Image.Image]]
```

- Messages use `{"type": "image"}` placeholders (no embedded data URI)
- Returns `(messages, images)` where `images` is a list of PIL Images from the numpy arrays
- Empty slices: returns messages with just instruction + query text, and an empty image list

All other functions unchanged: `parse_dicom_files`, `sample_slices`, `window_ct_slice`, `slices_to_gif_bytes`.

### `streamlit_app.py`

**Imports:** Replace `torch` and `transformers` with:
- `from mlx_vlm import load, stream_generate`
- `from mlx_vlm.prompt_utils import get_chat_template`

Remove `encode_slice_base64` from `ct_utils` imports (deleted).

**`load_model()`:** Replace CUDA guard + transformers loading:
- Guard: check `platform.machine() == "arm64"` and `sys.platform == "darwin"` (raise `RuntimeError` if not Apple Silicon)
- Load: `mlx_vlm.load("mlx-community/medgemma-1.5-4b-it-bf16")`
- Returns `(model, processor)`, still `@st.cache_resource`
- Still reads `HF_TOKEN` from environment

**`run_inference()` replaced with streaming approach:**
- Format prompt: `get_chat_template(processor, messages, add_generation_prompt=True)`
- Generate: `stream_generate(model, processor, prompt, image=images, max_tokens=2000, temperature=0.0)`
- Each chunk yields a `GenerationResult` with `.text`

**UI inference section:**
- Unpack `(messages, images)` from `build_messages`
- Stream tokens into the UI using `st.empty()` with iterative `markdown()` calls — accumulate text and re-render on each chunk (Streamlit's `write_stream` expects a simple string generator, but we need to extract `.text` from `GenerationResult` objects)

### `pyproject.toml`

**Remove:**
- `accelerate`, `flash-attn`, `torch`, `transformers` from dependencies
- `[tool.uv.sources]` torch index block
- `[[tool.uv.index]]` pytorch-cu124 block
- `no-build-isolation-package` setting

**Add:**
- `mlx-vlm` to dependencies

**Update `required-environments`:**
- Remove Linux/x86_64 entry
- Keep only `sys_platform == 'darwin' and platform_machine == 'arm64'`

### Tests (`tests/test_ct_utils.py`)

**`TestBuildMessages`** — update all 4 tests:
- Unpack `(messages, images)` from `build_messages`
- Assert image placeholders are `{"type": "image"}` (no `"image"` data key)
- Assert `images` list contains PIL Images of correct count
- Empty slices: `images == []`

**`TestEncodeSliceBase64`** — delete entirely (function removed).

**Remove `base64` import and `encode_slice_base64` from ct_utils import.**

All other test classes unchanged: `TestWindowCtSlice`, `TestSampleSlices`, `TestParseDicomFiles`, `TestSlicesToGifBytes`.

### `CLAUDE.md`

- Overview: MLX/Apple Silicon instead of CUDA/Linux
- Architecture: mlx-vlm + streaming instead of Flash Attention 2 + CUDA
- `ct_utils.py` pipeline: remove `encode_slice_base64` from function chain
- Environment: remove CUDA, `flash-attn`, `nvcc`/`CUDA_HOME`; add Apple Silicon + `mlx-vlm`

## Out of Scope

- Dual backend (CUDA + MLX) support
- New files or project restructuring
- Model selection UI (hardcoded to bf16 variant)
- Changes to DICOM parsing, windowing, or GIF generation
