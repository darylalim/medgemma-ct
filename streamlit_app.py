# streamlit_app.py
from __future__ import annotations

import io
import os

import numpy as np
import streamlit as st
import torch
import transformers
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
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required. This app does not support MPS or CPU inference."
        )
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set. Add it to your .env file.")
    model_id = "google/medgemma-1.5-4b-it"
    processor = transformers.AutoProcessor.from_pretrained(
        model_id,
        use_fast=True,
        token=hf_token,
    )
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        token=hf_token,
    ).to("cuda")
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
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=2000,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


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
    messages = build_messages(windowed, instruction, query)

    with st.spinner("Loading MedGemma model (first run may take a few minutes)..."):
        model, processor = load_model()

    with st.spinner("Running inference..."):
        response = run_inference(model, processor, messages)

    st.markdown(response)
