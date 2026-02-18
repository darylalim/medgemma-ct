---
name: streamlit-app-builder
description: >
  Generate Streamlit apps from Python scripts, Jupyter notebooks, or notebook
  URLs. Triggers: turn a notebook into a Streamlit app, build an interactive
  app, create a UI for a script, make code interactive, or any link to a
  .ipynb file.
---

# Streamlit App Builder

Generate single-page Streamlit apps that wrap existing Python code. Sources include local `.py` files, `.ipynb` notebooks in project knowledge, or notebooks fetched from URLs.

## Step 1: Analyze Source Code

Scan for Python files and Jupyter notebooks. Identify functions (with parameters and return types), imports, data patterns (DataFrames, file I/O, model loading), and visualizations.

### Fetching Notebooks from URLs

When the user provides a notebook link:

1. **Resolve the raw URL:**
   - GitHub: `github.com/.../blob/.../*.ipynb` → `raw.githubusercontent.com/.../*.ipynb`
   - Colab: `colab.research.google.com/drive/<id>` → export URL
   - GitLab: append `?ref=main&format=json` or use raw endpoint
   - Other: use URL directly if it serves `.ipynb` JSON

2. **Download and extract code cells:**
   ```bash
   curl -L -o notebook.ipynb "<resolved_raw_url>"
   ```
   ```python
   import json

   with open("notebook.ipynb") as f:
       nb = json.load(f)

   code_cells = [
       "".join(cell["source"])
       for cell in nb["cells"]
       if cell["cell_type"] == "code"
   ]
   ```

3. Proceed with analysis on the extracted code.

## Step 2: Classify and Select Pattern

| Pattern | Indicators | UI Approach |
|---------|-----------|-------------|
| ML Model | `sklearn`, `torch`, `keras`, `.predict()`, model files | Prediction interface |
| Data Processing | `pandas`, file I/O, DataFrame transforms | File upload → transform → download |
| Visualization | `matplotlib`, `plotly`, `seaborn`, `.plot()` | Interactive chart controls |
| General Script | Functions with parameters | Form inputs → output display |

## Step 3: Generate App

Follow these principles:
- Use Streamlit defaults — no custom CSS
- Single column unless data naturally splits
- Inputs at top → action button → results below
- Preserve original function logic — wrap, don't rewrite
- Include type annotations on function signatures and key variables
- Add brief inline comments explaining UI choices
- Load configuration from a `.env` file when present using `python-dotenv`

### Environment Configuration

Generated apps should load settings (API keys, file paths, model paths, thresholds, etc.) from a `.env` file when one exists. Add this near the top of every generated app:

```python
from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(".env")
if env_path.exists():
    load_dotenv(env_path)

# Example: read config with sensible defaults
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
API_KEY = os.getenv("API_KEY")
```

When generating an app, also create a `.env.example` file documenting all configurable values with placeholder defaults:

```
# .env.example
MODEL_PATH=model.pkl
API_KEY=your-api-key-here
```

### ML Model Template

```python
import streamlit as st
import joblib  # or appropriate loader
import pandas as pd

st.title("Model Prediction")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Map features to widgets: numeric → number_input, categorical → selectbox,
# boolean → checkbox, text → text_input
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.selectbox("Feature 2", ["A", "B", "C"])

if st.button("Predict"):
    with st.spinner("Running prediction..."):
        input_data = pd.DataFrame([[feature_1, feature_2]], columns=["f1", "f2"])
        prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
```

Widget heuristics: int → `number_input(step=1)`, float → `number_input(step=0.1)`, low cardinality (<10) → `selectbox()`, binary → `checkbox()`, free text → `text_input()` / `text_area()`.

### Data Processing Template

```python
import streamlit as st
import pandas as pd

st.title("Data Processor")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Input Data")
    st.dataframe(df.head())

    if st.button("Process"):
        with st.spinner("Processing..."):
            result = df  # [Insert transformation logic]

        st.subheader("Result")
        st.dataframe(result)
        csv = result.to_csv(index=False)
        st.download_button("Download CSV", csv, "result.csv", "text/csv")
```

Parameter heuristics: column names → `selectbox(df.columns)`, thresholds → `slider()` / `number_input()`, groupby → `multiselect()`, dates → `date_input()`.

### Visualization Template

```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Interactive Visualization")

uploaded_file = st.file_uploader("Upload data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    x_col = st.selectbox("X axis", df.columns)
    y_col = st.selectbox("Y axis", df.columns)
    color_col = st.selectbox("Color by", [None] + list(df.columns))

    fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
    st.plotly_chart(fig, use_container_width=True)
```

Prefer Plotly for interactivity. For matplotlib: use `st.pyplot(fig)`. Chart heuristics: scatter → x, y, color, size; line → x, y, group; bar → category, value; histogram → column, bins; heatmap → x, y, value.

### General Script Template

```python
import streamlit as st

st.title("Script Name")

param1 = st.text_input("Parameter 1")
param2 = st.number_input("Parameter 2", value=10)

if st.button("Run"):
    with st.spinner("Running..."):
        result = main_function(param1, param2)

    if isinstance(result, pd.DataFrame):
        st.dataframe(result)
    elif isinstance(result, (dict, list)):
        st.json(result)
    else:
        st.write(result)
```

### Component Quick Reference

| Need | Component |
|------|-----------|
| Text input | `st.text_input()`, `st.text_area()` |
| Numbers | `st.number_input()`, `st.slider()` |
| Selection | `st.selectbox()`, `st.multiselect()`, `st.radio()` |
| Boolean | `st.checkbox()`, `st.toggle()` |
| Date/time | `st.date_input()`, `st.time_input()` |
| File input | `st.file_uploader()` |
| Display data | `st.dataframe()`, `st.table()`, `st.json()` |
| Display text | `st.write()`, `st.markdown()`, `st.code()` |
| Charts | `st.plotly_chart()`, `st.pyplot()`, `st.altair_chart()` |
| Feedback | `st.success()`, `st.error()`, `st.warning()`, `st.info()` |
| Progress | `st.spinner()`, `st.progress()` |
| Download | `st.download_button()` |
| Layout | `st.columns()`, `st.expander()`, `st.sidebar` |

## Step 4: Code Quality

Run ruff and ty (both from Astral) to lint, format, and type check. Use uv for project management.

```bash
pip install uv --break-system-packages

uv init --name streamlit-app
uv add streamlit python-dotenv  # plus dependencies identified in Step 1
uv add --dev ruff ty pytest

uv run ruff check --fix streamlit_app.py
uv run ruff format streamlit_app.py
uv run ty check streamlit_app.py
```

Fix all lint and type errors before delivering.

## Step 5: Testing

Create `test_streamlit_app.py` with pytest unit tests for all functions that can be tested independently of the Streamlit UI (data transforms, validation, utilities, prediction wrappers).

```python
import pytest
from streamlit_app import function_a, function_b

def test_function_a_basic():
    result = function_a(input_value)
    assert result == expected_value

def test_function_a_edge_case():
    with pytest.raises(ValueError):
        function_a(bad_input)
```

Guidelines: at least one test per function, cover happy path + edge cases + errors, use `pytest.fixture` for shared setup, keep tests independent.

```bash
uv run pytest test_streamlit_app.py -v
```

Fix all failures before delivering.

## Output Checklist

- [ ] `streamlit_app.py` — single-file app with type annotations and inline comments
- [ ] `test_streamlit_app.py` — pytest unit tests for all testable functions
- [ ] `pyproject.toml` — uv-managed project with all dependencies
- [ ] `.env.example` — documents all configurable environment variables with defaults
- [ ] ruff check, ruff format, and ty check pass clean
- [ ] All pytest tests pass
