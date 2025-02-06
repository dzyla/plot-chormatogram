# Chromatogram Plotter

A Streamlit application for plotting chromatogram data from CSV, TXT, or ASC files (exported from Unicorn). It allows users to upload data, customize plot appearance, and export the resulting visualizations.

## Features

*   **Data Upload:** Supports uploading one or more CSV, TXT, or ASC files.
*   **Customizable Plots:** Users can adjust various plot parameters, including:
    *   X and Y axis columns
    *   Axis limits
    *   Title
    *   Colors
    *   Line widths
    *   Legend and Grid display
    *   Fraction markers
*   **Multiple Traces:** Plot multiple chromatogram traces on the same graph.
*   **Static and Interactive Plots:** Generates both static Matplotlib plots and interactive Plotly plots.
*   **Export Options:** Allows downloading the static plot in PNG, PDF, and SVG formats and the interactive plot as an HTML file.

## Installation

It is highly recommended to create a virtual environment to isolate project dependencies. Choose one of the following methods:

**1. Using `uv` (Fastest):**

*   Ensure you have `uv` installed (`pip install uv`).
*   Create a virtual environment: `uv venv`
*   Activate the environment: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
*   Install dependencies: `uv pip install -r requirements.txt`

**2. Using `conda`:**

*   Ensure you have Conda installed.
*   Create a conda environment: `conda create -n chromatogram_plotter python=3.9` (adjust Python version if needed)
*   Activate the environment: `conda activate chromatogram_plotter`
*   Install dependencies: `conda install --file requirements.txt`

**3. Using `venv` (Standard Python):**

*   Ensure you have Python 3.6+ installed.
*   Create a virtual environment: `python -m venv chromatogram_venv`
*   Activate the environment: `source chromatogram_venv/bin/activate` (Linux/macOS) or `chromatogram_venv\Scripts\activate` (Windows)
*   Install dependencies: `pip install -r requirements.txt`

After activating your chosen environment, install the necessary Python packages using the command:

```bash
pip install -r requirements.txt
```

**Libraries**

Streamlit
* Pandas
* Matplotlib
* Plotly

**Repository**
https://github.com/dzyla/plot-chormatogram/

**Live App**
https://happy-chromatogram.streamlit.app/