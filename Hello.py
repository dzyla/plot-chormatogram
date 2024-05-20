import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import numpy as np
import random
import matplotlib.colors as mcolors
import os


def generate_random_colors(num_colors):
    return [
        random.choice(list(mcolors.CSS4_COLORS.values())) for _ in range(num_colors)
    ]


# Function to plot chromatogram
def plot_chromatogram(
    ax,
    data,
    x_column,
    y_column,
    plot_every,
    title,
    sample_name,
    mod_y,
    y_scale,
    x_lim,
    ymin,
    ymax,
    do_fractions,
    column_fractions,
    move_fraction_text,
    color,
    line_width,
):
    if sample_name is None:
        sample_name = "UV 280 nm"

    ax.plot(
        data[x_column].astype(float),
        (data[y_column].astype(float) + mod_y) * y_scale,
        label=sample_name,
        color=color,
        linewidth=line_width,
    )

    # Fraction plotting
    if do_fractions:
        vline_scale = 0.1
        y_scale = max(data[y_column]) - min(data[y_column])
        vline_height = y_scale * vline_scale

        if ymax is not None:
            vline_height = ymax * vline_scale
        for n, (ml, fraction) in enumerate(
            zip(data[column_fractions].dropna(), data["Fraction"].dropna())
        ):
            ml = float(ml)
            if float(x_lim[0]) <= ml <= float(x_lim[1]):
                if pd.notna(ml) and pd.notna(fraction) and n % plot_every == 0:
                    font_dict = {"size": 8}
                    ax.text(
                        ml,
                        ymin + move_fraction_text,
                        fraction,
                        fontdict=font_dict,
                        rotation=90,
                    )
                    ax.vlines(
                        ml,
                        ymin,
                        ymin + vline_height,
                        alpha=0.4,
                        color="k",
                        linestyles="--",
                    )

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title, weight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)

    if ymax is not None:
        ax.set_ylim(ymin, ymax)

    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])

    if data[x_column].max() > 20:
        ax.xaxis.set_major_locator(
            ticker.MultipleLocator(int(data[x_column].max() / 25))
        )
        ax.set_xticklabels(np.array(ax.get_xticks()).astype(int), rotation=90)
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))


# Streamlit application starts here
st.title("Chromatogram Plotter")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV/Text file", type=["csv", "txt", "asc"])

if uploaded_file is not None:
    st.write(f"Selected file:", f"__{uploaded_file.name}__")

    # Read CSV file
    uploaded_file.seek(0)

    if uploaded_file.name.endswith(".asc"):
        data = pd.read_csv(uploaded_file, delimiter="\t", header=2)

    else:
        try:
            data = pd.read_csv(
                uploaded_file, encoding="utf-16", delimiter="\t", header=2
            )
        except UnicodeError:
            buffer = uploaded_file.getvalue().decode("utf-8")
            text_stream = io.StringIO(buffer)
            data = pd.read_csv(text_stream, header=2)
            data = data.dropna(axis=1, how="all")

    st.write("Columns in your file:", data.columns.tolist())

    # Filter numeric columns for slider limits
    numeric_cols = data.select_dtypes(include="number").columns

    # Slider for number of traces
    n_traces = len(numeric_cols) // 2
    if n_traces > 1:
        num_traces = st.slider("Number of Traces", 1, len(numeric_cols) // 2, 1)
        if_fractions = st.checkbox("# Plot fractions", True)
    else:
        num_traces = 1
        if_fractions = False

    # Initialize or retrieve the list of random colors
    if (
        "random_colors" not in st.session_state
        or len(st.session_state.random_colors) < num_traces
    ):
        st.session_state.random_colors = generate_random_colors(num_traces)

    # Main plotting parameters section
    with st.expander("Plotting Parameters", expanded=True):
        title = st.text_input("Title of Plot", os.path.basename(uploaded_file.name))
        figsize_width = st.slider("Figure Width", 5, 20, 10)
        figsize_height = st.slider("Figure Height", 3, 15, 5)
        figsize_dpi = st.selectbox("Figure DPI", [150, 200, 300, 500])
        fraction_column_x = st.selectbox("Fraction X Column (assuming Y column is called Fractions)", data.columns)

        if len(numeric_cols) > 0:
            first_numeric_col = numeric_cols[0]
            x_lim_min, x_lim_max = st.slider(
                "X-axis Limits",
                0,
                int(data[first_numeric_col].max()),
                (0, int(data[first_numeric_col].max())),
            )
            ymin, ymax = st.slider(
                "Y-axis Limits",
                int(data[numeric_cols].max().max()) * -2,
                int(data[numeric_cols].max().max()) * 2,
                (0, int(data[numeric_cols].max().max())),
            )

    # Trace settings section
    # if n_traces > 1:
    #     num_traces = st.slider("Number of Traces", 1, n_traces, 1)
    #     if_fractions = st.checkbox("# Plot fractions", True)
    # else:
    #     if_fractions = False

    trace_params = []
    for i in range(num_traces):
        with st.expander(f"Trace {i+1} Settings"):
            x_column = st.selectbox("X column", numeric_cols, key=f"x_column_trace_{i}")
            default_y_column = (
                numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            )
            y_column = st.selectbox(
                "Y column",
                numeric_cols,
                index=1 if len(numeric_cols) > 1 else 0,
                key=f"y_column_trace_{i}",
            )

            color = st.color_picker(
                "Color", value=st.session_state.random_colors[i], key=f"color_trace_{i}"
            )
            line_width = st.slider("Line Width", 1, 5, 2, key=f"line_width_trace_{i}")
            plot_every = st.slider(
                "Plot Every N Fractions", 1, 10, 1, key=f"plot_every_trace_{i}"
            )
            sample_name = st.text_input(
                "Sample Name", y_column, key=f"sample_name_trace_{i}"
            )
            mod_y = st.number_input("Modification Y", value=0, key=f"mod_y_trace_{i}")
            y_scale = st.number_input("Y Scale", value=1.0, key=f"y_scale_trace_{i}")
            trace_params.append(
                (
                    x_column,
                    y_column,
                    color,
                    plot_every,
                    sample_name,
                    mod_y,
                    y_scale,
                    line_width,
                )
            )

    # Plot button
    if st.button("Plot Chromatogram"):
        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height), dpi=figsize_dpi)
        for (
            x_column,
            y_column,
            color,
            plot_every,
            sample_name,
            mod_y,
            y_scale,
            line_width,
        ) in trace_params:
            plot_chromatogram(
                ax,
                data,
                x_column,
                y_column,
                plot_every,
                title,
                sample_name,
                mod_y,
                y_scale,
                (x_lim_min, x_lim_max),
                ymin,
                ymax,
                if_fractions,
                fraction_column_x,
                0,
                color,
                line_width,
            )
        st.pyplot(fig)

        # Download plot
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        filename = os.path.basename(uploaded_file.name).replace(".csv", "")
        st.download_button("Download Plot", buf, f"{filename}.png", "image/png")

st.write('Dawid Zyla 2024. Source code available on [GitHub](https://github.com/dzyla/plot-chormatogram/)')
