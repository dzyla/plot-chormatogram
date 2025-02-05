import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import numpy as np
import random
import matplotlib.colors as mcolors
import os

# Function to generate a random color.
def generate_random_color():
    return random.choice(list(mcolors.CSS4_COLORS.values()))

# Function to plot the chromatogram with modified X and Y.
def plot_chromatogram(
    ax,
    data,
    x_column,
    y_column,
    plot_every,
    title,
    sample_name,
    mod_x,
    x_scale,
    mod_y,
    y_scale,
    x_lim,
    ymin,
    ymax,
    do_fractions,
    fraction_column,
    move_fraction_text,
    color,
    line_width,
    show_legend,
    show_grid
):
    # Adjust x and y signals based on user modifications.
    x_data = (data[x_column].astype(float) + mod_x) * x_scale
    y_data = (data[y_column].astype(float) + mod_y) * y_scale

    if sample_name is None or sample_name.strip() == "":
        sample_name = "UV 280 nm"

    # Plot the trace.
    ax.plot(x_data, y_data, label=sample_name, color=color, linewidth=line_width)

    # Plot fractions if enabled.
    if do_fractions:
        vline_scale = 0.1
        y_scale_val = max(data[y_column]) - min(data[y_column])
        vline_height = y_scale_val * vline_scale
        if ymax is not None:
            vline_height = ymax * vline_scale
        # Iterate over fraction data.
        for n, (ml, fraction) in enumerate(zip(data[fraction_column].dropna(), data["Fraction"].dropna())):
            try:
                ml_val = float(ml)
            except ValueError:
                continue
            if float(x_lim[0]) <= ml_val <= float(x_lim[1]):
                if pd.notna(ml) and pd.notna(fraction) and n % plot_every == 0:
                    font_dict = {"size": 8}
                    ax.text(ml_val, ymin + move_fraction_text, fraction, fontdict=font_dict, rotation=90)
                    ax.vlines(ml_val, ymin, ymin + vline_height, alpha=0.4, color="k", linestyles="--")

    # Set labels and title.
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title, weight="bold")
    if show_legend:
        ax.legend()
    if show_grid:
        ax.grid(True, alpha=0.2)

    # Set axis limits.
    if ymax is not None:
        ax.set_ylim(ymin, ymax)
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    # Adjust tick spacing.
    if data[x_column].max() > 20:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(int(data[x_column].max() / 25)))
        ax.set_xticklabels(np.array(ax.get_xticks()).astype(int), rotation=90)
    else:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

# Application title.
st.title("Chromatogram Plotter")

# File uploader: allow multiple files.
uploaded_files = st.file_uploader(
    "Choose one or more CSV/Text files", 
    type=["csv", "txt", "asc"], 
    accept_multiple_files=True,
    help="Upload one or more files (CSV, TXT, or ASC) containing your chromatogram data."
)

# Parse the uploaded files.
data_dict = {}
if uploaded_files is not None and len(uploaded_files) > 0:
    st.write("Uploaded Files:")
    for file in uploaded_files:
        file_name = os.path.basename(file.name)
        file_key = os.path.splitext(file_name)[0]  # Remove extension for display.
        st.write(f"__{file_key}__")
        file.seek(0)
        try:
            if file.name.endswith(".asc"):
                df = pd.read_csv(file, delimiter="\t", header=2)
            else:
                try:
                    df = pd.read_csv(file, encoding="utf-16", delimiter="\t", header=2)
                except UnicodeError:
                    buffer = file.getvalue().decode("utf-8")
                    text_stream = io.StringIO(buffer)
                    df = pd.read_csv(text_stream, header=2)
                    df = df.dropna(axis=1, how="all")
            data_dict[file_key] = df
        except Exception as e:
            st.error(f"Error reading file {file_name}: {e}")

# Continue only if at least one file is successfully read.
if data_dict:
    # Choose the number of traces outside the form.
    num_traces = st.number_input(
        "Number of Traces", 
        min_value=1, 
        max_value=10, 
        value=1, 
        step=1,
        help="Set the number of chromatogram traces you wish to plot."
    )

    # Ensure that colors for each trace are stored in session state.
    for i in range(int(num_traces)):
        key = f"color_trace_{i}"
        if key not in st.session_state:
            st.session_state[key] = generate_random_color()

    # Create a form to group settings.
    with st.form("plot_settings_form"):
        st.subheader("Global Plotting Parameters")
        with st.expander("Global Plotting Parameters", expanded=False):
            # Global parameters come from the first uploaded file by default.
            first_key = list(data_dict.keys())[0]
            first_df = data_dict[first_key]

            default_title = first_key
            title = st.text_input(
                "Title of Plot", 
                default_title, 
                help="Enter a title for the overall plot."
            )
            # Use columns to narrow width for figure size parameters.
            col1, col2, col3 = st.columns(3)
            figsize_width = col1.slider(
                "Figure Width", 5, 20, 10, 
                help="Adjust the width of the figure (in inches)."
            )
            figsize_height = col2.slider(
                "Figure Height", 3, 15, 5, 
                help="Adjust the height of the figure (in inches)."
            )
            figsize_dpi = col3.selectbox(
                "Figure DPI", [150, 200, 300, 500], 
                help="Select the resolution (dots per inch) for the figure."
            )

            # For fraction plotting.
            fraction_column = st.selectbox(
                "Fraction X Column (for fraction markers)", 
                first_df.columns.tolist(),
                help="Select the column that contains the X-values for fraction markers."
            )
            do_fractions = st.checkbox(
                "Plot Fractions", 
                value=True,
                help="Toggle this to overlay fraction markers on the plot."
            )
            move_fraction_text = st.number_input(
                "Move Fraction Text (Y-offset)", 
                value=0,
                help="Adjust the vertical offset for fraction text annotations."
            )

            # Global slider for x- and y-axis limits.
            numeric_cols_global = first_df.select_dtypes(include="number").columns
            if len(numeric_cols_global) > 0:
                first_numeric = numeric_cols_global[0]
                x_lim_min, x_lim_max = st.slider(
                    "X-axis Limits",
                    float(first_df[first_numeric].min()),
                    float(first_df[first_numeric].max()),
                    (0.0, float(first_df[first_numeric].max())),
                    help="Set the minimum and maximum values for the X-axis."
                )
                global_ymax = float(first_df[numeric_cols_global].max().max())
                ymin, ymax = st.slider(
                    "Y-axis Limits",
                    -2 * global_ymax,
                    2 * global_ymax,
                    (0.0, global_ymax),
                    help="Set the minimum and maximum values for the Y-axis."
                )
            else:
                x_lim_min, x_lim_max = 0.0, 100.0
                ymin, ymax = 0.0, 100.0

            # Use columns for the two additional plot options.
            col_legend, col_grid = st.columns(2)
            show_legend = col_legend.checkbox(
                "Show Legend", 
                value=True,
                help="Toggle this to add or remove the legend from the plot."
            )
            show_grid = col_grid.checkbox(
                "Show Grid", 
                value=True,
                help="Toggle this to include or remove the grid from the plot."
            )

        st.markdown("---")
        st.subheader("Trace Settings")
        # Prepare a list to collect trace parameters.
        trace_params = []

        # For each trace, use an expander for trace settings.
        for i in range(int(num_traces)):
            with st.expander(f"Trace {i+1} Settings", expanded=False):
                # Select data file for the trace.
                file_key = st.selectbox(
                    "Data File", 
                    list(data_dict.keys()), 
                    key=f"data_file_trace_{i}",
                    help="Select which data file to use for this trace."
                )
                df = data_dict[file_key]
                # Show numeric columns available in that file.
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                if not numeric_cols:
                    st.error("No numeric columns found in the file. Please choose a different file.")
                    continue
                x_column = st.selectbox(
                    "X Column", 
                    numeric_cols, 
                    key=f"x_column_trace_{i}",
                    help="Select the column to be used for the X-axis. Modification and scaling will be applied."
                )
                y_column = st.selectbox(
                    "Y Column", 
                    numeric_cols, 
                    index=1 if len(numeric_cols) > 1 else 0, 
                    key=f"y_column_trace_{i}",
                    help="Select the column to be used for the Y-axis. Modification and scaling will be applied."
                )

                # Inputs for modifying the x and y signals.
                mod_x = st.number_input(
                    "Modification X", 
                    value=0.0, 
                    key=f"mod_x_trace_{i}",
                    help="Value added to the X-axis signal. This shifts the X-axis baseline."
                )
                x_scale = st.number_input(
                    "X Scale", 
                    value=1.0, 
                    key=f"x_scale_trace_{i}",
                    help="Factor to multiply the X-axis signal. Use to scale the X-axis values."
                )
                mod_y = st.number_input(
                    "Modification Y", 
                    value=0.0, 
                    key=f"mod_y_trace_{i}",
                    help="Value added to the Y-axis signal. This shifts the Y-axis baseline."
                )
                y_scale = st.number_input(
                    "Y Scale", 
                    value=1.0, 
                    key=f"y_scale_trace_{i}",
                    help="Factor to multiply the Y-axis signal. Use to scale the Y-axis values."
                )

                # Additional trace parameters.
                color = st.color_picker(
                    "Color", 
                    value=st.session_state[f"color_trace_{i}"], 
                    key=f"color_trace_{i}_picker",
                    help="Select a color for the trace."
                )
                st.session_state[f"color_trace_{i}"] = color

                line_width = st.slider(
                    "Line Width", 
                    1, 5, 2, 
                    key=f"line_width_trace_{i}",
                    help="Set the width of the trace line."
                )
                plot_every = st.slider(
                    "Plot Every N Fractions", 
                    1, 10, 1, 
                    key=f"plot_every_trace_{i}",
                    help="Choose how often to annotate fractions (every Nth fraction)."
                )
                sample_name = st.text_input(
                    "Sample Name", 
                    file_key, 
                    key=f"sample_name_trace_{i}",
                    help="Enter a sample name for the trace. This appears in the legend."
                )

                # Append dictionary with all settings for this trace.
                trace_params.append({
                    "file_key": file_key,
                    "df": df,
                    "x_column": x_column,
                    "y_column": y_column,
                    "mod_x": mod_x,
                    "x_scale": x_scale,
                    "mod_y": mod_y,
                    "y_scale": y_scale,
                    "color": color,
                    "line_width": line_width,
                    "plot_every": plot_every,
                    "sample_name": sample_name,
                })

        # Form submission button.
        submitted = st.form_submit_button("Submit Changes")
    # End of form

    # Plot the chromatogram only when the form is submitted.
    if submitted:
        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height), dpi=figsize_dpi)
        for params in trace_params:
            df = params["df"]
            try:
                plot_chromatogram(
                    ax,
                    df,
                    params["x_column"],
                    params["y_column"],
                    params["plot_every"],
                    title,
                    params["sample_name"],
                    params["mod_x"],
                    params["x_scale"],
                    params["mod_y"],
                    params["y_scale"],
                    (x_lim_min, x_lim_max),
                    ymin,
                    ymax,
                    do_fractions,
                    fraction_column,
                    move_fraction_text,
                    params["color"],
                    params["line_width"],
                    show_legend,
                    show_grid
                )
            except KeyError as e:
                st.warning(f"Missing column in file {params['file_key']}: {e}")
        st.pyplot(fig)

        # Provide an export section.
        with st.expander("Export Plot", expanded=True):
            col1, col2, col3 = st.columns(3)
            formats = {
                "png": "image/png",
                "pdf": "application/pdf",
                "svg": "image/svg+xml"
            }
            filename = title if title else first_key
            for col, (format_type, mime_type) in zip([col1, col2, col3], formats.items()):
                with col:
                    buf = io.BytesIO()
                    try:
                        fig.savefig(buf, format=format_type, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            f"Download {format_type.upper()}",
                            data=buf,
                            file_name=f"{filename}.{format_type}",
                            mime=mime_type,
                            use_container_width=True,
                            help=f"Download the plot in {format_type.upper()} format."
                        )
                    except Exception as e:
                        st.error(f"Error saving {format_type}: {e}")

    st.write(
        'Dawid Zyla 2025. Source code available on [GitHub](https://github.com/dzyla/plot-chormatogram/)'
    )
