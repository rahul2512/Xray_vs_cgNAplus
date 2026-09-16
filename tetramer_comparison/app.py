"""
Tetramer Structural Comparison — Shiny for Python Application
=============================================================
Compares sequence-resolved DNA structural coordinates across
X-ray crystallography, molecular dynamics (MD) simulation,
and optionally coarse-grained DNA (cgDNA) datasets.

The three built-in example datasets (data/example_xray.csv,
data/example_md.csv, data/example_cgdna.csv) are loaded
automatically on startup so the app is immediately usable when
deployed.  Users can:
  • Replace any built-in dataset by uploading their own CSV in the
    corresponding slot.
  • Add a fourth custom dataset (any name) via the "Custom dataset"
    upload slot.

Entry point:  shiny run --reload app.py
"""

from __future__ import annotations

import os
import io
from itertools import product as iproduct
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget

from utils.metrics import (
    CLASS_COLOURS,
    STEP_CLASS_LABELS,
    calculate_all_coords_stats,
    calculate_cosine,
    calculate_heatmap_stats,
    calculate_pearson,
    calculate_rmse,
    calculate_summary_statistics,
    check_dataset_compatibility,
    classify_central_step,
    detect_coord_type,
    get_coord_type_label,
    linear_regression,
    load_dataset,
    prepare_comparison,
    validate_dataset,
)

# ---------------------------------------------------------------------------
# Built-in default datasets — loaded once at module import time.
# Paths are relative to this file so they work regardless of cwd.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "data")

def _load_builtin(filename: str, label: str):
    """Load one of the bundled example CSVs.  Returns (df, label) or (None, label)."""
    path = os.path.join(_DATA_DIR, filename)
    if not os.path.exists(path):
        return None, label
    df_raw, err = load_dataset(path)
    if err:
        return None, label
    df, warns = validate_dataset(df_raw, label)
    return df, label   # df may be None if validation failed, that's fine

BUILTIN_DATASETS = {
    "xray":  _load_builtin("example_xray.csv",  "X-ray"),
    "md":    _load_builtin("example_md.csv",     "MD"),
    "cgdna": _load_builtin("example_cgdna.csv",  "cgDNA"),
}
# BUILTIN_DATASETS[key] = (df_or_None, label_str)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-dataset marker symbols and colours for the profile plot
DATASET_MARKERS = {
    "xray":   dict(symbol="square",          color="#1a2744", size=6),
    "md":     dict(symbol="diamond",         color="#e63946", size=7),
    "cgdna":  dict(symbol="circle",          color="#2a9d8f", size=7),
    "custom": dict(symbol="triangle-up",     color="#9b59b6", size=7),
}

DEFAULT_FIG_W  = 760
DEFAULT_FIG_H  = 540

PLOT_LAYOUT_BASE = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    # NOTE: `font` is intentionally omitted here so each call can pass it
    # as an explicit keyword without a duplicate-key conflict.
    margin=dict(l=72, r=40, t=58, b=70),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        showgrid=True, gridcolor="#e8e8e8",
        linecolor="#aaaaaa", linewidth=1,
        ticks="outside", ticklen=4, zeroline=False,
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#e8e8e8",
        linecolor="#aaaaaa", linewidth=1,
        ticks="outside", ticklen=4, zeroline=False,
    ),
)

# Base font dict – family + colour stay constant; size is overridden per call
_FONT_BASE = dict(family="Arial, Helvetica, sans-serif", color="#222222")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = ui.HTML("""
<style>
/* ── Global ── */
body { font-family: 'Arial', sans-serif; background: #f0f2f7; margin:0; }

/* ── Header ── */
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(135deg, #0f1f3d 0%, #1a3a6b 60%, #1e4d8c 100%);
  color: white;
  padding: 0 28px;
  height: 56px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.app-header-left  { display:flex; align-items:center; gap:12px; }
.app-header-icon  {
  font-size: 1.5rem; line-height:1;
  background: rgba(255,255,255,0.12);
  border-radius: 8px;
  padding: 5px 8px;
}
.app-header-title { font-size:1.1rem; font-weight:700; letter-spacing:0.02em; }
.app-header-sub   { font-size:0.72rem; color:#90b4e0; margin-top:1px; }
.app-header-right { display:flex; align-items:center; gap:10px; }
.app-header-pill  {
  background: rgba(255,255,255,0.13);
  border: 1px solid rgba(255,255,255,0.22);
  color: #c5d9f0;
  font-size: 0.68rem;
  font-weight:600;
  padding: 3px 10px;
  border-radius: 20px;
  letter-spacing: 0.04em;
}
.app-header-author {
  font-size:0.72rem; color:#90b4e0; font-style:italic;
}

/* ── Control bar ── */
.ctrl-bar {
  background: white;
  border: 1px solid #dde2ec;
  border-radius: 8px;
  padding: 8px 10px 6px 10px;
  margin-bottom: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.ctrl-group-label {
  font-size: 0.63rem; font-weight:700; color:#9ca3af;
  text-transform: uppercase; letter-spacing:0.08em;
  margin-bottom: 3px; white-space: nowrap;
}
/* Compact all inputs inside the ctrl-bar */
.ctrl-bar .form-group  { margin-bottom: 3px !important; }
.ctrl-bar label        { font-size: 0.76rem !important; color:#374151;
                          margin-bottom: 1px !important; white-space:nowrap; overflow:hidden;
                          text-overflow:ellipsis; }
.ctrl-bar .form-control{ font-size:0.8rem !important; padding:2px 5px !important;
                          height:26px !important; }
.ctrl-bar input[type=number]  { height:26px !important; padding:2px 5px !important; }
.ctrl-bar input[type=text]    { height:26px !important; padding:2px 5px !important;
                                  font-size:0.8rem !important; }
.ctrl-bar .checkbox label { font-size:0.76rem !important; }
.ctrl-bar .bslib-col      { padding-left:4px !important; padding-right:4px !important; }
/* Export buttons */
.ctrl-bar .btn { font-size:0.76rem !important; padding:3px 7px !important; }

/* ── KPI cards ── */
.kpi-card {
  background:white; border-radius:7px; border:1px solid #dde2ec;
  padding:8px 14px 6px 14px; text-align:center;
  box-shadow:0 1px 3px rgba(0,0,0,0.06); min-width:105px;
}
.kpi-label { font-size:0.67rem; color:#6b7280; text-transform:uppercase;
             letter-spacing:0.06em; margin-bottom:1px; }
.kpi-value { font-size:1.45rem; font-weight:700; color:#1a2744; line-height:1.1; }
.kpi-sub   { font-size:0.65rem; color:#9ca3af; margin-top:1px; }

/* ── Sidebar section titles ── */
.sb-title {
  font-size:0.7rem; font-weight:700; color:#6b7280;
  text-transform:uppercase; letter-spacing:0.07em;
  margin:12px 0 4px 0; border-top:1px solid #e5e7eb; padding-top:10px;
}
.sb-title:first-child { border-top:none; margin-top:0; padding-top:0; }

/* ── Message boxes ── */
.info-box {
  background:#f0f4ff; border-left:3px solid #3b5bdb;
  padding:6px 9px; font-size:0.76rem; color:#374151;
  border-radius:0 4px 4px 0; margin-top:5px; margin-bottom:5px;
}
.warn-box {
  background:#fff7ed; border-left:3px solid #f59e0b;
  padding:6px 9px; font-size:0.76rem; color:#374151;
  border-radius:0 4px 4px 0; margin-top:5px;
}
.err-box {
  background:#fef2f2; border-left:3px solid #ef4444;
  padding:6px 9px; font-size:0.76rem; color:#374151;
  border-radius:0 4px 4px 0; margin-top:5px;
}
.ok-box {
  background:#f0fdf4; border-left:3px solid #22c55e;
  padding:6px 9px; font-size:0.76rem; color:#374151;
  border-radius:0 4px 4px 0; margin-top:3px;
}
.stat-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px; }

/* ── Custom upload block ── */
.custom-upload-block {
  background:#fafafa; border:1px dashed #c7cfe0;
  border-radius:6px; padding:7px 9px 5px 9px; margin-top:4px;
}
.custom-upload-block .sb-title { border-top:none; margin-top:0; padding-top:0; }

/* ── Export buttons ── */
.export-btn-col .btn { width:100%; margin-bottom:3px; font-size:0.78rem; padding:4px 8px; }

/* ── Description / About page ── */
.desc-page { max-width:840px; margin:0 auto; padding:10px 6px 50px 6px; }
.desc-page h3 { color:#1a2744; margin-top:1.6rem; margin-bottom:0.35rem; font-size:1.05rem; }
.desc-page h4 { color:#374151; margin-top:1.1rem; margin-bottom:0.25rem; font-size:0.95rem; }
.desc-page p, .desc-page li { font-size:0.9rem; color:#374151; line-height:1.65; }
.desc-page code { background:#f1f5f9; padding:1px 5px; border-radius:3px;
                   font-size:0.85rem; color:#1a2744; }
.desc-page table { border-collapse:collapse; width:100%; margin:8px 0; font-size:0.87rem; }
.desc-page th { background:#1a2744; color:white; padding:6px 10px; text-align:left; }
.desc-page td { padding:5px 10px; border-bottom:1px solid #e5e7eb; }
.desc-page tr:nth-child(even) td { background:#f8fafc; }
.credit-footer { margin-top:2.5rem; padding-top:0.9rem; border-top:1px solid #e5e7eb;
                 font-size:0.8rem; color:#9ca3af; text-align:right; }
</style>
""")

# ---------------------------------------------------------------------------
# Helper: KPI card HTML
# ---------------------------------------------------------------------------

def _kpi(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# UI definition
# ---------------------------------------------------------------------------

app_ui = ui.page_fluid(
    CUSTOM_CSS,

    # ── Compact professional header ──
    ui.div(
        {"class": "app-header"},
        ui.div(
            {"class": "app-header-left"},
            ui.HTML('<div class="app-header-icon">🧬</div>'),
            ui.div(
                ui.HTML('<div class="app-header-title">Tetramer Structural Comparison</div>'),
                ui.HTML('<div class="app-header-sub">Sequence-resolved DNA structural coordinates — X-ray · MD · cgDNA</div>'),
            ),
        ),
        ui.div(
            {"class": "app-header-right"},
            ui.HTML('<span class="app-header-pill">256 tetramers · 18 coordinates</span>'),
            ui.HTML('<span class="app-header-author">Rahul Sharma</span>'),
        ),
    ),

    # ============================================================
    # TOP-LEVEL NAV — Description tab sits outside the sidebar layout
    # ============================================================
    ui.navset_tab(

        # ---- Description tab (full width, no sidebars) ----
        ui.nav_panel(
            "About",
            ui.div(
                {"class": "desc-page"},
                ui.HTML("""
<h3>Purpose</h3>
<p>This application compares <strong>sequence-resolved DNA structural coordinates</strong>
across data sources: X-ray crystallography, molecular dynamics (MD) simulation, and
coarse-grained DNA (cgDNA) models. Structural parameters — such as Twist, Roll, Propeller,
or Rise — depend on the local nucleotide sequence context. The finest practically useful
resolution is the DNA <strong>tetramer</strong>: the 4-base subsequence centred on the
dinucleotide step of interest. All 4<sup>4</sup> = 256 possible tetramers are compared
simultaneously.</p>

<h3>Input file format</h3>
<p>Each dataset must be a plain <code>.csv</code> file with the following structure:</p>
<table>
  <tr><th>Tetramer</th><th>Coord_1</th><th>Coord_2</th><th>…</th><th>Coord_18</th></tr>
  <tr><td>AAAA</td><td>0.123</td><td>35.4</td><td>…</td><td>3.38</td></tr>
  <tr><td>AAAC</td><td>0.087</td><td>34.8</td><td>…</td><td>3.35</td></tr>
  <tr><td>…</td><td>…</td><td>…</td><td>…</td><td>…</td></tr>
  <tr><td>TTTT</td><td>−0.456</td><td>36.1</td><td>…</td><td>3.41</td></tr>
</table>
<ul>
  <li>The <strong>first column must be named <code>Tetramer</code></strong>.</li>
  <li>Each tetramer must be exactly 4 characters from <code>A</code>, <code>C</code>, <code>G</code>, <code>T</code>.</li>
  <li>The recommended number of rows is 256 (all possible 4-mers); fewer rows are accepted with a warning.</li>
  <li>All remaining columns must be numeric. Column names are detected automatically — no specific naming convention is required.</li>
  <li>Coordinate names containing keywords such as <em>buckle</em>, <em>propeller</em>, <em>twist</em>, <em>roll</em> etc. are automatically labelled Intra or Inter in the UI.</li>
</ul>

<h3>Central-step classification</h3>
<p>The <strong>central dinucleotide</strong> of a tetramer is positions 2 and 3 (1-indexed).
For example, in <code>ACGT</code> the central dinucleotide is <code>CG</code>.
Each base is classified as purine (A or G) or pyrimidine (C or T), giving four step classes:</p>
<table>
  <tr><th>Class</th><th>Meaning</th><th>Example central steps</th></tr>
  <tr><td>PP</td><td>Purine / Purine</td><td>AA, AG, GA, GG</td></tr>
  <tr><td>PY</td><td>Purine / Pyrimidine</td><td>AC, AT, GC, GT</td></tr>
  <tr><td>YP</td><td>Pyrimidine / Purine</td><td>CA, CG, TA, TG</td></tr>
  <tr><td>YY</td><td>Pyrimidine / Pyrimidine</td><td>CC, CT, TC, TT</td></tr>
</table>
<p>Each class contains exactly 64 tetramers. Use the <em>Central-step filter</em> in the
right panel to restrict any plot to one class or one exact dinucleotide.</p>

<h3>Statistical metrics</h3>
<h4>Pearson correlation (r)</h4>
<p>Measures <strong>linear association</strong> after mean-centering both vectors.
It is insensitive to additive offsets or multiplicative scaling, making it
appropriate for comparing structural trends across datasets with systematic biases.</p>

<h4>Cosine similarity</h4>
<p>Measures the <strong>angle</strong> between the two raw (un-centred) vectors:
<code>cos θ = (x · y) / (‖x‖ ‖y‖)</code>.
Unlike Pearson r, cosine similarity is sensitive to the mean level of both vectors.
A high cosine with a low Pearson r indicates good shape agreement but a
mean-level difference (systematic bias). <em>These metrics are not interchangeable.</em></p>

<h3>Application tabs</h3>
<table>
  <tr><th>Tab</th><th>Description</th></tr>
  <tr><td>Scatter comparison</td><td>X vs Y scatter for one coordinate; KPI cards; regression; data table</td></tr>
  <tr><td>Coordinate overview</td><td>Pearson r bar chart across all coordinates; sortable table</td></tr>
  <tr><td>Similarity analysis</td><td>Pearson r vs cosine scatter — one point per coordinate</td></tr>
  <tr><td>Correlation heatmap</td><td>18 coords × step classes (PP/PY/YP/YY/All), switchable metric</td></tr>
  <tr><td>Difference plot</td><td>Y − X bar chart per tetramer, sortable</td></tr>
  <tr><td>Coordinate profile</td><td>Raw coordinate values per tetramer, all datasets overlaid</td></tr>
</table>

<h3>Datasets</h3>
<p>Three <strong>built-in example datasets</strong> are loaded automatically on startup
(from <code>data/example_xray.csv</code>, <code>data/example_md.csv</code>,
<code>data/example_cgdna.csv</code>). To use your own data:</p>
<ul>
  <li><strong>Replace a built-in</strong> — upload a CSV in the corresponding slot in the
      left panel. Your file overrides the built-in for that session.</li>
  <li><strong>Add a custom dataset</strong> — use the <em>Custom dataset</em> block in the
      left panel to name and upload a fourth CSV that appears alongside the three built-ins.</li>
  <li><strong>Permanent replacement</strong> — to ship a deployed version with your own
      defaults, simply replace the three CSVs in the <code>data/</code> folder before
      deploying. No code change required.</li>
</ul>

<h3>Export</h3>
<p>Download buttons in the right panel export the current scatter plot as
PNG (2× scale), PDF, or SVG, and the filtered comparison table as CSV.
PNG and SVG export requires <a href="https://github.com/plotly/Kaleido" target="_blank">Kaleido</a>.
PDF requires Kaleido ≥ 1.0 with Chrome; if unavailable the download completes
with an informative placeholder.</p>

<div class="credit-footer">
  Created by <strong>Rahul Sharma</strong>
</div>
"""),
            ),
        ),

        # ---- Main analysis tab (left sidebar + control bar + plots) ----
        ui.nav_panel(
            "Analysis",

            ui.layout_sidebar(
                # ============================================================
                # LEFT SIDEBAR — datasets only
                # ============================================================
                ui.sidebar(
                    ui.HTML('<div class="sb-title">Datasets</div>'),
                    ui.HTML(
                        '<div class="info-box" style="margin-bottom:6px;">'
                        'Built-in datasets load automatically. Upload a CSV to '
                        '<b>replace</b> any built-in, or add a fourth custom dataset below.'
                        '</div>'
                    ),
                    ui.input_file("file_xray",  "Replace X-ray (.csv)",
                                  accept=[".csv"], multiple=False),
                    ui.input_file("file_md",    "Replace MD (.csv)",
                                  accept=[".csv"], multiple=False),
                    ui.input_file("file_cgdna", "Replace cgDNA (.csv)",
                                  accept=[".csv"], multiple=False),
                    ui.div(
                        {"class": "custom-upload-block"},
                        ui.HTML('<div class="sb-title">Custom dataset (optional)</div>'),
                        ui.input_text("custom_label", "Name", value="Custom",
                                      placeholder="e.g. My simulation"),
                        ui.input_file("file_custom", "Upload CSV",
                                      accept=[".csv"], multiple=False),
                    ),
                    ui.output_ui("dataset_status_ui"),
                    width=250,
                    open="open",
                    bg="#f9fafc",
                ),

                # ============================================================
                # MAIN AREA — compact single-row control bar + tabbed plots
                # ============================================================
                ui.div(
                    {"class": "ctrl-bar"},
                    ui.layout_columns(
                        # Axes + coordinate
                        ui.div(
                            ui.HTML('<div class="ctrl-group-label">Axes &amp; Coordinate</div>'),
                            ui.output_ui("dataset_x_ui"),
                            ui.output_ui("dataset_y_ui"),
                            ui.output_ui("coord_select_ui"),
                        ),
                        # Filters
                        ui.div(
                            ui.HTML('<div class="ctrl-group-label">Filters</div>'),
                            ui.input_select(
                                "step_class_filter", "Step class",
                                choices={"All":"All","PP":"PP","PY":"PY","YP":"YP","YY":"YY"},
                                selected="All",
                            ),
                            ui.output_ui("step_dinuc_ui"),
                        ),
                        # Options
                        ui.div(
                            ui.HTML('<div class="ctrl-group-label">Options</div>'),
                            ui.input_checkbox("show_identity",   "y = x line", value=True),
                            ui.input_checkbox("show_regression", "Regression", value=True),
                        ),
                        # Figure
                        ui.div(
                            ui.HTML('<div class="ctrl-group-label">Figure</div>'),
                            ui.input_numeric("fig_width",    "W",          DEFAULT_FIG_W, min=400, max=2000, step=50),
                            ui.input_numeric("fig_height",   "H",          DEFAULT_FIG_H, min=300, max=1600, step=50),
                            ui.input_numeric("fig_fontsize", "Axis font",  13, min=8, max=24, step=1),
                            ui.input_numeric("fig_xfont",    "Tick/legend", 9, min=5, max=20, step=1),
                        ),
                        # Filename + Export
                        ui.div(
                            ui.HTML('<div class="ctrl-group-label">Filename &amp; Export</div>'),
                            ui.input_text("dl_filename", None, value="tetramer_plot",
                                          placeholder="filename (no ext.)"),
                            ui.div(
                                {"style": "display:flex; gap:4px; margin-top:5px; flex-wrap:nowrap;"},
                                ui.download_button("dl_png", "PNG"),
                                ui.download_button("dl_pdf", "PDF"),
                                ui.download_button("dl_svg", "SVG"),
                                ui.download_button("dl_csv", "CSV"),
                            ),
                        ),
                        col_widths=[4, 2, 2, 2, 2],
                    ),
                ),


                ui.navset_tab(

                    # ---- Tab 1: Scatter ----
                    ui.nav_panel(
                        "Scatter comparison",
                        ui.output_ui("kpi_row_ui"),
                        ui.tags.br(),
                        output_widget("scatter_plot"),
                        ui.output_ui("diff_stats_ui"),
                        ui.tags.br(),
                        ui.tags.h5("Comparison table",
                                   style="margin:6px 0 4px 0; font-size:0.93rem; color:#374151;"),
                        ui.output_data_frame("comparison_table"),
                    ),

                    # ---- Tab 2: Overview ----
                    ui.nav_panel(
                        "Coordinate overview",
                        ui.HTML('<div class="info-box">Pearson r and cosine across all '
                                'coordinates for the selected dataset pair.</div>'),
                        ui.tags.br(),
                        output_widget("overview_bar_plot"),
                        ui.tags.br(),
                        ui.output_data_frame("overview_table"),
                    ),

                    # ---- Tab 3: Similarity ----
                    ui.nav_panel(
                        "Similarity analysis",
                        ui.HTML(
                            '<div class="info-box">'
                            "<b>Pearson r</b> measures linear association after centering each vector. "
                            "<b>Cosine similarity</b> measures the angle between the raw (un-centred) "
                            "vectors — it is sensitive to the mean level. These metrics are "
                            "<em>not</em> interchangeable: high cosine with low Pearson r indicates "
                            "good shape agreement but a systematic mean-level offset."
                            "</div>"
                        ),
                        ui.tags.br(),
                        output_widget("similarity_scatter_plot"),
                    ),

                    # ---- Tab 4: Heatmap ----
                    ui.nav_panel(
                        "Correlation heatmap",
                        ui.input_radio_buttons(
                            "heatmap_metric", "Metric",
                            choices={"pearson": "Pearson r",
                                     "cosine":  "Cosine similarity"},
                            selected="pearson",
                            inline=True,
                        ),
                        output_widget("heatmap_plot"),
                    ),

                    # ---- Tab 5: Difference ----
                    ui.nav_panel(
                        "Difference plot",
                        ui.layout_columns(
                            ui.input_select(
                                "diff_sort", "Sort tetramers by",
                                choices={"alpha": "Alphabetical",
                                         "abs_diff": "Absolute difference"},
                                selected="abs_diff",
                            ),
                            ui.input_checkbox("diff_hline", "Show zero line", value=True),
                            col_widths=[6, 6],
                        ),
                        output_widget("difference_plot"),
                    ),

                    # ---- Tab 6: Coordinate profile ----
                    ui.nav_panel(
                        "Coordinate profile",
                        ui.layout_columns(
                            ui.input_select(
                                "profile_sort", "Sort tetramers by",
                                choices={
                                    "alpha":    "Alphabetical",
                                    "xray":     "X-ray value",
                                    "md":       "MD value",
                                    "cgdna":    "cgDNA value",
                                    "stepclass":"Step class",
                                },
                                selected="alpha",
                            ),
                            ui.output_ui("profile_dataset_selector_ui"),
                            col_widths=[4, 8],
                        ),
                        output_widget("profile_plot"),
                    ),
                ),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input: Inputs, output: Outputs, session: Session):

    # Per-session input accessors (closures over the server's `input` object)
    def _input_val(input_id: str, default=None):
        """Read an input value safely; returns default if not yet initialised."""
        try:
            return input[input_id]()
        except Exception:
            return default

    def _input_exists(input_id: str) -> bool:
        try:
            input[input_id]()
            return True
        except Exception:
            return False

    # ================================================================
    # Dataset parsing (one per upload slot)
    # ================================================================

    def _parse_slot(file_input, name: str):
        fi = file_input()
        if fi is None:
            return None, [], []
        try:
            path = fi[0]["datapath"]
            df_raw, load_err = load_dataset(path)
        except Exception as exc:
            return None, [f"{name}: unexpected error: {exc}"], []
        if load_err:
            return None, [f"{name}: {load_err}"], []
        df, warns = validate_dataset(df_raw, name)
        if df is None:
            return None, warns, []
        return df, [], warns

    @reactive.calc
    def _xray():   return _parse_slot(input.file_xray,   "X-ray (uploaded)")

    @reactive.calc
    def _md():     return _parse_slot(input.file_md,     "MD (uploaded)")

    @reactive.calc
    def _cgdna():  return _parse_slot(input.file_cgdna,  "cgDNA (uploaded)")

    @reactive.calc
    def _custom():
        lbl = (_input_val("custom_label", "Custom") or "Custom").strip() or "Custom"
        return _parse_slot(input.file_custom, lbl)

    @reactive.calc
    def available_datasets():
        """
        Build the active dataset registry.

        Priority for the three built-in slots:
          1. User-uploaded file (overrides built-in)
          2. Built-in default from data/

        A fourth 'custom' slot is added if a file has been uploaded there.
        """
        avail = {}

        for key, builtin_label in [("xray", "X-ray"), ("md", "MD"), ("cgdna", "cgDNA")]:
            slot_func = {"xray": _xray, "md": _md, "cgdna": _cgdna}[key]
            uploaded_df, errs, _ = slot_func()
            if uploaded_df is not None and not errs:
                # User uploaded a replacement — use it with the slot's default label
                avail[key] = (builtin_label, uploaded_df)
            else:
                # Fall back to built-in
                builtin_df, builtin_lbl = BUILTIN_DATASETS.get(key, (None, builtin_label))
                if builtin_df is not None:
                    avail[key] = (builtin_lbl, builtin_df)

        # Custom (4th) dataset
        custom_df, custom_errs, _ = _custom()
        if custom_df is not None and not custom_errs:
            lbl = (_input_val("custom_label", "Custom") or "Custom").strip() or "Custom"
            avail["custom"] = (lbl, custom_df)

        return avail

    # ================================================================
    # Dynamic sidebar UI
    # ================================================================

    @output
    @render.ui
    def dataset_status_ui():
        msgs = []

        # Report upload status for the three replaceable slots
        for key, label, slot in [
            ("xray",  "X-ray", _xray),
            ("md",    "MD",    _md),
            ("cgdna", "cgDNA", _cgdna),
        ]:
            df, errs, warns = slot()
            for e in errs:
                msgs.append(ui.HTML(f'<div class="err-box">&#10006; {e}</div>'))
            for w in warns:
                msgs.append(ui.HTML(f'<div class="warn-box">&#9888; {w}</div>'))
            if df is not None and not errs:
                msgs.append(ui.HTML(
                    f'<div class="ok-box">&#10003; {label} replaced: '
                    f'{len(df)} tetramers, {len(df.columns)} coords</div>'
                ))

        # Report custom dataset
        custom_df, custom_errs, custom_warns = _custom()
        for e in custom_errs:
            msgs.append(ui.HTML(f'<div class="err-box">&#10006; {e}</div>'))
        for w in custom_warns:
            msgs.append(ui.HTML(f'<div class="warn-box">&#9888; {w}</div>'))
        if custom_df is not None and not custom_errs:
            lbl = (_input_val("custom_label", "Custom") or "Custom").strip() or "Custom"
            msgs.append(ui.HTML(
                f'<div class="ok-box">&#10003; Custom "{lbl}": '
                f'{len(custom_df)} tetramers, {len(custom_df.columns)} coords</div>'
            ))

        # Summary of what is actually active (built-ins + overrides + custom)
        avail = available_datasets()
        active_labels = [v[0] for v in avail.values()]
        if avail:
            msgs.append(ui.HTML(
                f'<div class="info-box" style="margin-top:8px;">'
                f'<b>Active datasets:</b> {", ".join(active_labels)}</div>'
            ))
        else:
            msgs.append(ui.HTML(
                '<div class="warn-box">No datasets loaded. '
                'Built-in CSVs missing from <code>data/</code>.</div>'
            ))
        return ui.tags.div(*msgs)

    @output
    @render.ui
    def dataset_x_ui():
        avail = available_datasets()
        choices = {k: v[0] for k, v in avail.items()} or {"xray": "X-ray (not loaded)"}
        return ui.input_select("ds_x", "X-axis dataset", choices=choices,
                               selected=list(choices.keys())[0])

    @output
    @render.ui
    def dataset_y_ui():
        avail = available_datasets()
        choices = {k: v[0] for k, v in avail.items()} or {"md": "MD (not loaded)"}
        keys = list(choices.keys())
        default = keys[1] if len(keys) > 1 else keys[0]
        return ui.input_select("ds_y", "Y-axis dataset", choices=choices, selected=default)

    @output
    @render.ui
    def coord_select_ui():
        avail = available_datasets()
        if not avail:
            return ui.input_select("coord", "Coordinate",
                                   choices={"": "(upload data first)"}, selected="")
        df_first = list(avail.values())[0][1]
        choices = {c: f"{c} {get_coord_type_label(c)}".strip() for c in df_first.columns}
        return ui.input_select("coord", "Coordinate", choices=choices,
                               selected=list(choices.keys())[0])

    @output
    @render.ui
    def step_dinuc_ui():
        cls = input.step_class_filter() if _input_exists("step_class_filter") else "All"
        all_dinucs = ["".join(x) for x in iproduct("ACGT", repeat=2)]
        if cls != "All":
            all_dinucs = [d for d in all_dinucs
                          if classify_central_step("A" + d + "A")[1] == cls]
        choices = {"All": "All"} | {d: d for d in all_dinucs}
        return ui.input_select("step_dinuc_filter", "Central dinucleotide",
                               choices=choices, selected="All")

    # ================================================================
    # Core reactive: get the two selected dfs and their labels
    # ================================================================

    @reactive.calc
    def selected_pair():
        """Return (label_x, df_x, label_y, df_y) or (error_str, None, None, None)."""
        avail = available_datasets()
        if len(avail) < 2:
            return "Need at least two valid datasets.", None, None, None

        keys = list(avail.keys())
        ds_x = _input_val("ds_x", keys[0])
        ds_y = _input_val("ds_y", keys[min(1, len(keys)-1)])

        # Prevent identical axes
        if ds_x == ds_y:
            ds_y = keys[(keys.index(ds_x) + 1) % len(keys)]

        if ds_x not in avail or ds_y not in avail:
            return "Selected dataset not loaded.", None, None, None

        lx, dfx = avail[ds_x]
        ly, dfy = avail[ds_y]
        return None, (lx, dfx), (ly, dfy), None

    # ================================================================
    # Filtered comparison dataframe for the active coordinate
    # ================================================================

    @reactive.calc
    def comparison_df():
        err, pair_x, pair_y, _ = selected_pair()
        if err:
            return None, err

        lx, dfx = pair_x
        ly, dfy = pair_y

        coord = _input_val("coord", None)
        if not coord:
            shared, _, _ = check_dataset_compatibility(dfx, dfy)
            if not shared:
                return None, "No shared coordinate columns."
            coord = shared[0]

        if coord not in dfx.columns or coord not in dfy.columns:
            return None, f"Coordinate '{coord}' not in both datasets."

        try:
            df = prepare_comparison(dfx, dfy, coord, lx, ly)
        except Exception as exc:
            return None, f"Comparison error: {exc}"

        cls = _input_val("step_class_filter", "All")
        if cls != "All":
            df = df[df["Step_class"] == cls].copy()

        dinuc = _input_val("step_dinuc_filter", "All")
        if dinuc != "All":
            df = df[df["Central_step"] == dinuc].copy()

        if df.empty:
            return None, "No tetramers match the current filter."

        return df, None

    # ================================================================
    # KPI row
    # ================================================================

    @output
    @render.ui
    def kpi_row_ui():
        df, err = comparison_df()
        if err:
            return ui.HTML(f'<div class="warn-box">{err}</div>')

        _, pair_x, pair_y, _ = selected_pair()
        lx = pair_x[0]; ly = pair_y[0]
        coord = _input_val("coord", "")
        type_lbl = detect_coord_type(coord)

        x = df[lx].values; y = df[ly].values
        r, p = calculate_pearson(x, y)
        cos = calculate_cosine(x, y)

        def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "—"
        p_str = f"p = {p:.2e}" if not np.isnan(p) else ""

        return ui.HTML(
            '<div class="stat-row">'
            + _kpi("Pearson r", fmt(r), p_str)
            + _kpi("Cosine similarity", fmt(cos))
            + _kpi("N tetramers", str(len(df)))
            + _kpi("Coordinate", coord, type_lbl if type_lbl != "Unknown" else "")
            + '</div>'
        )

    # ================================================================
    # Scatter plot  +  reactive cache for image export
    # ================================================================

    # Temp file path used to persist the last rendered scatter figure for downloads.
    # We write the figure JSON to disk in scatter_plot(); download handlers read it back.
    import tempfile, json as _json
    _scatter_tmp = os.path.join(tempfile.gettempdir(),
                                f"tetramer_scatter_{id(session)}.json")

    @reactive.calc
    def _fig_dims():
        w   = _input_val("fig_width",    DEFAULT_FIG_W) or DEFAULT_FIG_W
        h   = _input_val("fig_height",   DEFAULT_FIG_H) or DEFAULT_FIG_H
        fs  = _input_val("fig_fontsize", 13) or 13
        xfs = _input_val("fig_xfont",   9)  or 9
        return int(w), int(h), int(fs), int(xfs)

    def _empty_fig(msg: str):
        w, h, fs, xfs = _fig_dims()
        fig = go.Figure()
        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w, height=h,
            title=dict(text=msg, x=0.5),
        )
        return fig

    def _build_scatter():
        df, err = comparison_df()
        w, h, fs, xfs = _fig_dims()

        if err or df is None:
            return _empty_fig(err or "No data")

        _, pair_x, pair_y, _ = selected_pair()
        lx = pair_x[0]; ly = pair_y[0]
        coord    = _input_val("coord", "")
        type_lbl = get_coord_type_label(coord)
        coord_disp = f"{coord} {type_lbl}".strip()

        show_id  = _input_val("show_identity",  True)
        show_reg = _input_val("show_regression", True)
        cls_filt = _input_val("step_class_filter", "All")

        fig = go.Figure()
        classes_to_plot = (
            [cls_filt] if cls_filt != "All"
            else [c for c in ["PP","PY","YP","YY"] if c in df["Step_class"].values]
        )

        for cls in ["PP","PY","YP","YY"]:
            if cls not in classes_to_plot:
                continue
            sub = df[df["Step_class"] == cls]
            hover = [
                f"<b>Tetramer:</b> {row['Tetramer']}<br>"
                f"<b>Central step:</b> {row['Central_step']} ({row['Step_class']})<br>"
                f"<b>{lx}:</b> {row[lx]:.4f}<br>"
                f"<b>{ly}:</b> {row[ly]:.4f}<br>"
                f"<b>Diff (Y−X):</b> {row['Difference']:.4f}"
                for _, row in sub.iterrows()
            ]
            fig.add_trace(go.Scatter(
                x=sub[lx], y=sub[ly],
                mode="markers",
                name=STEP_CLASS_LABELS.get(cls, cls),
                marker=dict(color=CLASS_COLOURS[cls], size=7, opacity=0.82,
                            line=dict(width=0.5, color="white")),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            ))

        # Identity line
        if show_id:
            all_v = pd.concat([df[lx], df[ly]])
            lo, hi = float(all_v.min()), float(all_v.max())
            pad = (hi - lo) * 0.05
            fig.add_shape(type="line",
                          x0=lo-pad, y0=lo-pad, x1=hi+pad, y1=hi+pad,
                          line=dict(color="#888888", dash="dash", width=1.2),
                          layer="below")

        # Regression
        reg_str = ""
        if show_reg:
            reg = linear_regression(df[lx].values, df[ly].values)
            if reg:
                fig.add_trace(go.Scatter(
                    x=reg["x_line"], y=reg["y_line"],
                    mode="lines",
                    name=f"Regression (R²={reg['r_squared']:.3f})",
                    line=dict(color="#e63946", width=1.8),
                    hoverinfo="skip",
                ))
                sgn = "+" if reg["intercept"] >= 0 else "−"
                reg_str = (
                    f"y = {reg['slope']:.3f}x {sgn} {abs(reg['intercept']):.3f}  "
                    f"R²={reg['r_squared']:.3f}"
                )

        # Annotation
        r, p = calculate_pearson(df[lx].values, df[ly].values)
        cos   = calculate_cosine(df[lx].values, df[ly].values)
        ann   = f"r = {r:.4f}  |  cos = {cos:.4f}"
        if reg_str:
            ann += f"<br>{reg_str}"
        fig.add_annotation(
            x=1.0, y=0.01, xref="paper", yref="paper",
            xanchor="right", yanchor="bottom",
            text=ann, showarrow=False,
            font=dict(size=10, color="#555555"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc", borderwidth=1, borderpad=4,
        )

        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w, height=h,
            title=dict(text=f"{lx} vs {ly}  ·  {coord_disp}",
                       x=0.5, font=dict(size=fs+1)),
            xaxis_title=f"{lx} — {coord_disp}",
            yaxis_title=f"{ly} — {coord_disp}",
        )
        fig.update_layout(legend_font_size=xfs)
        return fig

    @render_widget
    def scatter_plot():
        fig = _build_scatter()
        # Persist figure JSON for the download handler (runs outside reactive context)
        try:
            import plotly.io as _pio
            with open(_scatter_tmp, "w") as _f:
                _f.write(_pio.to_json(fig))
        except Exception:
            pass
        return fig

    # ================================================================
    # Difference statistics card
    # ================================================================

    @output
    @render.ui
    def diff_stats_ui():
        df, err = comparison_df()
        if err or df is None:
            return ui.HTML("")
        _, pair_x, pair_y, _ = selected_pair()
        if pair_x is None:
            return ui.HTML("")
        lx = pair_x[0]; ly = pair_y[0]
        stats = calculate_summary_statistics(df[lx].values, df[ly].values)
        def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "—"
        return ui.HTML(
            "<b style='font-size:0.82rem;color:#6b7280;'>Difference statistics (Y − X)</b>"
            "<br><br>"
            '<div class="stat-row">'
            + _kpi("Mean |diff|",   fmt(stats["mean_abs_diff"]))
            + _kpi("Median |diff|", fmt(stats["median_abs_diff"]))
            + _kpi("RMSE",          fmt(stats["rmse"]))
            + _kpi("Max |diff|",    fmt(stats["max_abs_diff"]))
            + '</div>'
        )

    # ================================================================
    # Data table
    # ================================================================

    @output
    @render.data_frame
    def comparison_table():
        df, err = comparison_df()
        if err or df is None:
            return render.DataTable(pd.DataFrame({"Message": [err or "No data"]}))
        _, pair_x, pair_y, _ = selected_pair()
        if pair_x is None:
            return render.DataTable(pd.DataFrame())
        lx = pair_x[0]; ly = pair_y[0]
        cols = ["Tetramer","Central_step","Step_class", lx, ly, "Difference","Abs_difference"]
        cols = [c for c in cols if c in df.columns]
        tbl = df[cols].copy()
        for c in [lx, ly, "Difference", "Abs_difference"]:
            if c in tbl.columns:
                tbl[c] = tbl[c].round(5)
        return render.DataTable(tbl, selection_mode="none", filters=True)

    # ================================================================
    # Overview stats (shared reactive)
    # ================================================================

    @reactive.calc
    def overview_stats():
        err, pair_x, pair_y, _ = selected_pair()
        if err:
            return None, err
        lx, dfx = pair_x
        ly, dfy = pair_y
        shared, _, _ = check_dataset_compatibility(dfx, dfy)
        if not shared:
            return None, "No shared coordinate columns."

        # Apply filters to shared index
        shared_idx = dfx.index.intersection(dfy.index)
        cls = _input_val("step_class_filter", "All")
        dinuc = _input_val("step_dinuc_filter", "All")
        if cls != "All":
            shared_idx = shared_idx[[classify_central_step(t)[1] == cls for t in shared_idx]]
        if dinuc != "All":
            shared_idx = shared_idx[[classify_central_step(t)[0] == dinuc for t in shared_idx]]
        if len(shared_idx) == 0:
            return None, "No tetramers match the filter."

        stats = calculate_all_coords_stats(dfx.loc[shared_idx], dfy.loc[shared_idx], shared)
        return stats, None

    # ================================================================
    # Tab 2: Coordinate overview
    # ================================================================

    @render_widget
    def overview_bar_plot():
        stats, err = overview_stats()
        w, h, fs, xfs = _fig_dims()
        if err or stats is None:
            return _empty_fig(err or "No data")

        _, pair_x, pair_y, _ = selected_pair()
        lx = pair_x[0] if pair_x else "X"
        ly = pair_y[0] if pair_y else "Y"

        hover = [
            f"<b>{row['Coordinate']}</b><br>Pearson r: {row['Pearson_r']:.4f}<br>"
            f"Cosine: {row['Cosine']:.4f}<br>N: {row['N']}"
            for _, row in stats.iterrows()
        ]
        bar_colors = [
            "#3b5bdb" if detect_coord_type(c) == "Intra"
            else "#e63946" if detect_coord_type(c) == "Inter"
            else "#555555"
            for c in stats["Coordinate"]
        ]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stats["Coordinate"], y=stats["Pearson_r"],
            marker_color=bar_colors,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))
        fig.add_hline(y=1.0, line_dash="dot", line_color="#999999", line_width=1)
        fig.add_hline(y=0.0, line_dash="dot", line_color="#cccccc", line_width=1)
        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w, height=int(h * 0.85),
            title=dict(text=f"Pearson r — {lx} vs {ly}", x=0.5, font=dict(size=fs+1)),
            xaxis_title="Coordinate",
            yaxis_title="Pearson r",
            showlegend=False,
        )
        # Extend the y-axis range without spreading a duplicate yaxis key
        fig.update_yaxes(range=[-0.1, 1.05])
        fig.update_xaxes(tickfont=dict(size=xfs))
        return fig

    @output
    @render.data_frame
    def overview_table():
        stats, err = overview_stats()
        if err or stats is None:
            return render.DataTable(pd.DataFrame({"Message": [err or "No data"]}))
        tbl = stats.copy()
        tbl["Pearson_r"] = tbl["Pearson_r"].round(4)
        tbl["Cosine"]    = tbl["Cosine"].round(4)
        return render.DataTable(tbl, selection_mode="none", filters=False)

    # ================================================================
    # Tab 3: Similarity scatter
    # ================================================================

    @render_widget
    def similarity_scatter_plot():
        stats, err = overview_stats()
        w, h, fs, xfs = _fig_dims()
        if err or stats is None:
            return _empty_fig(err or "No data")

        _, pair_x, pair_y, _ = selected_pair()
        lx = pair_x[0] if pair_x else "X"
        ly = pair_y[0] if pair_y else "Y"

        def _col(name):
            t = detect_coord_type(name)
            return "#3b5bdb" if t == "Intra" else "#e63946" if t == "Inter" else "#555555"

        hover = [
            f"<b>{row['Coordinate']}</b><br>Pearson r: {row['Pearson_r']:.4f}<br>"
            f"Cosine: {row['Cosine']:.4f}"
            for _, row in stats.iterrows()
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stats["Pearson_r"],
            y=stats["Cosine"],
            mode="markers+text",
            text=stats["Coordinate"],
            textposition="top center",
            textfont=dict(size=9, color="#555555"),
            marker=dict(
                color=[_col(c) for c in stats["Coordinate"]],
                size=10, line=dict(width=1, color="white"),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))
        fig.add_annotation(
            x=0.01, y=0.99, xref="paper", yref="paper",
            xanchor="left", yanchor="top",
            text=(
                "<span style='color:#3b5bdb'>&#9632;</span> Intra &nbsp;"
                "<span style='color:#e63946'>&#9632;</span> Inter &nbsp;"
                "<span style='color:#555555'>&#9632;</span> Other"
            ),
            showarrow=False, font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc", borderwidth=1, borderpad=4,
        )
        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w, height=h,
            title=dict(
                text=f"Pearson r vs Cosine similarity — {lx} vs {ly}",
                x=0.5, font=dict(size=fs+1),
            ),
            xaxis_title="Pearson r",
            yaxis_title="Cosine similarity",
            showlegend=False,
        )
        fig.update_xaxes(tickfont=dict(size=xfs))
        return fig

    # ================================================================
    # Tab 4: Correlation heatmap
    # ================================================================

    @render_widget
    def heatmap_plot():
        w, h, fs, xfs = _fig_dims()
        err, pair_x, pair_y, _ = selected_pair()
        if err:
            return _empty_fig(err)
        lx, dfx = pair_x
        ly, dfy = pair_y
        shared, _, _ = check_dataset_compatibility(dfx, dfy)
        if not shared:
            return _empty_fig("No shared coordinate columns.")

        metric = _input_val("heatmap_metric", "pearson")
        heat = calculate_heatmap_stats(dfx, dfy, shared, metric=metric)
        cols = ["PP","PY","YP","YY","All"]
        heat = heat[cols]

        metric_label = "Pearson r" if metric == "pearson" else "Cosine similarity"
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=cols, y=shared,
            colorscale="RdBu",
            zmid=0.0 if metric == "pearson" else 0.5,
            zmin=-1 if metric == "pearson" else 0,
            zmax=1,
            colorbar=dict(title=metric_label, thickness=14, len=0.9),
            hovertemplate="Coord: %{y}<br>Class: %{x}<br>%{z:.4f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Arial, Helvetica, sans-serif", size=fs, color="#222222"),
            margin=dict(l=120, r=50, t=60, b=50),
            width=w,
            height=max(h, len(shared) * 30 + 120),
            title=dict(
                text=f"{metric_label} — {lx} vs {ly}",
                x=0.5, font=dict(size=fs+1),
            ),
            xaxis=dict(side="top"),
        )
        fig.update_xaxes(tickfont=dict(size=xfs))
        return fig

    # ================================================================
    # Tab 5: Difference bar plot
    # ================================================================

    @render_widget
    def difference_plot():
        df, err = comparison_df()
        w, h, fs, xfs = _fig_dims()
        if err or df is None:
            return _empty_fig(err or "No data")

        _, pair_x, pair_y, _ = selected_pair()
        if pair_x is None:
            return _empty_fig("No data")
        lx = pair_x[0]; ly = pair_y[0]
        coord    = _input_val("coord", "")
        sort_by  = _input_val("diff_sort",  "abs_diff")
        show_zl  = _input_val("diff_hline", True)

        data = df.copy()
        if sort_by == "abs_diff":
            data = data.sort_values("Abs_difference", ascending=False)
        else:
            data = data.sort_values("Tetramer")

        fig = go.Figure()
        cls_filt = _input_val("step_class_filter", "All")
        classes  = [cls_filt] if cls_filt != "All" else ["PP","PY","YP","YY"]

        for cls in classes:
            sub = data[data["Step_class"] == cls]
            if sub.empty:
                continue
            hover = [
                f"<b>{row['Tetramer']}</b> ({row['Central_step']}, {row['Step_class']})<br>"
                f"Diff (Y−X): {row['Difference']:.4f}<br>"
                f"{lx}: {row[lx]:.4f}   {ly}: {row[ly]:.4f}"
                for _, row in sub.iterrows()
            ]
            fig.add_trace(go.Bar(
                x=sub["Tetramer"], y=sub["Difference"],
                name=STEP_CLASS_LABELS.get(cls, cls),
                marker_color=CLASS_COLOURS[cls],
                opacity=0.85,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            ))

        if show_zl:
            fig.add_hline(y=0, line_dash="dot", line_color="#777777", line_width=1.2)

        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w, height=h,
            title=dict(
                text=f"Difference (Y−X) — {ly} minus {lx}  ·  {coord}",
                x=0.5, font=dict(size=fs+1),
            ),
            xaxis_title="Tetramer",
            yaxis_title=f"Difference ({ly} − {lx})",
            barmode="relative",
        )
        # Extend the x-axis tick styling without spreading a duplicate xaxis key.
        # Scale font inversely with number of tetramers so labels stay readable.
        n_bars = len(data)
        auto_tick_fs = max(5, min(11, int(round(400 / max(n_bars, 1)))))
        tick_fs = max(auto_tick_fs, xfs)
        fig.update_xaxes(tickangle=90, tickfont=dict(size=tick_fs))
        return fig

    # ================================================================
    # Tab 6: Coordinate profile
    # ================================================================

    @output
    @render.ui
    def profile_dataset_selector_ui():
        avail = available_datasets()
        choices = {k: v[0] for k, v in avail.items()} or {"xray": "X-ray (not loaded)"}
        # Pre-tick all loaded datasets
        return ui.input_checkbox_group(
            "profile_datasets",
            "Datasets to show",
            choices=choices,
            selected=list(choices.keys()),
            inline=True,
        )

    @render_widget
    def profile_plot():
        avail = available_datasets()
        w, h, fs, xfs = _fig_dims()
        coord = _input_val("coord", None)

        if not avail:
            return _empty_fig("Upload at least one dataset.")
        if not coord:
            return _empty_fig("Select a coordinate in the sidebar.")

        # Which datasets the user wants overlaid
        selected_ds = _input_val("profile_datasets", list(avail.keys()))
        if not selected_ds:
            return _empty_fig("Select at least one dataset to display.")
        # selected_ds may be a single string if only one is ticked
        if isinstance(selected_ds, str):
            selected_ds = [selected_ds]

        # Build a merged frame: Tetramer as index, one column per dataset
        frames = {}
        for key in selected_ds:
            if key not in avail:
                continue
            lbl, df = avail[key]
            if coord not in df.columns:
                continue
            frames[key] = df[coord].rename(key)

        if not frames:
            return _empty_fig(f"Coordinate '{coord}' not found in any selected dataset.")

        merged = pd.concat(frames.values(), axis=1).dropna(how="all")
        merged.index.name = "Tetramer"
        merged = merged.reset_index()

        # Add classification columns for colouring / sorting
        merged["Central_step"] = merged["Tetramer"].apply(
            lambda t: classify_central_step(t)[0])
        merged["Step_class"] = merged["Tetramer"].apply(
            lambda t: classify_central_step(t)[1])

        # Apply step-class / dinucleotide filters (same as other tabs)
        cls_filt = _input_val("step_class_filter", "All")
        if cls_filt != "All":
            merged = merged[merged["Step_class"] == cls_filt].copy()
        dinuc_filt = _input_val("step_dinuc_filter", "All")
        if dinuc_filt != "All":
            merged = merged[merged["Central_step"] == dinuc_filt].copy()

        if merged.empty:
            return _empty_fig("No tetramers match the current filter.")

        # Sort order
        sort_by = _input_val("profile_sort", "alpha")
        if sort_by == "alpha":
            merged = merged.sort_values("Tetramer")
        elif sort_by == "stepclass":
            merged = merged.sort_values(["Step_class", "Central_step", "Tetramer"])
        elif sort_by in frames:
            merged = merged.sort_values(sort_by, ascending=True)
        else:
            merged = merged.sort_values("Tetramer")

        type_lbl = get_coord_type_label(coord)
        coord_disp = f"{coord} {type_lbl}".strip()

        fig = go.Figure()

        for key in selected_ds:
            if key not in frames or key not in merged.columns:
                continue
            mk = DATASET_MARKERS.get(key, dict(symbol="circle", color="#555555",
                                                size=6, name_suffix=key))
            label = avail[key][0] if key in avail else key
            hover = [
                f"<b>{row['Tetramer']}</b><br>"
                f"Central step: {row['Central_step']} ({row['Step_class']})<br>"
                f"{label}: {row[key]:.4f}"
                for _, row in merged.iterrows()
            ]
            fig.add_trace(go.Scatter(
                x=merged["Tetramer"],
                y=merged[key],
                mode="markers",
                name=label,
                marker=dict(
                    symbol=mk["symbol"],
                    color=mk["color"],
                    size=mk["size"],
                    opacity=0.80,
                    line=dict(width=0.6, color="white"),
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            ))

        fig.update_layout(
            **PLOT_LAYOUT_BASE,
            font=dict(**_FONT_BASE, size=fs),
            width=w,
            height=h,
            title=dict(
                text=f"Coordinate profile — {coord_disp}",
                x=0.5, font=dict(size=fs + 1),
            ),
            xaxis_title="Tetramer",
            yaxis_title=coord_disp,
        )
        fig.update_layout(legend_font_size=xfs)
        n_pts = len(merged)
        auto_tick_fs = max(5, min(11, int(round(400 / max(n_pts, 1)))))
        tick_fs = max(auto_tick_fs, xfs)
        fig.update_xaxes(tickangle=90, tickfont=dict(size=tick_fs))
        return fig

    # ================================================================
    # Downloads
    # ================================================================

    def _scatter_bytes(fmt: str) -> bytes:
        # Read the figure from the temp file written by scatter_plot().
        # This avoids re-entering the reactive graph from the download handler.
        import plotly.io as _pio
        fig = None
        try:
            if os.path.exists(_scatter_tmp):
                with open(_scatter_tmp) as _f:
                    fig = _pio.from_json(_f.read())
        except Exception:
            pass
        if fig is None:
            fig = _build_scatter()
        try:
            if fmt == "png":
                return fig.to_image(format="png", scale=2)
            elif fmt == "pdf":
                return fig.to_image(format="pdf")
            elif fmt == "svg":
                return fig.to_image(format="svg")
        except Exception as exc:
            msg = str(exc).split("\n")[0][:200]
            return (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="80">'
                f'<text x="10" y="40" font-size="12" fill="#cc0000">'
                f'Export failed: {msg}</text></svg>'
            ).encode()
        return b""

    def _dl_fname(ext: str) -> str:
        base = (_input_val("dl_filename", "tetramer_plot") or "tetramer_plot").strip()
        # Sanitise: keep alphanumeric, dash, underscore, dot
        import re as _re
        base = _re.sub(r'[^\w\-\.]', '_', base).strip('_') or "tetramer_plot"
        return f"{base}.{ext}"

    @render.download(filename=lambda: _dl_fname("png"))
    def dl_png():
        yield _scatter_bytes("png")

    @render.download(filename=lambda: _dl_fname("pdf"))
    def dl_pdf():
        yield _scatter_bytes("pdf")

    @render.download(filename=lambda: _dl_fname("svg"))
    def dl_svg():
        yield _scatter_bytes("svg")

    @render.download(filename=lambda: _dl_fname("csv"))
    def dl_csv():
        df, err = comparison_df()
        if err or df is None:
            yield "error,message\n1,No data\n"
            return
        _, pair_x, pair_y, _ = selected_pair()
        if pair_x is None:
            yield "error,message\n1,No data\n"
            return
        lx = pair_x[0]; ly = pair_y[0]
        cols = ["Tetramer","Central_step","Step_class", lx, ly, "Difference","Abs_difference"]
        cols = [c for c in cols if c in df.columns]
        yield df[cols].to_csv(index=False)


# ---------------------------------------------------------------------------
# Input helpers  (NOTE: these are module-level placeholders;
# the real per-session versions are created inside server() and shadow these)
# ---------------------------------------------------------------------------

def _input_exists(input_id: str) -> bool:
    return False

def _input_val(input_id: str, default=None):
    return default


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = App(app_ui, server)
