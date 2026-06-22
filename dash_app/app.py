from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, callback_context, Patch, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots


STATE_TO_CODE = {
    "Wake": 0,
    "NREM": 1,
    "REM": 2,
    "Sleep": 1,
    "Uncertain": -1,
    "Undefined": -1,
    "Artifact": -2,
}

CODE_TO_COLOR = {
    0: "#1f77b4",   # Wake
    1: "#ff7f0e",   # NREM / Sleep
    2: "#2ca02c",   # REM
    -1: "#9e9e9e",  # uncertain
    -2: "#000000",  # artifact
}

STATE_TO_NUM = {
    "Wake": 0,
    "Sleep": 1,
    "NREM": 1,
    "REM": 2,
    "Uncertain": -1,
    "Undefined": -1,
    "Artifact": -2,
}


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def state_codes(labels):
    return np.array([STATE_TO_NUM.get(str(x), -1) for x in labels], dtype=float)


def labels_at_epoch_midpoints(epoch_df, source_df, label_col):
    mids = (epoch_df["t0_s"].to_numpy(float) + epoch_df["t1_s"].to_numpy(float)) / 2.0

    src = source_df.copy()
    src = src.dropna(subset=["t0_s", "t1_s"])
    src = src.sort_values("t0_s")

    out = np.full(len(epoch_df), "Undefined", dtype=object)

    j = 0
    rows = src[["t0_s", "t1_s", label_col]].to_numpy(object)

    for i, mid in enumerate(mids):
        while j < len(rows) and float(rows[j][1]) <= mid:
            j += 1

        if j < len(rows):
            t0 = float(rows[j][0])
            t1 = float(rows[j][1])
            if t0 <= mid < t1:
                out[i] = str(rows[j][2])

    return out


def ensure_final_scoring(recording_dir: Path, recording_id: str) -> Path:
    final_file = recording_dir / "final_scoring.csv"

    if final_file.exists():
        return final_file

    layer1 = pd.read_csv(recording_dir / "layer1_wake_sleep.csv")

    out = pd.DataFrame()
    out["recording_id"] = recording_id
    out["epoch_id"] = np.arange(len(layer1))
    out["t0_s"] = layer1["t0_s"].astype(float)
    out["t1_s"] = layer1["t1_s"].astype(float)

    init_state = []
    for x in layer1["layer1_label"].fillna("Undefined").astype(str):
        if x == "Wake":
            init_state.append("Wake")
        elif x == "Sleep":
            init_state.append("NREM")
        else:
            init_state.append("Undefined")

    out["final_state"] = init_state
    out["final_code"] = [STATE_TO_CODE.get(x, -1) for x in init_state]
    out["final_source"] = "initial_layer1_dash"
    out["review_status"] = "not_reviewed"
    out["review_notes"] = ""

    out.to_csv(final_file, index=False)

    return final_file


def downsample_window(y, fs, start_s, end_s, max_points=60000):
    y = np.load(y, mmap_mode="r")
    i0 = max(0, int(start_s * fs))
    i1 = min(len(y), int(end_s * fs))

    if i1 <= i0:
        return np.array([]), np.array([])

    n = i1 - i0

    if n <= max_points:
        idx = np.arange(i0, i1)
    else:
        step = int(np.ceil(n / max_points))
        idx = np.arange(i0, i1, step)

    t_min = idx / fs / 60.0
    values = np.asarray(y[idx], dtype=float)

    return t_min, values


def get_recording_id(recording_dir: Path) -> str:
    return recording_dir.name


def load_recording(recording_dir: str):
    recording_dir = Path(recording_dir)

    metadata = read_json(recording_dir / "metadata.json")
    fs = float(metadata["sampling_rate_hz"])
    duration_s = float(metadata["duration_s"])
    recording_id = get_recording_id(recording_dir)

    layer1 = pd.read_csv(recording_dir / "layer1_wake_sleep.csv")

    manual = None
    manual_file = recording_dir / "manual_scoring_aligned.csv"
    if manual_file.exists():
        manual = pd.read_csv(manual_file)

    som = None
    som_file = recording_dir / "somnotate" / "somnotate_results_timeseries.csv"
    if som_file.exists():
        som = pd.read_csv(som_file)

    features = None
    features_file = recording_dir / "epoch_features.csv"
    if features_file.exists():
        features = pd.read_csv(features_file)

    final_file = ensure_final_scoring(recording_dir, recording_id)
    final = pd.read_csv(final_file)

    return {
        "recording_dir": recording_dir,
        "recording_id": recording_id,
        "metadata": metadata,
        "fs": fs,
        "duration_s": duration_s,
        "layer1": layer1,
        "manual": manual,
        "som": som,
        "features": features,
        "final": final,
    }


def scoring_rows_for_window(rec, start_min, end_min):
    layer1 = rec["layer1"].copy()
    window_mask = (
        (layer1["t0_s"].astype(float) < end_min * 60)
        & (layer1["t1_s"].astype(float) > start_min * 60)
    )
    epoch_df = layer1.loc[window_mask, ["t0_s", "t1_s"]].copy()

    if len(epoch_df) == 0:
        return [], [], []

    rows = []
    names = []

    if rec["manual"] is not None:
        labels = labels_at_epoch_midpoints(epoch_df, rec["manual"], "manual_state")
        rows.append(state_codes(labels))
        names.append("Manual")

    layer1_labels = layer1.loc[window_mask, "layer1_label"].fillna("Uncertain").astype(str).to_numpy()
    rows.append(state_codes(layer1_labels))
    names.append("Layer 1")

    if rec["som"] is not None:
        labels = labels_at_epoch_midpoints(epoch_df, rec["som"], "somnotate_state")
        rows.append(state_codes(labels))
        names.append("Somnotate")

    labels = labels_at_epoch_midpoints(epoch_df, rec["final"], "final_state")
    rows.append(state_codes(labels))
    names.append("Final")

    x = ((epoch_df["t0_s"].to_numpy(float) + epoch_df["t1_s"].to_numpy(float)) / 2.0) / 60.0

    return x, rows, names


def make_review_figure(recording_dir: str, start_min: float, window_min: float):
    rec = load_recording(recording_dir)
    end_min = min(rec["duration_s"] / 60.0, start_min + window_min)

    fs = rec["fs"]

    t_eeg, eeg = downsample_window(
        rec["recording_dir"] / "eeg.npy",
        fs,
        start_min * 60,
        end_min * 60,
        max_points=60000,
    )

    t_emg, emg = downsample_window(
        rec["recording_dir"] / "emg.npy",
        fs,
        start_min * 60,
        end_min * 60,
        max_points=60000,
    )

    score_x, score_rows, score_names = scoring_rows_for_window(rec, start_min, end_min)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.18, 0.30, 0.25, 0.27],
        vertical_spacing=0.035,
        subplot_titles=["Scoring rows", "EEG", "EMG", "Probabilities / features"],
    )

    if len(score_rows):
        z = np.vstack(score_rows)

        colorscale = [
            [0.00, CODE_TO_COLOR[-2]],
            [0.20, CODE_TO_COLOR[-1]],
            [0.40, CODE_TO_COLOR[0]],
            [0.70, CODE_TO_COLOR[1]],
            [1.00, CODE_TO_COLOR[2]],
        ]

        fig.add_trace(
            go.Heatmap(
                x=score_x,
                y=score_names,
                z=z,
                zmin=-2,
                zmax=2,
                colorscale=colorscale,
                showscale=False,
                hovertemplate="Time=%{x:.2f} min<br>Layer=%{y}<br>Code=%{z}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scattergl(x=t_eeg, y=eeg, mode="lines", name="EEG", line=dict(width=1)),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(x=t_emg, y=emg, mode="lines", name="EMG", line=dict(width=1)),
        row=3,
        col=1,
    )

    layer1 = rec["layer1"].copy()
    layer1["time_min"] = layer1["t0_s"].astype(float) / 60.0
    wmask = (layer1["time_min"] >= start_min) & (layer1["time_min"] <= end_min)

    if "p_wake" in layer1.columns:
        fig.add_trace(
            go.Scatter(
                x=layer1.loc[wmask, "time_min"],
                y=layer1.loc[wmask, "p_wake"],
                mode="lines",
                name="Layer 1 P(Wake)",
                line=dict(dash="dash"),
            ),
            row=4,
            col=1,
        )

    if "p_sleep" in layer1.columns:
        fig.add_trace(
            go.Scatter(
                x=layer1.loc[wmask, "time_min"],
                y=layer1.loc[wmask, "p_sleep"],
                mode="lines",
                name="Layer 1 P(Sleep)",
                line=dict(dash="dash"),
            ),
            row=4,
            col=1,
        )

    if rec["som"] is not None:
        som = rec["som"].copy()
        if "time_min" not in som.columns:
            som["time_min"] = som["t0_s"].astype(float) / 60.0
        smask = (som["time_min"] >= start_min) & (som["time_min"] <= end_min)

        for col, label in [
            ("p_wake", "Somnotate P(Wake)"),
            ("p_nrem", "Somnotate P(NREM)"),
            ("p_rem", "Somnotate P(REM)"),
        ]:
            if col in som.columns:
                fig.add_trace(
                    go.Scatter(
                        x=som.loc[smask, "time_min"],
                        y=som.loc[smask, col],
                        mode="lines",
                        name=label,
                    ),
                    row=4,
                    col=1,
                )

    if rec["features"] is not None:
        feat = rec["features"].copy()
        if "time_min" not in feat.columns:
            feat["time_min"] = feat["t0_s"].astype(float) / 60.0
        fmask = (feat["time_min"] >= start_min) & (feat["time_min"] <= end_min)

        for candidate in ["emg_rms_z", "emg_rms_zscore", "emg_rms"]:
            if candidate in feat.columns:
                fig.add_trace(
                    go.Scatter(
                        x=feat.loc[fmask, "time_min"],
                        y=feat.loc[fmask, candidate],
                        mode="lines",
                        name=candidate,
                        opacity=0.55,
                    ),
                    row=4,
                    col=1,
                )
                break

    fig.update_layout(
        height=900,
        margin=dict(l=70, r=25, t=70, b=45),
        hovermode="x unified",
        dragmode="select",
        uirevision=f"{recording_dir}-{start_min}-{window_min}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    fig.update_xaxes(range=[start_min, end_min])
    fig.update_xaxes(title_text="Time (min)", row=4, col=1)

    return fig


def apply_manual_label(recording_dir: str, start_min: float, end_min: float, label: str):
    rec = load_recording(recording_dir)
    final_file = ensure_final_scoring(rec["recording_dir"], rec["recording_id"])
    final = pd.read_csv(final_file)

    start_s = float(start_min) * 60.0
    end_s = float(end_min) * 60.0

    mask = (
        (final["t0_s"].astype(float) < end_s)
        & (final["t1_s"].astype(float) > start_s)
    )

    if int(mask.sum()) == 0:
        return False, "No epochs found in selected interval."

    final.loc[mask, "final_state"] = label
    final.loc[mask, "final_code"] = STATE_TO_CODE.get(label, -1)
    final.loc[mask, "final_source"] = "dash_manual"
    final.loc[mask, "review_status"] = "reviewed"
    final.loc[mask, "review_notes"] = "dash edit"

    final.to_csv(final_file, index=False)

    return True, f"Saved {label} for {int(mask.sum())} epochs."


def apply_source_label(recording_dir: str, start_min: float, end_min: float, source_name: str):
    rec = load_recording(recording_dir)
    final_file = ensure_final_scoring(rec["recording_dir"], rec["recording_id"])
    final = pd.read_csv(final_file)

    start_s = float(start_min) * 60.0
    end_s = float(end_min) * 60.0

    mask = (
        (final["t0_s"].astype(float) < end_s)
        & (final["t1_s"].astype(float) > start_s)
    )

    if int(mask.sum()) == 0:
        return False, "No epochs found in selected interval."

    epoch_df = final[["t0_s", "t1_s"]].copy()

    if source_name == "Manual":
        if rec["manual"] is None:
            return False, "Manual scoring not found."
        source_labels = labels_at_epoch_midpoints(epoch_df, rec["manual"], "manual_state")
        final_source = "dash_accept_manual"

    elif source_name == "Somnotate":
        if rec["som"] is None:
            return False, "Somnotate scoring not found."
        source_labels = labels_at_epoch_midpoints(epoch_df, rec["som"], "somnotate_state")
        final_source = "dash_accept_somnotate"

    elif source_name == "Layer 1":
        source_labels = rec["layer1"]["layer1_label"].fillna("Undefined").astype(str).to_numpy()
        converted = []
        for x in source_labels:
            if x == "Wake":
                converted.append("Wake")
            elif x == "Sleep":
                converted.append("NREM")
            else:
                converted.append("Undefined")
        source_labels = np.asarray(converted, dtype=object)
        final_source = "dash_accept_layer1"

    else:
        return False, f"Unknown source: {source_name}"

    selected = np.asarray(source_labels, dtype=object)[mask.to_numpy()]

    final.loc[mask, "final_state"] = selected
    final.loc[mask, "final_code"] = [STATE_TO_CODE.get(str(x), -1) for x in selected]
    final.loc[mask, "final_source"] = final_source
    final.loc[mask, "review_status"] = "reviewed"
    final.loc[mask, "review_notes"] = "dash source approval"

    final.to_csv(final_file, index=False)

    return True, f"Accepted {source_name} for {int(mask.sum())} epochs."


app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "12px"},
    children=[
        html.H2("Sleep Stage QC Dash prototype"),

        html.Div(
            style={"display": "flex", "gap": "8px", "alignItems": "center"},
            children=[
                html.Label("Prepared recording folder:"),
                dcc.Input(
                    id="recording-dir-input",
                    type="text",
                    placeholder="/path/to/prepared/recording_folder",
                    style={"width": "650px"},
                ),
                html.Button("Load recording", id="load-recording", n_clicks=0),
            ],
        ),

        html.Div(id="load-status", style={"marginTop": "8px"}),

        dcc.Store(id="recording-dir-store"),
        dcc.Store(id="window-store", data={"start_min": 0.0, "window_min": 15.0}),
        dcc.Store(id="selected-interval-store"),

        html.Hr(),

        html.Div(
            id="review-panel",
            style={"display": "none"},
            children=[
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr 2fr 1fr 1fr",
                        "gap": "6px",
                        "alignItems": "center",
                        "marginBottom": "6px",
                    },
                    children=[
                        html.Button("◀ 15 min", id="back-15"),
                        html.Button("◀ 5 min", id="back-5"),
                        html.Div(id="window-label", style={"textAlign": "center", "fontWeight": "bold"}),
                        html.Button("5 min ▶", id="forward-5"),
                        html.Button("15 min ▶", id="forward-15"),
                    ],
                ),

                dcc.Graph(
                    id="qc-graph",
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToAdd": ["select2d", "pan2d", "zoom2d", "resetScale2d"],
                    },
                ),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(6, 1fr)",
                        "gap": "6px",
                        "marginTop": "8px",
                    },
                    children=[
                        html.Button("1 = Wake", id="score-wake", n_clicks=0),
                        html.Button("2 = NREM", id="score-nrem", n_clicks=0),
                        html.Button("3 = REM", id="score-rem", n_clicks=0),
                        html.Button("S = Somnotate", id="score-somnotate", n_clicks=0),
                        html.Button("L = Layer 1", id="score-layer1", n_clicks=0),
                        html.Button("M = Manual", id="score-manual", n_clicks=0),
                    ],
                ),

                html.Div(
                    id="selected-interval-label",
                    style={"marginTop": "8px", "fontWeight": "bold", "color": "#1357c8"},
                ),

                html.Div(id="score-status", style={"marginTop": "8px"}),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr 2fr 1fr 1fr",
                        "gap": "6px",
                        "alignItems": "center",
                        "marginTop": "8px",
                    },
                    children=[
                        html.Button("◀ 15 min", id="back-15-bottom"),
                        html.Button("◀ 5 min", id="back-5-bottom"),
                        html.Div("Navigation repeated here to avoid scrolling.", style={"textAlign": "center"}),
                        html.Button("5 min ▶", id="forward-5-bottom"),
                        html.Button("15 min ▶", id="forward-15-bottom"),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("recording-dir-store", "data"),
    Output("load-status", "children"),
    Output("review-panel", "style"),
    Output("qc-graph", "figure"),
    Output("window-label", "children"),
    Input("load-recording", "n_clicks"),
    State("recording-dir-input", "value"),
    State("window-store", "data"),
    prevent_initial_call=True,
)
def load_recording_callback(n, recording_dir, window_data):
    if not recording_dir:
        return no_update, "Please enter a recording folder.", {"display": "none"}, no_update, ""

    recording_dir = str(Path(recording_dir).expanduser())

    try:
        rec = load_recording(recording_dir)
        start_min = float(window_data.get("start_min", 0.0))
        window_min = float(window_data.get("window_min", 15.0))
        fig = make_review_figure(recording_dir, start_min, window_min)
        end_min = min(rec["duration_s"] / 60.0, start_min + window_min)

        return (
            recording_dir,
            f"Loaded: {recording_dir}",
            {"display": "block"},
            fig,
            f"Window: {start_min:.2f}–{end_min:.2f} min",
        )

    except Exception as e:
        return no_update, f"Could not load recording: {e}", {"display": "none"}, no_update, ""


@app.callback(
    Output("window-store", "data"),
    Input("back-15", "n_clicks"),
    Input("back-5", "n_clicks"),
    Input("forward-5", "n_clicks"),
    Input("forward-15", "n_clicks"),
    Input("back-15-bottom", "n_clicks"),
    Input("back-5-bottom", "n_clicks"),
    Input("forward-5-bottom", "n_clicks"),
    Input("forward-15-bottom", "n_clicks"),
    State("window-store", "data"),
    State("recording-dir-store", "data"),
    prevent_initial_call=True,
)
def navigate_window(*args):
    window_data = args[-2]
    recording_dir = args[-1]

    if not recording_dir:
        return no_update

    triggered = callback_context.triggered_id

    delta = {
        "back-15": -15,
        "back-5": -5,
        "forward-5": 5,
        "forward-15": 15,
        "back-15-bottom": -15,
        "back-5-bottom": -5,
        "forward-5-bottom": 5,
        "forward-15-bottom": 15,
    }.get(triggered, 0)

    rec = load_recording(recording_dir)
    duration_min = rec["duration_s"] / 60.0

    window_min = float(window_data.get("window_min", 15.0))
    old_start = float(window_data.get("start_min", 0.0))

    new_start = max(0.0, min(duration_min - window_min, old_start + delta))

    return {"start_min": new_start, "window_min": window_min}


@app.callback(
    Output("qc-graph", "figure", allow_duplicate=True),
    Output("window-label", "children", allow_duplicate=True),
    Output("selected-interval-store", "data", allow_duplicate=True),
    Input("window-store", "data"),
    State("recording-dir-store", "data"),
    prevent_initial_call=True,
)
def update_figure_for_window(window_data, recording_dir):
    if not recording_dir:
        return no_update, no_update, no_update

    rec = load_recording(recording_dir)

    start_min = float(window_data.get("start_min", 0.0))
    window_min = float(window_data.get("window_min", 15.0))
    end_min = min(rec["duration_s"] / 60.0, start_min + window_min)

    fig = make_review_figure(recording_dir, start_min, window_min)

    return fig, f"Window: {start_min:.2f}–{end_min:.2f} min", None


@app.callback(
    Output("selected-interval-store", "data"),
    Output("selected-interval-label", "children"),
    Output("qc-graph", "figure", allow_duplicate=True),
    Input("qc-graph", "selectedData"),
    State("qc-graph", "figure"),
    prevent_initial_call=True,
)
def update_selection(selected_data, fig):
    if not selected_data:
        return no_update, no_update, no_update

    x0 = x1 = None

    if "range" in selected_data and "x" in selected_data["range"]:
        x0, x1 = selected_data["range"]["x"]

    elif "points" in selected_data and selected_data["points"]:
        xs = [p["x"] for p in selected_data["points"] if "x" in p]
        if len(xs) >= 2:
            x0, x1 = min(xs), max(xs)

    if x0 is None or x1 is None:
        return no_update, "Could not read selected interval.", no_update

    x0 = float(x0)
    x1 = float(x1)

    if x1 < x0:
        x0, x1 = x1, x0

    if x1 <= x0:
        return no_update, "Invalid selected interval.", no_update

    selected = {"start_min": x0, "end_min": x1}

    patched = Patch()
    patched["layout"]["shapes"] = [
        {
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": x0,
            "x1": x1,
            "y0": 0,
            "y1": 1,
            "fillcolor": "rgba(0, 120, 255, 0.12)",
            "line": {"color": "rgba(0, 90, 220, 0.95)", "width": 2},
            "layer": "above",
        }
    ]

    dur_s = (x1 - x0) * 60.0
    label = f"Selected interval: {x0:.2f}–{x1:.2f} min ({dur_s:.1f} s)"

    return selected, label, patched


@app.callback(
    Output("score-status", "children"),
    Output("qc-graph", "figure", allow_duplicate=True),
    Input("score-wake", "n_clicks"),
    Input("score-nrem", "n_clicks"),
    Input("score-rem", "n_clicks"),
    Input("score-somnotate", "n_clicks"),
    Input("score-layer1", "n_clicks"),
    Input("score-manual", "n_clicks"),
    State("selected-interval-store", "data"),
    State("recording-dir-store", "data"),
    State("window-store", "data"),
    prevent_initial_call=True,
)
def apply_score(n_wake, n_nrem, n_rem, n_som, n_l1, n_man, selected, recording_dir, window_data):
    if not recording_dir:
        return "No recording loaded.", no_update

    if not selected:
        return "Select an interval first. The app will not score the full visible window automatically.", no_update

    triggered = callback_context.triggered_id

    start_min = float(selected["start_min"])
    end_min = float(selected["end_min"])

    if triggered == "score-wake":
        ok, msg = apply_manual_label(recording_dir, start_min, end_min, "Wake")
    elif triggered == "score-nrem":
        ok, msg = apply_manual_label(recording_dir, start_min, end_min, "NREM")
    elif triggered == "score-rem":
        ok, msg = apply_manual_label(recording_dir, start_min, end_min, "REM")
    elif triggered == "score-somnotate":
        ok, msg = apply_source_label(recording_dir, start_min, end_min, "Somnotate")
    elif triggered == "score-layer1":
        ok, msg = apply_source_label(recording_dir, start_min, end_min, "Layer 1")
    elif triggered == "score-manual":
        ok, msg = apply_source_label(recording_dir, start_min, end_min, "Manual")
    else:
        return "Unknown scoring action.", no_update

    if not ok:
        return msg, no_update

    # Patch only the scoring heatmap, not EEG/EMG traces.
    rec = load_recording(recording_dir)
    start_window = float(window_data.get("start_min", 0.0))
    window_min = float(window_data.get("window_min", 15.0))
    end_window = min(rec["duration_s"] / 60.0, start_window + window_min)

    score_x, score_rows, score_names = scoring_rows_for_window(rec, start_window, end_window)

    patched = Patch()
    if len(score_rows):
        patched["data"][0]["z"] = np.vstack(score_rows).tolist()
        patched["data"][0]["y"] = score_names

    return msg, patched


app.clientside_callback(
    """
    function(id) {
        document.addEventListener("keydown", function(e) {
            const tag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
            if (tag === "input" || tag === "textarea" || tag === "select") {
                return;
            }
            if (e.ctrlKey || e.metaKey || e.altKey) {
                return;
            }

            const key = e.key.toLowerCase();
            const map = {
                "1": "score-wake",
                "2": "score-nrem",
                "3": "score-rem",
                "s": "score-somnotate",
                "l": "score-layer1",
                "m": "score-manual"
            };

            if (map[key]) {
                e.preventDefault();
                const btn = document.getElementById(map[key]);
                if (btn) {
                    btn.click();
                }
            }
        });

        return window.dash_clientside.no_update;
    }
    """,
    Output("score-status", "data-dummy"),
    Input("review-panel", "id"),
)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
