from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from components.plotly_relayout_viewer import plotly_relayout_viewer
import streamlit.components.v1 as components

try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None

from plotly.subplots import make_subplots
from scipy.io import loadmat


# =============================================================================
# BASIC HELPERS
# =============================================================================

st.set_page_config(
    page_title="Sleep Stage QC v2",
    layout="wide",
)


def run_command(cmd):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )


def load_manifest(project_root):
    path = Path(project_root) / "recordings_manifest.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


def safe_mat_keys(mat_file):
    p = Path(mat_file).expanduser()

    if not p.exists():
        return []

    try:
        d = loadmat(p)
        return [k for k in d.keys() if not k.startswith("__")]
    except NotImplementedError:
        import h5py

        keys = []

        with h5py.File(p, "r") as f:
            def visitor(name, obj):
                if hasattr(obj, "shape"):
                    keys.append(name)
            f.visititems(visitor)

        return keys


def load_recording_arrays(recording_dir):
    eeg = np.load(Path(recording_dir) / "eeg.npy", mmap_mode="r")
    emg = np.load(Path(recording_dir) / "emg.npy", mmap_mode="r")
    metadata = json.loads((Path(recording_dir) / "metadata.json").read_text())
    return eeg, emg, metadata


def downsample_trace(x, t, max_points=25000):
    n = len(x)

    if n <= max_points:
        return t, x

    step = int(np.ceil(n / max_points))
    return t[::step], x[::step]


def make_state_bar(df, time_col, label_col, name):
    code_map = {
        "Wake": 0,
        "Awake": 0,
        "Sleep": 1,
        "NREM": 2,
        "REM": 3,
        "Uncertain": 4,
        "Undefined": 4,
        "Artifact": 5,
    }

    labels = df[label_col].fillna("Undefined").astype(str)
    codes = labels.map(code_map).fillna(4).to_numpy()

    return go.Heatmap(
        x=df[time_col],
        y=[name],
        z=[codes],
        zmin=0,
        zmax=5,
        colorscale=[
            [0.00, "#4e79a7"], [0.16, "#4e79a7"],  # Wake
            [0.17, "#f28e2b"], [0.33, "#f28e2b"],  # Sleep
            [0.34, "#f28e2b"], [0.50, "#f28e2b"],  # NREM
            [0.51, "#2ca25f"], [0.67, "#2ca25f"],  # REM
            [0.68, "#bdbdbd"], [0.84, "#bdbdbd"],  # Uncertain/Undefined
            [0.85, "#000000"], [1.00, "#000000"],  # Artifact
        ],
        showscale=False,
        customdata=labels,
        hovertemplate="time=%{x:.2f} min<br>" + name + "=%{customdata}<extra></extra>",
        name=name,
    )


def labels_at_epoch_midpoints(epoch_df, label_df, label_col, default="Undefined"):
    if label_df is None or len(label_df) == 0 or label_col not in label_df.columns:
        return np.array([default] * len(epoch_df), dtype=object)

    starts = label_df["t0_s"].to_numpy(dtype=float)
    ends = label_df["t1_s"].to_numpy(dtype=float)
    labels = label_df[label_col].fillna(default).astype(str).to_numpy()

    mids = (
        epoch_df["t0_s"].to_numpy(dtype=float)
        + epoch_df["t1_s"].to_numpy(dtype=float)
    ) / 2

    idx = np.searchsorted(starts, mids, side="right") - 1
    out = np.array([default] * len(epoch_df), dtype=object)

    valid = (idx >= 0) & (idx < len(starts))
    valid_idx = idx[valid]
    valid_mid = mids[valid]

    inside = valid_mid < ends[valid_idx]
    out[np.where(valid)[0][inside]] = labels[valid_idx[inside]]

    return out


def state_codes(labels):
    code_map = {
        "Wake": 0,
        "Awake": 0,
        "Sleep": 1,
        "NREM": 2,
        "REM": 3,
        "Uncertain": 4,
        "Undefined": 4,
        "Artifact": 5,
    }

    return pd.Series(labels).astype(str).map(code_map).fillna(4).to_numpy(dtype=float)


def downsample_full_trace_for_browser(x, fs, max_points=250000):
    """
    Browser-friendly full-recording signal.

    If the recording is short enough, use the raw signal.
    If it is long, use min/max envelope bins so EMG bursts are still visible.
    """
    n = len(x)

    if n <= max_points:
        idx = np.arange(n)
        return idx / fs / 60, np.asarray(x, dtype=float), "raw"

    # Use min/max envelope. This preserves bursts better than taking every nth point.
    n_bins = max_points // 2
    bin_size = int(np.ceil(n / n_bins))
    usable = (n // bin_size) * bin_size

    x_trim = np.asarray(x[:usable], dtype=float).reshape(-1, bin_size)

    mins = np.nanmin(x_trim, axis=1)
    maxs = np.nanmax(x_trim, axis=1)

    centers = (np.arange(len(mins)) * bin_size + bin_size / 2) / fs / 60

    t = np.repeat(centers, 2)
    y = np.empty(len(mins) * 2, dtype=float)
    y[0::2] = mins
    y[1::2] = maxs

    return t, y, "envelope"


def make_qc_plot(recording_dir, start_min, window_min):
    recording_dir = Path(recording_dir)

    metadata = read_json_fast(recording_dir / "metadata.json")
    fs = float(metadata["sampling_rate_hz"])
    duration_min = float(metadata["duration_s"]) / 60

    layer1_file = recording_dir / "layer1_wake_sleep.csv"
    features_file = recording_dir / "epoch_features.csv"
    manual_file = recording_dir / "manual_scoring_aligned.csv"
    som_file = recording_dir / "somnotate" / "somnotate_results_timeseries.csv"

    if not layer1_file.exists():
        raise FileNotFoundError(layer1_file)

    # Full-recording traces, browser-friendly and disk-cached.
    cache_dir = recording_dir / "plot_cache"

    t_eeg, eeg_plot, eeg_mode = load_or_build_display_cache(
        recording_dir / "eeg.npy",
        cache_dir / "eeg_display.npz",
        fs,
        max_points=350000,
    )

    t_emg, emg_plot, emg_mode = load_or_build_display_cache(
        recording_dir / "emg.npy",
        cache_dir / "emg_display.npz",
        fs,
        max_points=350000,
    )

    layer1 = read_csv_fast(layer1_file)
    layer1["time_min"] = layer1["t0_s"] / 60

    features = None
    if features_file.exists():
        features = read_csv_fast(features_file)
        features["time_min"] = features["t0_s"] / 60

    som = None
    if som_file.exists():
        som = read_csv_fast(som_file)
        if "time_min" not in som.columns:
            som["time_min"] = som["t0_s"] / 60

    # ------------------------------------------------------------------
    # Scoring rows across full recording
    # ------------------------------------------------------------------
    scoring_rows = []
    scoring_names = []
    scoring_labels = []

    if manual_file.exists():
        manual = read_csv_fast(manual_file)
        if all(c in manual.columns for c in ["t0_s", "t1_s"]):
            manual_labels = labels_at_epoch_midpoints(layer1, manual, "manual_state")
            scoring_rows.append(state_codes(manual_labels))
            scoring_names.append("Manual")
            scoring_labels.append(manual_labels)

    layer1_labels = layer1["layer1_label"].fillna("Uncertain").astype(str).to_numpy()
    scoring_rows.append(state_codes(layer1_labels))
    scoring_names.append("Layer 1")
    scoring_labels.append(layer1_labels)

    if som is not None:
        som_labels = labels_at_epoch_midpoints(layer1, som, "somnotate_state")
        scoring_rows.append(state_codes(som_labels))
        scoring_names.append("Somnotate")
        scoring_labels.append(som_labels)

    # Final scoring, if it exists
    final_reviewed_bouts = []
    final_file = recording_dir / "final_scoring.csv"

    if final_file.exists():
        final = read_csv_fast(final_file)

        if all(c in final.columns for c in ["t0_s", "t1_s", "final_state"]):
            final_labels = labels_at_epoch_midpoints(layer1, final, "final_state")
            scoring_rows.append(state_codes(final_labels))
            scoring_names.append("Final")
            scoring_labels.append(final_labels)

            # Compact only reviewed intervals for shading the signal.
            if "review_status" in final.columns:
                reviewed = final[
                    final["review_status"].astype(str).str.lower() == "reviewed"
                ].copy()
            else:
                reviewed = pd.DataFrame()

            if len(reviewed):
                reviewed = reviewed.sort_values("t0_s").reset_index(drop=True)

                current = None

                for _, rr in reviewed.iterrows():
                    state = str(rr["final_state"])
                    t0 = float(rr["t0_s"])
                    t1 = float(rr["t1_s"])

                    if current is None:
                        current = {"state": state, "t0_s": t0, "t1_s": t1}
                        continue

                    # Merge contiguous reviewed epochs with the same label.
                    if state == current["state"] and t0 <= current["t1_s"] + 1.01:
                        current["t1_s"] = max(current["t1_s"], t1)
                    else:
                        final_reviewed_bouts.append(current)
                        current = {"state": state, "t0_s": t0, "t1_s": t1}

                if current is not None:
                    final_reviewed_bouts.append(current)

    # Safety: never show Final as an extra scoring row.
    if "Final" in scoring_names:
        keep = [i for i, name in enumerate(scoring_names) if name != "Final"]
        scoring_rows = [scoring_rows[i] for i in keep]
        scoring_names = [scoring_names[i] for i in keep]
        scoring_labels = [scoring_labels[i] for i in keep]

    # Final scoring is visual feedback only: colored shading over raw traces.
    # It should not appear as a separate scoring row.
    if "Final" in scoring_names:
        keep = [i for i, name in enumerate(scoring_names) if name != "Final"]
        scoring_rows = [scoring_rows[i] for i in keep]
        scoring_names = [scoring_names[i] for i in keep]
        scoring_labels = [scoring_labels[i] for i in keep]

    z = np.vstack(scoring_rows)
    custom = np.vstack(scoring_labels)

    color_wake = "#1f77b4"
    color_layer1_sleep = "#f7c6d9"  # light pink for Layer 1 Sleep
    color_nrem_sleep = "#ff7f0e"
    color_rem = "#2ca02c"
    color_uncertain = "#9e9e9e"

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.13, 0.25, 0.25, 0.17, 0.20],
        subplot_titles=[
            "Scoring layers",
            f"Raw EEG ({eeg_mode} display)",
            f"Raw EMG ({emg_mode} display)",
            "EMG RMS z",
            "Layer 1 + Somnotate probabilities",
        ],
    )

    # ------------------------------------------------------------------
    # Row 1 — scoring layers
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Heatmap(
            x=layer1["time_min"],
            y=scoring_names,
            z=z,
            zmin=0,
            zmax=5,
            colorscale=[
                [0.00, color_wake], [0.16, color_wake],
                [0.17, color_layer1_sleep], [0.33, color_layer1_sleep],
                [0.34, color_nrem_sleep], [0.50, color_nrem_sleep],
                [0.51, color_rem], [0.67, color_rem],
                [0.68, color_uncertain], [0.84, color_uncertain],
                [0.85, "#000000"], [1.00, "#000000"],
            ],
            showscale=False,
            customdata=custom,
            hovertemplate="time=%{x:.2f} min<br>row=%{y}<br>label=%{customdata}<extra></extra>",
            name="Scoring",
        ),
        row=1,
        col=1,
    )

    # ------------------------------------------------------------------
    # Rows 2–3 — full EEG/EMG
    # Use Scattergl for smoother interaction with many points.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Override EEG/EMG display with raw data from the current visible window.
    # This preserves signal quality while the app-level window slider provides navigation.
    # ------------------------------------------------------------------
    visible_end_min_for_signal = min(float(duration_min), float(start_min) + float(window_min))

    t_eeg, eeg_plot, eeg_mode = make_raw_window_trace_cached(
        str(recording_dir / "eeg.npy"),
        float(fs),
        round(float(start_min), 4),
        round(float(visible_end_min_for_signal), 4),
        200000,
        file_mtime_or_zero_local(recording_dir / "eeg.npy"),
    )

    t_emg, emg_plot, emg_mode = make_raw_window_trace_cached(
        str(recording_dir / "emg.npy"),
        float(fs),
        round(float(start_min), 4),
        round(float(visible_end_min_for_signal), 4),
        200000,
        file_mtime_or_zero_local(recording_dir / "emg.npy"),
    )

    fig.add_trace(
        go.Scattergl(
            x=t_eeg,
            y=eeg_plot,
            mode="lines",
            name="Raw EEG",
            line=dict(color="black", width=0.55),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=t_emg,
            y=emg_plot,
            mode="lines",
            name="Raw EMG",
            line=dict(color="black", width=0.55),
        ),
        row=3,
        col=1,
    )

    # ------------------------------------------------------------------
    # Row 4 — full EMG RMS z
    # ------------------------------------------------------------------
    if features is not None and "emg_rms_z" in features.columns:
        fig.add_trace(
            go.Scattergl(
                x=features["time_min"],
                y=features["emg_rms_z"],
                mode="lines",
                name="EMG RMS z",
                line=dict(color="#d62728", width=1.0),
            ),
            row=4,
            col=1,
        )

    # ------------------------------------------------------------------
    # Row 5 — full probabilities
    # Layer 1 dashed, Somnotate solid
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Scattergl(
            x=layer1["time_min"],
            y=layer1["layer1_P_Wake"],
            mode="lines",
            name="Layer 1 P(Wake)",
            line=dict(color=color_wake, width=1.1, dash="dash"),
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=layer1["time_min"],
            y=layer1["layer1_P_Sleep"],
            mode="lines",
            name="Layer 1 P(Sleep)",
            line=dict(color=color_nrem_sleep, width=1.1, dash="dash"),
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=layer1["time_min"],
            y=layer1["layer1_uncertainty"],
            mode="lines",
            name="Layer 1 uncertainty",
            line=dict(color=color_uncertain, width=1.0, dash="dot"),
        ),
        row=5,
        col=1,
    )

    if som is not None:
        prob_cols = [c for c in som.columns if c.startswith("somnotate_P_")]

        color_map = {
            "somnotate_P_Wake": color_wake,
            "somnotate_P_Awake": color_wake,
            "somnotate_P_NREM": color_nrem_sleep,
            "somnotate_P_Sleep": color_nrem_sleep,
            "somnotate_P_REM": color_rem,
            "somnotate_P_Undefined": color_uncertain,
            "somnotate_P_Uncertain": color_uncertain,
        }

        for col in prob_cols:
            fig.add_trace(
                go.Scattergl(
                    x=som["time_min"],
                    y=som[col],
                    mode="lines",
                    name=col.replace("somnotate_P_", "Somnotate P(") + ")",
                    line=dict(color=color_map.get(col, "#555555"), width=1.4),
                ),
                row=5,
                col=1,
            )

        if "somnotate_uncertainty" in som.columns:
            fig.add_trace(
                go.Scattergl(
                    x=som["time_min"],
                    y=som["somnotate_uncertainty"],
                    mode="lines",
                    name="Somnotate uncertainty",
                    line=dict(color=color_uncertain, width=1.2),
                ),
                row=5,
                col=1,
            )

    # ------------------------------------------------------------------
    # Shade reviewed final scoring decisions on the signal/probability panels
    # ------------------------------------------------------------------
    final_shade_colors = {
        "Wake": "rgba(31, 119, 180, 0.12)",
        "NREM": "rgba(255, 127, 14, 0.12)",
        "Sleep": "rgba(255, 127, 14, 0.12)",
        "REM": "rgba(44, 160, 44, 0.12)",
        "Uncertain": "rgba(150, 150, 150, 0.12)",
        "Undefined": "rgba(150, 150, 150, 0.12)",
        "Artifact": "rgba(0, 0, 0, 0.10)",
    }

    for bout in final_reviewed_bouts:
        x0 = float(bout["t0_s"]) / 60
        x1 = float(bout["t1_s"]) / 60
        state = str(bout["state"])
        fill = final_shade_colors.get(state, "rgba(150, 150, 150, 0.10)")

        for rr in [2, 3, 4, 5]:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=fill,
                line_width=0,
                row=rr,
                col=1,
            )

    # ------------------------------------------------------------------
    # Initial visible window.
    # Full data are loaded; Plotly range slider lets the user slide through it.
    # ------------------------------------------------------------------
    initial_end = min(duration_min, float(start_min) + float(window_min))

    for r in range(1, 6):
        fig.update_xaxes(range=[float(start_min), float(initial_end)], row=r, col=1)

    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.08,
        ),
        row=5,
        col=1,
    )

    # ------------------------------------------------------------------
    # Interaction behavior
    # Keep x-axis navigation active so pan and the range slider work.
    # Restrict y-axis zooming on overview-style rows.
    # ------------------------------------------------------------------

    # X-axis must stay interactive on all rows, otherwise Plotly pan breaks.
    for r in range(1, 6):
        fig.update_xaxes(fixedrange=False, row=r, col=1)

    # Y-axis behavior:
    # - scoring, EMG RMS, probabilities: fixed y-scale
    # - EEG and EMG: y zoom allowed for signal inspection
    fig.update_yaxes(fixedrange=True, row=1, col=1)
    fig.update_yaxes(fixedrange=False, title_text="EEG", row=2, col=1)
    fig.update_yaxes(fixedrange=False, title_text="EMG", row=3, col=1)
    fig.update_yaxes(fixedrange=True, title_text="EMG RMS z", row=4, col=1)
    fig.update_yaxes(fixedrange=True, title_text="prob.", range=[-0.05, 1.05], row=5, col=1)

    fig.update_xaxes(title_text="Time from recording start (min)", row=5, col=1)

    fig.update_layout(
        height=980,
        hovermode="x unified",
        margin=dict(l=70, r=30, t=80, b=150),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.75)",
            borderwidth=0,
        ),
        title=(
            f"Raw review window: {float(start_min):.1f}–{float(initial_end):.1f} min "
            f"of {duration_min:.1f} min"
        ),
        dragmode="pan",
    )

    return fig



# =============================================================================
# FAST CACHED LOADERS
# =============================================================================

@st.cache_data(show_spinner=False)
def read_csv_fast_cached(path_str, mtime):
    return pd.read_csv(path_str)


def read_csv_fast(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    return read_csv_fast_cached(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def read_json_fast_cached(path_str, mtime):
    return json.loads(Path(path_str).read_text())


def read_json_fast(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    return read_json_fast_cached(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def downsample_npy_for_browser_cached(npy_path_str, fs, mtime, max_points):
    """
    Cached browser-friendly full-recording display trace.

    This avoids recomputing the min/max envelope every time the app reruns.
    """
    x = np.load(npy_path_str, mmap_mode="r")
    n = len(x)

    if n <= max_points:
        idx = np.arange(n)
        return idx / fs / 60, np.asarray(x, dtype=float), "raw"

    n_bins = max_points // 2
    bin_size = int(np.ceil(n / n_bins))
    usable = (n // bin_size) * bin_size

    x_trim = np.asarray(x[:usable], dtype=float).reshape(-1, bin_size)

    mins = np.nanmin(x_trim, axis=1)
    maxs = np.nanmax(x_trim, axis=1)

    centers = (np.arange(len(mins)) * bin_size + bin_size / 2) / fs / 60

    t = np.repeat(centers, 2)
    y = np.empty(len(mins) * 2, dtype=float)
    y[0::2] = mins
    y[1::2] = maxs

    return t, y, "envelope"


def downsample_npy_for_browser(npy_path, fs, max_points=350000):
    npy_path = Path(npy_path)

    if not npy_path.exists():
        raise FileNotFoundError(npy_path)

    return downsample_npy_for_browser_cached(
        str(npy_path),
        float(fs),
        npy_path.stat().st_mtime,
        int(max_points),
    )



# =============================================================================
# DISK PLOT CACHE
# =============================================================================

def make_display_trace_from_npy(npy_path, fs, max_points=350000):
    """
    Build a continuous full-recording display trace.

    This keeps the whole recording available for Plotly range-slider navigation,
    but avoids the artificial sawtooth/min-max envelope appearance.

    For large recordings, it uses uniform decimation:
        x[0], x[step], x[2*step], ...

    This preserves a natural-looking continuous EEG/EMG trace while keeping
    the browser load manageable.
    """
    x = np.load(npy_path, mmap_mode="r")
    n = len(x)

    if n <= max_points:
        idx = np.arange(n, dtype=np.int64)
        mode = "raw"
    else:
        step = int(np.ceil(n / int(max_points)))
        idx = np.arange(0, n, step, dtype=np.int64)
        mode = f"continuous-decimated-{step}x"

    y = np.asarray(x[idx], dtype=np.float32)
    t = idx.astype(np.float64) / float(fs) / 60.0

    return t.astype(np.float32), y, mode


def cache_is_valid(source_path, cache_path):
    source_path = Path(source_path)
    cache_path = Path(cache_path)

    if not cache_path.exists():
        return False

    return cache_path.stat().st_mtime >= source_path.stat().st_mtime


def build_one_display_cache(npy_path, cache_path, fs, max_points=350000):
    npy_path = Path(npy_path)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    t, y, mode = make_display_trace_from_npy(
        npy_path=npy_path,
        fs=fs,
        max_points=max_points,
    )

    np.savez_compressed(
        cache_path,
        t=t,
        y=y,
        mode=np.array(mode),
        source_mtime=np.array(npy_path.stat().st_mtime),
        fs=np.array(float(fs)),
        max_points=np.array(int(max_points)),
    )

    return t, y, mode


@st.cache_data(show_spinner=False)
def load_display_cache_npz(cache_path_str, cache_mtime):
    z = np.load(cache_path_str, allow_pickle=True)
    t = z["t"]
    y = z["y"]
    mode = str(z["mode"])
    return t, y, mode


def load_or_build_display_cache(npy_path, cache_path, fs, max_points=350000):
    npy_path = Path(npy_path)
    cache_path = Path(cache_path)

    if cache_is_valid(npy_path, cache_path):
        return load_display_cache_npz(str(cache_path), cache_path.stat().st_mtime)

    return build_one_display_cache(
        npy_path=npy_path,
        cache_path=cache_path,
        fs=fs,
        max_points=max_points,
    )


def build_recording_plot_cache(recording_dir, max_points=350000):
    recording_dir = Path(recording_dir)
    metadata = read_json_fast(recording_dir / "metadata.json")
    fs = float(metadata["sampling_rate_hz"])

    cache_dir = recording_dir / "plot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    eeg_cache = cache_dir / "eeg_display.npz"
    emg_cache = cache_dir / "emg_display.npz"

    build_one_display_cache(
        recording_dir / "eeg.npy",
        eeg_cache,
        fs=fs,
        max_points=max_points,
    )

    build_one_display_cache(
        recording_dir / "emg.npy",
        emg_cache,
        fs=fs,
        max_points=max_points,
    )

    return eeg_cache, emg_cache



def make_clickable_scoring_strip(recording_dir, start_min, window_min):
    recording_dir = Path(recording_dir)

    layer1_file = recording_dir / "layer1_wake_sleep.csv"

    if not layer1_file.exists():
        raise FileNotFoundError(layer1_file)

    layer1 = read_csv_fast(layer1_file) if "read_csv_fast" in globals() else pd.read_csv(layer1_file)
    layer1["time_min"] = layer1["t0_s"] / 60

    start_s = float(start_min) * 60
    end_s = float(start_min + window_min) * 60

    layer1_sub = layer1[
        (layer1["t1_s"] >= start_s)
        & (layer1["t0_s"] <= end_s)
    ].copy()

    rows = []
    names = []
    hover_labels = []

    manual_file = recording_dir / "manual_scoring_aligned.csv"

    if manual_file.exists():
        manual = read_csv_fast(manual_file) if "read_csv_fast" in globals() else pd.read_csv(manual_file)
        manual_labels = labels_at_epoch_midpoints(layer1_sub, manual, "manual_state")
        rows.append(state_codes(manual_labels))
        names.append("Manual")
        hover_labels.append(manual_labels)

    l1_labels = layer1_sub["layer1_label"].fillna("Uncertain").astype(str).to_numpy()
    rows.append(state_codes(l1_labels))
    names.append("Layer 1")
    hover_labels.append(l1_labels)

    som_file = recording_dir / "somnotate" / "somnotate_results_timeseries.csv"

    if som_file.exists():
        som = read_csv_fast(som_file) if "read_csv_fast" in globals() else pd.read_csv(som_file)
        som_labels = labels_at_epoch_midpoints(layer1_sub, som, "somnotate_state")
        rows.append(state_codes(som_labels))
        names.append("Somnotate")
        hover_labels.append(som_labels)

    final_file = recording_dir / "final_scoring.csv"

    if final_file.exists():
        final = read_csv_fast(final_file) if "read_csv_fast" in globals() else pd.read_csv(final_file)
        final_labels = labels_at_epoch_midpoints(layer1_sub, final, "final_state")
        rows.append(state_codes(final_labels))
        names.append("Final")
        hover_labels.append(final_labels)

    z = np.vstack(rows)
    custom = np.vstack(hover_labels)

    color_wake = "#1f77b4"
    color_layer1_sleep = "#f7c6d9"  # light pink for Layer 1 Sleep
    color_nrem_sleep = "#ff7f0e"
    color_rem = "#2ca02c"
    color_uncertain = "#9e9e9e"

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=layer1_sub["time_min"],
            y=names,
            z=z,
            zmin=0,
            zmax=5,
            colorscale=[
                [0.00, color_wake], [0.16, color_wake],
                [0.17, color_layer1_sleep], [0.33, color_layer1_sleep],
                [0.34, color_nrem_sleep], [0.50, color_nrem_sleep],
                [0.51, color_rem], [0.67, color_rem],
                [0.68, color_uncertain], [0.84, color_uncertain],
                [0.85, "#000000"], [1.00, "#000000"],
            ],
            showscale=False,
            customdata=custom,
            hovertemplate="time=%{x:.2f} min<br>source=%{y}<br>state=%{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        height=170,
        margin=dict(l=70, r=20, t=30, b=45),
        title="Clickable scoring strip: click a colored bout to select it",
        clickmode="event+select",
        xaxis_title="Time from recording start (min)",
        yaxis_title="",
    )

    fig.update_xaxes(range=[float(start_min), float(start_min + window_min)])

    return fig



def extract_selected_time_range_from_plotly_state(plotly_state):
    if not plotly_state:
        return None

    def get_attr(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    try:
        selection = get_attr(plotly_state, "selection", None)

        if selection is None:
            return None

        xs = []
        points = get_attr(selection, "points", [])

        if points is not None:
            for p in points:
                x = get_attr(p, "x", None)

                if x is None:
                    continue

                try:
                    xs.append(float(x))
                except Exception:
                    pass

        if len(xs) >= 2:
            x0 = float(np.nanmin(xs))
            x1 = float(np.nanmax(xs))

            if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
                return x0, x1

        boxes = get_attr(selection, "box", [])

        if boxes is not None:
            for b in boxes:
                x_range = get_attr(b, "x", None)

                if x_range is not None and len(x_range) >= 2:
                    try:
                        x0 = float(x_range[0])
                        x1 = float(x_range[1])

                        if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
                            return x0, x1
                    except Exception:
                        pass

        return None

    except Exception:
        return None



def compact_bouts_from_epoch_labels(epoch_df, labels, source_name):
    labels = pd.Series(labels).fillna("Undefined").astype(str).to_numpy()

    rows = []
    if len(labels) == 0:
        return pd.DataFrame()

    start_idx = 0
    current = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current:
            t0 = float(epoch_df.iloc[start_idx]["t0_s"])
            t1 = float(epoch_df.iloc[i - 1]["t1_s"])
            rows.append({
                "source": source_name,
                "state": current,
                "start_s": t0,
                "end_s": t1,
                "start_min": t0 / 60,
                "end_min": t1 / 60,
                "duration_s": t1 - t0,
            })
            start_idx = i
            current = labels[i]

    t0 = float(epoch_df.iloc[start_idx]["t0_s"])
    t1 = float(epoch_df.iloc[len(labels) - 1]["t1_s"])
    rows.append({
        "source": source_name,
        "state": current,
        "start_s": t0,
        "end_s": t1,
        "start_min": t0 / 60,
        "end_min": t1 / 60,
        "duration_s": t1 - t0,
    })

    out = pd.DataFrame(rows)
    if len(out):
        out["bout_id"] = [f"{source_name}_{i:05d}" for i in range(len(out))]
    return out


def get_review_source_bouts(recording_dir, source_name):
    recording_dir = Path(recording_dir)
    layer1_file = recording_dir / "layer1_wake_sleep.csv"

    if not layer1_file.exists():
        return pd.DataFrame()

    layer1 = read_csv_fast(layer1_file) if "read_csv_fast" in globals() else pd.read_csv(layer1_file)
    epoch_df = layer1[["t0_s", "t1_s"]].copy()

    if source_name == "Layer 1":
        labels = layer1["layer1_label"].fillna("Undefined").astype(str).to_numpy()
        return compact_bouts_from_epoch_labels(epoch_df, labels, source_name)

    if source_name == "Manual":
        manual_file = recording_dir / "manual_scoring_aligned.csv"
        if not manual_file.exists():
            return pd.DataFrame()
        manual = read_csv_fast(manual_file) if "read_csv_fast" in globals() else pd.read_csv(manual_file)
        labels = labels_at_epoch_midpoints(epoch_df, manual, "manual_state")
        return compact_bouts_from_epoch_labels(epoch_df, labels, source_name)

    if source_name == "Somnotate":
        som_file = recording_dir / "somnotate" / "somnotate_results_timeseries.csv"
        if not som_file.exists():
            return pd.DataFrame()
        som = read_csv_fast(som_file) if "read_csv_fast" in globals() else pd.read_csv(som_file)
        labels = labels_at_epoch_midpoints(epoch_df, som, "somnotate_state")
        return compact_bouts_from_epoch_labels(epoch_df, labels, source_name)

    return pd.DataFrame()


def find_bout_from_clicked_scoring(recording_dir, source_name, clicked_min):
    bouts = get_review_source_bouts(recording_dir, source_name)

    if bouts is None or len(bouts) == 0:
        return None

    clicked_s = float(clicked_min) * 60
    hit = bouts[
        (bouts["start_s"].astype(float) <= clicked_s)
        & (bouts["end_s"].astype(float) > clicked_s)
    ].copy()

    if len(hit) == 0:
        return None

    return hit.iloc[0].to_dict()


def mode_for_scoring_source(source_name):
    if source_name == "Somnotate":
        return "use_somnotate", ""
    if source_name == "Manual":
        return "use_manual", ""
    if source_name == "Layer 1":
        return "use_layer1", ""
    return "manual_label", ""


def approve_interval_from_app(project_root, recording_id, final_file, start_s, end_s, mode, label="", notes=""):
    if not Path(final_file).exists():
        init_cmd = [
            sys.executable,
            "pipelines/20_review_layer.py",
            "init-final",
            "--project-root", str(project_root),
            "--recording-id", str(recording_id),
        ]
        run_command(init_cmd)

    cmd = [
        sys.executable,
        "pipelines/20_review_layer.py",
        "apply-edit",
        "--project-root", str(project_root),
        "--recording-id", str(recording_id),
        "--start-s", str(float(start_s)),
        "--end-s", str(float(end_s)),
        "--mode", str(mode),
        "--label", str(label),
        "--notes", str(notes),
    ]

    return run_command(cmd)


def parse_review_plot_selection(selection_state):
    """
    Distinguish scoring-row selection from manual time selection.

    Returns:
      {"kind": "scoring_bout", "source": ..., "time_min": ...}
      {"kind": "manual_range", "start_min": ..., "end_min": ...}
      None
    """
    if not selection_state:
        return None

    try:
        selection = getattr(selection_state, "selection", None)
        if selection is None and isinstance(selection_state, dict):
            selection = selection_state.get("selection", None)

        if selection is None:
            return None

        points = getattr(selection, "points", None)
        if points is None and isinstance(selection, dict):
            points = selection.get("points", [])

        if not points:
            return None

        xs = []
        scoring_hits = []

        for p in points:
            if isinstance(p, dict):
                x = p.get("x", None)
                y = p.get("y", None)
                customdata = p.get("customdata", None)
            else:
                x = getattr(p, "x", None)
                y = getattr(p, "y", None)
                customdata = getattr(p, "customdata", None)

            if x is not None:
                try:
                    xs.append(float(x))
                except Exception:
                    pass

            # Hidden helper points in scoring row store source in customdata.
            if customdata in ["Manual", "Layer 1", "Somnotate"]:
                try:
                    scoring_hits.append((str(customdata), float(x)))
                except Exception:
                    pass

            # Sometimes heatmap itself gives y as the row/source.
            if str(y) in ["Manual", "Layer 1", "Somnotate"]:
                try:
                    scoring_hits.append((str(y), float(x)))
                except Exception:
                    pass

        if scoring_hits:
            # Use the middle selected time/source.
            source = scoring_hits[len(scoring_hits) // 2][0]
            times = [t for s, t in scoring_hits if s == source]
            if not times:
                times = [t for _, t in scoring_hits]
            return {
                "kind": "scoring_bout",
                "source": source,
                "time_min": float(np.nanmedian(times)),
            }

        if len(xs) >= 2:
            x0 = float(np.nanmin(xs))
            x1 = float(np.nanmax(xs))
            if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
                return {
                    "kind": "manual_range",
                    "start_min": x0,
                    "end_min": x1,
                }

        return None

    except Exception:
        return None


# =============================================================================
# FAST FINAL-SCORING EDITS
# =============================================================================

FINAL_STATE_TO_CODE = {
    "Wake": 0,
    "NREM": 1,
    "REM": 2,
    "Sleep": 1,
    "Uncertain": -1,
    "Undefined": -1,
    "Artifact": -2,
}


def ensure_final_scoring_fast(recording_dir, recording_id):
    """
    Create final_scoring.csv quickly if it does not exist.

    This avoids calling the slower external review script just to make
    a file before a manual edit.
    """
    recording_dir = Path(recording_dir)
    final_file = recording_dir / "final_scoring.csv"

    if final_file.exists():
        return final_file

    layer1_file = recording_dir / "layer1_wake_sleep.csv"

    if not layer1_file.exists():
        raise FileNotFoundError(layer1_file)

    layer1 = pd.read_csv(layer1_file)

    out = pd.DataFrame()
    out["recording_id"] = recording_id
    out["epoch_id"] = np.arange(len(layer1))
    out["t0_s"] = layer1["t0_s"].astype(float)
    out["t1_s"] = layer1["t1_s"].astype(float)

    if "layer1_label" in layer1.columns:
        init_state = []
        for x in layer1["layer1_label"].fillna("Undefined").astype(str):
            if x == "Wake":
                init_state.append("Wake")
            elif x == "Sleep":
                init_state.append("NREM")
            else:
                init_state.append("Undefined")
    else:
        init_state = ["Undefined"] * len(layer1)

    out["final_state"] = init_state
    out["final_code"] = [FINAL_STATE_TO_CODE.get(x, -1) for x in init_state]
    out["final_source"] = "initial_layer1_fast"
    out["review_status"] = "not_reviewed"
    out["review_notes"] = ""

    out.to_csv(final_file, index=False)

    return final_file


def apply_manual_label_fast(recording_dir, recording_id, start_s, end_s, label, notes=""):
    """
    Fast in-app manual label application.

    This only edits final_scoring.csv and avoids launching a subprocess.
    """
    recording_dir = Path(recording_dir)
    final_file = ensure_final_scoring_fast(recording_dir, recording_id)

    final = pd.read_csv(final_file)

    start_s = float(start_s)
    end_s = float(end_s)
    label = str(label)

    mask = (
        (final["t0_s"].astype(float) < end_s)
        & (final["t1_s"].astype(float) > start_s)
    )

    n_epochs = int(mask.sum())

    if n_epochs == 0:
        return False, "No epochs found in selected interval."

    record_review_undo_snapshot(
        recording_dir=recording_dir,
        recording_id=recording_id,
        final_df=final,
        mask=mask,
        action_label=f"manual label {label}",
    )

    final.loc[mask, "final_state"] = label
    final.loc[mask, "final_code"] = FINAL_STATE_TO_CODE.get(label, -1)
    final.loc[mask, "final_source"] = "manual_fast_edit"
    final.loc[mask, "review_status"] = "reviewed"
    final.loc[mask, "review_notes"] = str(notes)

    final.to_csv(final_file, index=False)

    log_file = recording_dir / "review_edit_log.csv"

    log_row = pd.DataFrame([{
        "recording_id": recording_id,
        "start_s": start_s,
        "end_s": end_s,
        "mode": "manual_label_fast",
        "label": label,
        "source": "manual_fast_edit",
        "notes": str(notes),
        "n_epochs": n_epochs,
        "saved_at": pd.Timestamp.now().isoformat(),
    }])

    if log_file.exists():
        log = pd.read_csv(log_file)
        log = pd.concat([log, log_row], ignore_index=True)
    else:
        log = log_row

    log.to_csv(log_file, index=False)

    return True, f"Saved {label} for {n_epochs} epochs."



def apply_source_label_fast(recording_dir, recording_id, start_s, end_s, source_name, notes=""):
    """
    Fast in-app approval of an existing scoring source for a selected interval.

    source_name:
      - "Manual"
      - "Somnotate"
      - "Layer 1"
    """
    recording_dir = Path(recording_dir)
    final_file = ensure_final_scoring_fast(recording_dir, recording_id)

    final = pd.read_csv(final_file)

    start_s = float(start_s)
    end_s = float(end_s)

    mask = (
        (final["t0_s"].astype(float) < end_s)
        & (final["t1_s"].astype(float) > start_s)
    )

    if int(mask.sum()) == 0:
        return False, "No epochs found in selected interval."

    epoch_df = final[["t0_s", "t1_s"]].copy()

    if source_name == "Manual":
        source_file = recording_dir / "manual_scoring_aligned.csv"
        label_col = "manual_state"

        if not source_file.exists():
            return False, "Manual scoring file not found."

        source_df = pd.read_csv(source_file)
        source_labels = labels_at_epoch_midpoints(epoch_df, source_df, label_col)

        final_source = "accepted_manual_fast"

    elif source_name == "Somnotate":
        source_file = recording_dir / "somnotate" / "somnotate_results_timeseries.csv"
        label_col = "somnotate_state"

        if not source_file.exists():
            return False, "Somnotate results file not found."

        source_df = pd.read_csv(source_file)
        source_labels = labels_at_epoch_midpoints(epoch_df, source_df, label_col)

        final_source = "accepted_somnotate_fast"

    elif source_name == "Layer 1":
        source_file = recording_dir / "layer1_wake_sleep.csv"
        label_col = "layer1_label"

        if not source_file.exists():
            return False, "Layer 1 scoring file not found."

        source_df = pd.read_csv(source_file)
        source_labels = labels_at_epoch_midpoints(epoch_df, source_df, label_col)

        converted = []

        for x in source_labels:
            x = str(x)

            if x == "Wake":
                converted.append("Wake")
            elif x == "Sleep":
                converted.append("NREM")
            else:
                converted.append("Undefined")

        source_labels = np.array(converted, dtype=object)
        final_source = "accepted_layer1_fast"

    else:
        return False, f"Unknown source: {source_name}"

    record_review_undo_snapshot(
        recording_dir=recording_dir,
        recording_id=recording_id,
        final_df=final,
        mask=mask,
        action_label=f"accept {source_name}",
    )

    selected_labels = np.asarray(source_labels, dtype=object)[mask.to_numpy()]

    final.loc[mask, "final_state"] = selected_labels
    final.loc[mask, "final_code"] = [FINAL_STATE_TO_CODE.get(str(x), -1) for x in selected_labels]
    final.loc[mask, "final_source"] = final_source
    final.loc[mask, "review_status"] = "reviewed"
    final.loc[mask, "review_notes"] = str(notes)

    final.to_csv(final_file, index=False)

    log_file = recording_dir / "review_edit_log.csv"

    log_row = pd.DataFrame([{
        "recording_id": recording_id,
        "start_s": start_s,
        "end_s": end_s,
        "mode": "source_approval_fast",
        "label": source_name,
        "source": final_source,
        "notes": str(notes),
        "n_epochs": int(mask.sum()),
        "saved_at": pd.Timestamp.now().isoformat(),
    }])

    if log_file.exists():
        log = pd.read_csv(log_file)
        log = pd.concat([log, log_row], ignore_index=True)
    else:
        log = log_row

    log.to_csv(log_file, index=False)

    return True, f"Accepted {source_name} scoring for {int(mask.sum())} epochs."



# =============================================================================
# LIGHTWEIGHT REVIEW SHADING
# =============================================================================

REVIEW_SHADE_COLORS = {
    "Wake": "rgba(31, 119, 180, 0.16)",
    "NREM": "rgba(255, 127, 14, 0.16)",
    "Sleep": "rgba(255, 127, 14, 0.16)",
    "REM": "rgba(44, 160, 44, 0.16)",
    "Uncertain": "rgba(150, 150, 150, 0.16)",
    "Undefined": "rgba(150, 150, 150, 0.16)",
    "Artifact": "rgba(0, 0, 0, 0.14)",
}


def remember_review_shade(recording_id, start_min, end_min, label, source="manual"):
    """
    Store a small visual overlay in session_state so the user gets immediate
    feedback without forcing a full cache refresh.
    """
    key = f"review_recent_shades_{recording_id}"

    if key not in st.session_state:
        st.session_state[key] = []

    shade = {
        "start_min": float(start_min),
        "end_min": float(end_min),
        "label": str(label),
        "source": str(source),
    }

    st.session_state[key].append(shade)

    # Keep only recent edits to avoid accumulating too many shapes in the plot.
    st.session_state[key] = st.session_state[key][-200:]


def add_recent_review_shades_to_fig(fig, recording_id):
    """
    Draw recent in-session edits as colored shades on EEG/EMG/RMS/probability panels.
    This gives immediate feedback after pressing 0/1/2 or accepting a source.
    """
    key = f"review_recent_shades_{recording_id}"
    shades = st.session_state.get(key, [])

    for shade in shades:
        x0 = float(shade["start_min"])
        x1 = float(shade["end_min"])
        label = str(shade["label"])

        if x1 <= x0:
            continue

        fill = REVIEW_SHADE_COLORS.get(label, "rgba(150,150,150,0.12)")

        # Shade raw EEG, raw EMG, EMG RMS, and probability panels.
        # Do not shade the scoring-row panel.
        for rr in [2, 3, 4, 5]:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=fill,
                line_width=0,
                row=rr,
                col=1,
            )

    return fig


def dominant_state_for_source_interval(recording_dir, source_name, start_s, end_s):
    """
    Lightweight dominant state estimate for visual feedback after accepting
    Manual / Somnotate / Layer 1 over a selected interval.
    """
    recording_dir = Path(recording_dir)

    try:
        layer1_file = recording_dir / "layer1_wake_sleep.csv"
        layer1 = read_csv_fast(layer1_file) if "read_csv_fast" in globals() else pd.read_csv(layer1_file)
        epoch_df = layer1[["t0_s", "t1_s"]].copy()

        mask = (
            (epoch_df["t0_s"].astype(float) < float(end_s))
            & (epoch_df["t1_s"].astype(float) > float(start_s))
        )

        if source_name == "Layer 1":
            labels = layer1["layer1_label"].fillna("Undefined").astype(str).to_numpy()
            converted = []
            for x in labels:
                if x == "Wake":
                    converted.append("Wake")
                elif x == "Sleep":
                    converted.append("NREM")
                else:
                    converted.append("Undefined")
            labels = np.array(converted, dtype=object)

        elif source_name == "Manual":
            source_file = recording_dir / "manual_scoring_aligned.csv"
            if not source_file.exists():
                return "Uncertain"
            source_df = read_csv_fast(source_file) if "read_csv_fast" in globals() else pd.read_csv(source_file)
            labels = labels_at_epoch_midpoints(epoch_df, source_df, "manual_state")

        elif source_name == "Somnotate":
            source_file = recording_dir / "somnotate" / "somnotate_results_timeseries.csv"
            if not source_file.exists():
                return "Uncertain"
            source_df = read_csv_fast(source_file) if "read_csv_fast" in globals() else pd.read_csv(source_file)
            labels = labels_at_epoch_midpoints(epoch_df, source_df, "somnotate_state")

        else:
            return "Uncertain"

        selected = pd.Series(labels[mask.to_numpy()]).astype(str)

        if len(selected) == 0:
            return "Uncertain"

        return selected.value_counts().idxmax()

    except Exception:
        return "Uncertain"


def undo_last_pending_review_edit(recording_dir, recording_id):
    """
    Undo the last pending queued edit.

    This is intentionally lightweight:
    - removes last pending edit from session_state
    - marks/removes it from review_edit_queue.csv
    - removes the last immediate visual shade

    It does not undo already committed final_scoring.csv edits.
    """
    recording_dir = Path(recording_dir)

    key = review_queue_key(recording_id)

    if key not in st.session_state:
        st.session_state[key] = []

    if len(st.session_state[key]) == 0:
        return False, "No pending edit to undo."

    last_edit = st.session_state[key].pop()
    last_edit_id = str(last_edit.get("edit_id", ""))

    # Remove matching row from disk queue if it exists.
    queue_file = recording_dir / "review_edit_queue.csv"

    if queue_file.exists() and last_edit_id:
        try:
            q = pd.read_csv(queue_file)

            if "edit_id" in q.columns:
                q = q[q["edit_id"].astype(str) != last_edit_id].copy()
                q.to_csv(queue_file, index=False)
        except Exception:
            pass

    # Remove last recent visual shade.
    shade_key = f"review_recent_shades_{recording_id}"

    if shade_key in st.session_state and len(st.session_state[shade_key]) > 0:
        st.session_state[shade_key].pop()

    label = str(last_edit.get("label", last_edit.get("source_name", "")))
    start_min = float(last_edit.get("start_min", 0.0))
    end_min = float(last_edit.get("end_min", 0.0))

    return True, f"Undid pending edit: {label} | {start_min:.2f}–{end_min:.2f} min."


# =============================================================================
# REVIEW UNDO SYSTEM
# =============================================================================

def review_undo_dir(recording_dir):
    d = Path(recording_dir) / "review_undo_stack"
    d.mkdir(parents=True, exist_ok=True)
    return d


def record_review_undo_snapshot(recording_dir, recording_id, final_df, mask, action_label="review_edit"):
    """
    Save previous final_scoring rows before an edit.
    This allows Undo to restore the exact previous labels.
    """
    recording_dir = Path(recording_dir)
    undo_dir = review_undo_dir(recording_dir)

    previous = final_df.loc[mask].copy()

    if len(previous) == 0:
        return None

    undo_id = pd.Timestamp.now().strftime("undo_%Y%m%d_%H%M%S_%f")
    snapshot_file = undo_dir / f"{undo_id}.csv"

    previous.to_csv(snapshot_file, index=False)

    registry_file = undo_dir / "undo_registry.csv"

    row = pd.DataFrame([{
        "undo_id": undo_id,
        "recording_id": recording_id,
        "snapshot_file": str(snapshot_file),
        "action_label": str(action_label),
        "start_s": float(previous["t0_s"].min()),
        "end_s": float(previous["t1_s"].max()),
        "n_epochs": int(len(previous)),
        "status": "active",
        "created_at": pd.Timestamp.now().isoformat(),
    }])

    if registry_file.exists():
        reg = pd.read_csv(registry_file)
        reg = pd.concat([reg, row], ignore_index=True)
    else:
        reg = row

    reg.to_csv(registry_file, index=False)

    return undo_id


def undo_last_review_action(recording_dir, recording_id):
    """
    Undo the last active edit.

    Restores the previous rows saved in review_undo_stack.
    Works for direct edits already written to final_scoring.csv.
    """
    recording_dir = Path(recording_dir)
    undo_dir = review_undo_dir(recording_dir)
    registry_file = undo_dir / "undo_registry.csv"
    final_file = recording_dir / "final_scoring.csv"

    if not registry_file.exists():
        return False, "No undo history yet."

    if not final_file.exists():
        return False, "No final_scoring.csv found."

    reg = pd.read_csv(registry_file)

    if "status" not in reg.columns:
        return False, "Undo registry is missing status column."

    active = reg[reg["status"].astype(str) == "active"].copy()

    if len(active) == 0:
        return False, "No action left to undo."

    last_idx = active.index[-1]
    last = reg.loc[last_idx]

    snapshot_file = Path(str(last["snapshot_file"]))

    if not snapshot_file.exists():
        reg.loc[last_idx, "status"] = "missing_snapshot"
        reg.to_csv(registry_file, index=False)
        return False, "Undo snapshot file is missing."

    previous = pd.read_csv(snapshot_file)
    final = pd.read_csv(final_file)

    restored = 0

    # Only restore columns that exist in both tables.
    # Do not try to read epoch_id from prev_row after set_index().
    common_cols = [c for c in previous.columns if c in final.columns]

    if "epoch_id" in final.columns and "epoch_id" in previous.columns:
        prev_by_epoch = previous.set_index("epoch_id", drop=False)

        for epoch_id, prev_row in prev_by_epoch.iterrows():
            hit = final["epoch_id"].astype(str) == str(epoch_id)

            if hit.any():
                for col in common_cols:
                    final.loc[hit, col] = prev_row[col]

                restored += int(hit.sum())

    else:
        for _, prev_row in previous.iterrows():
            hit = (
                (final["t0_s"].astype(float) == float(prev_row["t0_s"]))
                & (final["t1_s"].astype(float) == float(prev_row["t1_s"]))
            )

            if hit.any():
                for col in common_cols:
                    final.loc[hit, col] = prev_row[col]

                restored += int(hit.sum())

    if restored == 0:
        return False, "Undo snapshot was found, but no matching epochs were restored."

    final.to_csv(final_file, index=False)

    reg.loc[last_idx, "status"] = "undone"
    reg.loc[last_idx, "undone_at"] = pd.Timestamp.now().isoformat()
    reg.to_csv(registry_file, index=False)

    # Remove the most recent immediate visual shade, if present.
    shade_key = f"review_recent_shades_{recording_id}"

    if shade_key in st.session_state and len(st.session_state[shade_key]) > 0:
        st.session_state[shade_key].pop()

    return True, f"Undid last action: restored {restored} epochs."


# =============================================================================
# CACHED REVIEW QC FIGURE
# =============================================================================

def file_mtime_or_zero(path):
    path = Path(path)
    if path.exists():
        return float(path.stat().st_mtime)
    return 0.0


@st.cache_data(show_spinner=False)
def make_cached_review_base_qc_plot(
    recording_dir_str,
    start_min,
    window_min,
    metadata_mtime,
    eeg_mtime,
    emg_mtime,
    layer1_mtime,
    features_mtime,
    manual_mtime,
    som_mtime,
):
    """
    Cached base QC plot.

    Important:
    final_scoring.csv is intentionally not part of the cache key.
    New edits are shown through lightweight session-state shades instead of
    forcing the whole base plot to rebuild after every scoring decision.
    """
    return make_qc_plot(
        recording_dir=Path(recording_dir_str),
        start_min=float(start_min),
        window_min=float(window_min),
    )


def get_cached_review_base_qc_plot(recording_dir, start_min, window_min):
    recording_dir = Path(recording_dir)

    som_file = recording_dir / "somnotate" / "somnotate_results_timeseries.csv"

    fig = make_cached_review_base_qc_plot(
        str(recording_dir),
        float(start_min),
        float(window_min),
        file_mtime_or_zero(recording_dir / "metadata.json"),
        file_mtime_or_zero(recording_dir / "eeg.npy"),
        file_mtime_or_zero(recording_dir / "emg.npy"),
        file_mtime_or_zero(recording_dir / "layer1_wake_sleep.csv"),
        file_mtime_or_zero(recording_dir / "epoch_features.csv"),
        file_mtime_or_zero(recording_dir / "manual_scoring_aligned.csv"),
        file_mtime_or_zero(som_file),
    )

    # Deep-copy so every rerun can safely add temporary shades/helpers
    # without modifying the cached object.
    return copy.deepcopy(fig)


# =============================================================================
# EXPORT FINAL / MIXED SCORING
# =============================================================================

EXPORT_STATE_TO_CODE = {
    "Wake": 0,
    "NREM": 1,
    "REM": 2,
    "Sleep": 1,
    "Uncertain": -1,
    "Undefined": -1,
    "Artifact": -2,
}


def normalize_export_state(x):
    x = str(x).strip()

    mapping = {
        "Awake": "Wake",
        "Wake": "Wake",
        "W": "Wake",
        "WK": "Wake",
        "NREM": "NREM",
        "SWS": "NREM",
        "Sleep": "NREM",
        "REM": "REM",
        "PS": "REM",
        "Uncertain": "Uncertain",
        "Undefined": "Undefined",
        "nan": "Undefined",
        "NaN": "Undefined",
        "": "Undefined",
        "-1": "Undefined",
        "Artifact": "Artifact",
    }

    return mapping.get(x, x)


def ensure_final_scoring_for_export(recording_dir, recording_id):
    """
    Make sure final_scoring.csv exists.

    If it already exists, keep it.
    If it does not exist, create a reasonable default:
      manual > somnotate > layer1
    """
    recording_dir = Path(recording_dir)
    final_file = recording_dir / "final_scoring.csv"

    if final_file.exists():
        return final_file

    layer1_file = recording_dir / "layer1_wake_sleep.csv"

    if not layer1_file.exists():
        raise FileNotFoundError(layer1_file)

    layer1 = pd.read_csv(layer1_file)
    epoch_df = layer1[["t0_s", "t1_s"]].copy()

    manual_file = recording_dir / "manual_scoring_aligned.csv"
    som_file = recording_dir / "somnotate" / "somnotate_results_timeseries.csv"

    if manual_file.exists():
        manual = pd.read_csv(manual_file)
        init_state = labels_at_epoch_midpoints(epoch_df, manual, "manual_state")
        init_source = "initial_manual"

    elif som_file.exists():
        som = pd.read_csv(som_file)
        init_state = labels_at_epoch_midpoints(epoch_df, som, "somnotate_state")
        init_source = "initial_somnotate"

    else:
        raw = layer1["layer1_label"].fillna("Undefined").astype(str).to_numpy()
        init_state = []
        for x in raw:
            if x == "Wake":
                init_state.append("Wake")
            elif x == "Sleep":
                init_state.append("NREM")
            else:
                init_state.append("Undefined")
        init_state = np.array(init_state, dtype=object)
        init_source = "initial_layer1"

    init_state = np.array([normalize_export_state(x) for x in init_state], dtype=object)

    out = pd.DataFrame()
    out["recording_id"] = recording_id
    out["epoch_id"] = np.arange(len(epoch_df))
    out["t0_s"] = epoch_df["t0_s"].astype(float)
    out["t1_s"] = epoch_df["t1_s"].astype(float)
    out["final_state"] = init_state
    out["final_code"] = [EXPORT_STATE_TO_CODE.get(str(x), -1) for x in init_state]
    out["final_source"] = init_source
    out["review_status"] = "not_reviewed"
    out["review_notes"] = ""

    out.to_csv(final_file, index=False)

    return final_file


def reviewed_mask_from_final(final_df):
    """
    Define which epochs were actively scored/reviewed inside the app.
    """
    if "review_status" in final_df.columns:
        mask = final_df["review_status"].astype(str).str.lower().eq("reviewed")
    else:
        mask = pd.Series(False, index=final_df.index)

    # Safety: also treat known app-edit sources as reviewed.
    if "final_source" in final_df.columns:
        src = final_df["final_source"].astype(str).str.lower()
        source_mask = (
            src.str.contains("manual_fast")
            | src.str.contains("manual_edit")
            | src.str.contains("accepted")
            | src.str.contains("queued")
            | src.str.contains("review")
        )
        mask = mask | source_mask

    return mask.to_numpy(dtype=bool)


def export_scoring_from_app(recording_dir, recording_id, mode):
    """
    Export scoring as CSV and MAT.

    mode:
      - "app": export final_scoring.csv as produced by the app
      - "mixed_manual_app": use app-reviewed periods, keep original manual elsewhere
    """
    recording_dir = Path(recording_dir)
    export_dir = recording_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    final_file = ensure_final_scoring_for_export(recording_dir, recording_id)
    final = pd.read_csv(final_file)

    if not all(c in final.columns for c in ["t0_s", "t1_s", "final_state"]):
        return False, "final_scoring.csv is missing required columns.", None, None

    epoch_df = final[["t0_s", "t1_s"]].copy()

    if mode == "app":
        export_state = final["final_state"].map(normalize_export_state).astype(str).to_numpy()
        export_source = final.get("final_source", pd.Series(["app"] * len(final))).astype(str).to_numpy()
        mode_label = "app_scoring"

    elif mode == "mixed_manual_app":
        manual_file = recording_dir / "manual_scoring_aligned.csv"

        if not manual_file.exists():
            return False, "Manual scoring was not provided, so mixed export is not available.", None, None

        manual = pd.read_csv(manual_file)
        manual_labels = labels_at_epoch_midpoints(epoch_df, manual, "manual_state")
        manual_labels = np.array([normalize_export_state(x) for x in manual_labels], dtype=object)

        app_labels = final["final_state"].map(normalize_export_state).astype(str).to_numpy()
        app_reviewed = reviewed_mask_from_final(final)

        export_state = manual_labels.copy()
        export_state[app_reviewed] = app_labels[app_reviewed]

        export_source = np.array(["original_manual"] * len(final), dtype=object)
        export_source[app_reviewed] = "app_reviewed"

        mode_label = "mixed_manual_plus_app"

    else:
        return False, f"Unknown export mode: {mode}", None, None

    export_code = np.array([EXPORT_STATE_TO_CODE.get(str(x), -1) for x in export_state], dtype=np.int16)

    out = pd.DataFrame()
    out["recording_id"] = recording_id
    if "epoch_id" in final.columns:
        out["epoch_id"] = final["epoch_id"]
    else:
        out["epoch_id"] = np.arange(len(final))
    out["t0_s"] = final["t0_s"].astype(float)
    out["t1_s"] = final["t1_s"].astype(float)
    out["state"] = export_state
    out["code"] = export_code
    out["source"] = export_source

    csv_out = export_dir / f"{recording_id}_{mode_label}.csv"
    mat_out = export_dir / f"{recording_id}_{mode_label}.mat"

    out.to_csv(csv_out, index=False)

    savemat(mat_out, {
        "scoring": export_code,
        "scoring_state": np.asarray(export_state, dtype=object),
        "scoring_source": np.asarray(export_source, dtype=object),
        "t0_s": out["t0_s"].to_numpy(dtype=float),
        "t1_s": out["t1_s"].to_numpy(dtype=float),
        "epoch_id": out["epoch_id"].to_numpy(),
        "state_code_map": {
            "Wake": 0,
            "NREM": 1,
            "REM": 2,
            "Uncertain": -1,
            "Artifact": -2,
        },
        "export_mode": mode_label,
    })

    return True, f"Exported {mode_label}.", csv_out, mat_out


# =============================================================================
# RAW WINDOW TRACE FOR MAIN QC PLOT
# =============================================================================

def make_raw_window_trace_from_npy(npy_path, fs, start_min, end_min, max_points=200000):
    """
    Load only the current visible window from the raw .npy signal.

    This keeps EEG/EMG quality high while allowing the user to move through
    the recording with a fixed-size review window.
    """
    npy_path = Path(npy_path)
    x = np.load(str(npy_path), mmap_mode="r")

    fs = float(fs)
    n_total = len(x)

    start_s = max(0.0, float(start_min) * 60.0)
    end_s = max(start_s + 1.0, float(end_min) * 60.0)

    i0 = int(np.floor(start_s * fs))
    i1 = int(np.ceil(end_s * fs))

    i0 = max(0, min(i0, n_total - 1))
    i1 = max(i0 + 1, min(i1, n_total))

    xw = np.asarray(x[i0:i1], dtype=np.float32)
    n = len(xw)

    if n <= int(max_points):
        idx = np.arange(n, dtype=np.int64)
        mode = "raw-window"
    else:
        step = int(np.ceil(n / int(max_points)))
        idx = np.arange(0, n, step, dtype=np.int64)
        mode = f"raw-window-decimated-{step}x"

    y = xw[idx].astype(np.float32)
    t = (i0 + idx).astype(np.float64) / fs / 60.0

    return t.astype(np.float32), y, mode


# =============================================================================
# WINDOWED RAW EEG/EMG DISPLAY
# =============================================================================

def file_mtime_or_zero_local(path):
    path = Path(path)
    return float(path.stat().st_mtime) if path.exists() else 0.0


@st.cache_data(show_spinner=False, max_entries=256)
def make_raw_window_trace_cached(
    npy_path_str,
    fs,
    start_min,
    end_min,
    max_points,
    source_mtime,
):
    """
    Load only the currently visible window from the raw .npy signal.

    This keeps the main QC plot good-quality while still allowing the user
    to move through the whole recording with a window slider.
    """
    npy_path = Path(npy_path_str)
    x = np.load(str(npy_path), mmap_mode="r")

    fs = float(fs)
    n_total = len(x)

    start_min = round(float(start_min), 4)
    end_min = round(float(end_min), 4)

    start_s = max(0.0, start_min * 60.0)
    end_s = max(start_s + 1.0, end_min * 60.0)

    i0 = int(np.floor(start_s * fs))
    i1 = int(np.ceil(end_s * fs))

    i0 = max(0, min(i0, n_total - 1))
    i1 = max(i0 + 1, min(i1, n_total))

    xw = np.asarray(x[i0:i1], dtype=np.float32)
    n = len(xw)

    if n <= int(max_points):
        idx = np.arange(n, dtype=np.int64)
        mode = "raw-window"
    else:
        step = int(np.ceil(n / int(max_points)))
        idx = np.arange(0, n, step, dtype=np.int64)
        mode = f"raw-window-decimated-{step}x"

    y = xw[idx].astype(np.float32)
    t = (i0 + idx).astype(np.float64) / fs / 60.0

    return t.astype(np.float32), y, mode


def set_review_window_around_event(ev, duration_min_review, fixed_window_min=15.0):
    """
    Center a dissociation event inside a fixed-size review window.
    The selected period remains the event itself.
    """
    event_start = float(ev["start_min"])
    event_end = float(ev["end_min"])

    event_start = max(0.0, min(event_start, float(duration_min_review)))
    event_end = max(event_start + 0.01, min(event_end, float(duration_min_review)))

    visible_window = min(float(fixed_window_min), float(duration_min_review))
    event_mid = (event_start + event_end) / 2.0

    visible_start = event_mid - visible_window / 2.0
    visible_start = max(
        0.0,
        min(visible_start, max(0.0, float(duration_min_review) - visible_window)),
    )

    st.session_state.review_manual_start_min = float(visible_start)
    st.session_state.review_manual_window_min = float(visible_window)

    st.session_state.selected_label_start_min = float(event_start)
    st.session_state.selected_label_end_min = float(event_end)

    st.session_state.review_window_slider_revision = (
        st.session_state.get("review_window_slider_revision", 0) + 1
    )
    st.session_state.selected_period_slider_revision = (
        st.session_state.get("selected_period_slider_revision", 0) + 1
    )

# =============================================================================
# APP
# =============================================================================

st.title("Sleep Stage QC v2")
st.caption(
    "Clean workflow: load preprocessed .mat → run fast Wake/Sleep Layer 1 → inspect EEG/EMG and scoring."
)

tab_import, tab_qc, tab_somnotate, tab_review, tab_stats, tab_about = st.tabs([
    "1. Import .mat + Layer 1",
    "2. QC viewer",
    "3. Somnotate",
    "4. Review / Edit scoring",
    "5. Model statistics / dissociation",
    "About",
])


# =============================================================================
# TAB 1 — IMPORT .MAT + LAYER 1
# =============================================================================

with tab_import:
    st.subheader("Import preprocessed .mat recording")

    default_project = str(Path.home() / "Desktop" / "SleepStageQC_v2_Project")

    project_root = st.text_input(
        "Project root",
        default_project,
        key="import_project_root",
    )

    mat_file = st.text_input(
        ".mat file",
        "",
        help="Preprocessed lab file containing EEG, EMG, and optionally scoring.",
    )

    if mat_file:
        keys = safe_mat_keys(mat_file)

        if keys:
            st.success("Detected .mat variables")
            st.write(keys)
        else:
            st.warning("Could not read variables yet. Check that the file path is correct.")

    c1, c2, c3 = st.columns(3)

    with c1:
        recording_id = st.text_input("Recording ID", "test_recording")
        mouse_id = st.text_input("Mouse ID", "")

    with c2:
        group = st.text_input("Group", "")
        condition = st.text_input("Condition", "")

    with c3:
        week = st.text_input("Week", "")
        fs = st.number_input("Sampling rate, Hz", min_value=1.0, value=1017.2526, step=1.0)

    st.markdown("### Variable mapping")

    if mat_file and safe_mat_keys(mat_file):
        keys = safe_mat_keys(mat_file)
        eeg_key = st.selectbox("EEG variable", keys)
        emg_key = st.selectbox("EMG variable", keys)
        scoring_options = [""] + keys
        scoring_key = st.selectbox("Optional scoring variable", scoring_options)
    else:
        eeg_key = st.text_input("EEG variable", "EEG")
        emg_key = st.text_input("EMG variable", "EMG")
        scoring_key = st.text_input("Optional scoring variable", "")

    epoch_sec = st.number_input("Epoch length, seconds", min_value=0.1, value=1.0, step=0.5)

    code_map = st.text_area(
        "Manual scoring code map",
        value='{"0":"Wake","1":"NREM","2":"REM","15":"Wake","-1":"Undefined"}',
        help="Used only if the .mat file contains manual scoring.",
    )

    st.markdown("### Run pipeline")

    run_cols = st.columns(3)

    with run_cols[0]:
        if st.button("1. Import .mat", type="primary"):
            cmd = [
                sys.executable,
                "pipelines/01_import_mat_recording.py",
                "--mat-file", mat_file,
                "--project-root", project_root,
                "--recording-id", recording_id,
                "--mouse-id", mouse_id,
                "--group", group,
                "--condition", condition,
                "--week", week,
                "--eeg-key", eeg_key,
                "--emg-key", emg_key,
                "--fs", str(fs),
                "--epoch-sec", str(epoch_sec),
                "--code-map", code_map,
            ]

            if scoring_key:
                cmd += ["--scoring-key", scoring_key]

            with st.spinner("Importing .mat recording..."):
                result = run_command(cmd)

            if result.returncode == 0:
                st.success("Import complete.")
                st.code(result.stdout)
            else:
                st.error("Import failed.")
                st.code(result.stdout)
                st.code(result.stderr)

    with run_cols[1]:
        if st.button("2. Compute epoch features"):
            cmd = [
                sys.executable,
                "pipelines/02_compute_epoch_features.py",
                "--project-root", project_root,
                "--recording-id", recording_id,
                "--epoch-sec", str(epoch_sec),
            ]

            with st.spinner("Computing features..."):
                result = run_command(cmd)

            if result.returncode == 0:
                st.success("Features computed.")
                st.code(result.stdout)
            else:
                st.error("Feature computation failed.")
                st.code(result.stdout)
                st.code(result.stderr)

    with run_cols[2]:
        if st.button("3. Run Layer 1 Wake/Sleep"):
            cmd = [
                sys.executable,
                "pipelines/03_layer1_emg_wake_sleep.py",
                "--project-root", project_root,
                "--recording-id", recording_id,
            ]

            with st.spinner("Running Layer 1..."):
                result = run_command(cmd)

            if result.returncode == 0:
                st.success("Layer 1 complete.")
                st.code(result.stdout)
            else:
                st.error("Layer 1 failed.")
                st.code(result.stdout)
                st.code(result.stderr)

    st.markdown("---")
    manifest = load_manifest(project_root)

    if manifest is not None:
        st.subheader("Current recordings")
        st.dataframe(manifest, width='stretch')


# =============================================================================
# TAB 2 — QC VIEWER
# =============================================================================

with tab_qc:
    st.subheader("QC viewer")

    project_root_qc = st.text_input(
        "Project root",
        default_project,
        key="qc_project_root",
    )

    manifest = load_manifest(project_root_qc)

    if manifest is None:
        st.info("No recordings found yet. Import a .mat recording first.")
        st.stop()

    recording_ids = manifest["recording_id"].astype(str).tolist()

    recording_id_qc = st.selectbox(
        "Recording",
        recording_ids,
        key="qc_recording_id",
    )

    row = manifest[manifest["recording_id"].astype(str) == recording_id_qc].iloc[0]
    recording_dir = Path(row["recording_dir"])

    layer1_file = recording_dir / "layer1_wake_sleep.csv"
    metadata_file = recording_dir / "metadata.json"

    if not metadata_file.exists():
        st.error("metadata.json not found for this recording.")
        st.stop()

    metadata = json.loads(metadata_file.read_text())
    duration_min = float(metadata["duration_s"]) / 60

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Duration", f"{duration_min / 60:.2f} h")
    with m2:
        st.metric("Sampling rate", f"{float(metadata['sampling_rate_hz']):.1f} Hz")
    with m3:
        st.metric("Manual scoring", "yes" if (recording_dir / "manual_scoring_aligned.csv").exists() else "no")
    with m4:
        st.metric("Layer 1", "yes" if layer1_file.exists() else "missing")

    if not layer1_file.exists():
        st.warning("Layer 1 has not been computed yet.")
        st.stop()

    st.markdown("### Navigate recording")

    if "viewer_window_min" not in st.session_state:
        st.session_state.viewer_window_min = min(15.0, duration_min)

    if "viewer_start_min" not in st.session_state:
        st.session_state.viewer_start_min = 0.0

    nav_cols = st.columns([1, 1, 1, 1, 1.5, 3])

    with nav_cols[0]:
        if st.button("5 min"):
            st.session_state.viewer_window_min = min(5.0, duration_min)
            st.rerun()

    with nav_cols[1]:
        if st.button("15 min"):
            st.session_state.viewer_window_min = min(15.0, duration_min)
            st.rerun()

    with nav_cols[2]:
        if st.button("30 min"):
            st.session_state.viewer_window_min = min(30.0, duration_min)
            st.rerun()

    with nav_cols[3]:
        if st.button("60 min"):
            st.session_state.viewer_window_min = min(60.0, duration_min)
            st.rerun()

    with nav_cols[4]:
        st.session_state.viewer_window_min = st.number_input(
            "Minutes shown",
            min_value=0.5,
            max_value=float(duration_min),
            value=float(st.session_state.viewer_window_min),
            step=0.5,
        )

    max_start = max(0.0, duration_min - st.session_state.viewer_window_min)

    st.session_state.viewer_start_min = min(st.session_state.viewer_start_min, max_start)

    move_cols = st.columns([1, 1, 6, 1, 1])

    with move_cols[0]:
        if st.button("◀ −window"):
            st.session_state.viewer_start_min = max(
                0.0,
                st.session_state.viewer_start_min - st.session_state.viewer_window_min,
            )
            st.rerun()

    with move_cols[1]:
        if st.button("◀ −10"):
            st.session_state.viewer_start_min = max(0.0, st.session_state.viewer_start_min - 10)
            st.rerun()

    with move_cols[2]:
        st.session_state.viewer_start_min = st.slider(
            "Progress through recording",
            min_value=0.0,
            max_value=float(max_start),
            value=float(st.session_state.viewer_start_min),
            step=0.5,
        )

    with move_cols[3]:
        if st.button("+10 ▶"):
            st.session_state.viewer_start_min = min(max_start, st.session_state.viewer_start_min + 10)
            st.rerun()

    with move_cols[4]:
        if st.button("+window ▶"):
            st.session_state.viewer_start_min = min(
                max_start,
                st.session_state.viewer_start_min + st.session_state.viewer_window_min,
            )
            st.rerun()

    start_min = float(st.session_state.viewer_start_min)
    window_min = float(st.session_state.viewer_window_min)
    end_min = min(duration_min, start_min + window_min)

    st.write(f"**Showing:** {start_min:.1f}–{end_min:.1f} min")

    # -------------------------------------------------------------------------
    # Optional video QC
    # -------------------------------------------------------------------------
    with st.expander("Video QC for selected window", expanded=False):
        current_video_file = str(metadata.get("video_file", ""))

        video_file_input = st.text_input(
            "Video file path",
            value=current_video_file,
            key="qc_video_file_path",
            help="Use .mp4, .mov, .m4v or .webm for best browser playback. AVI may not play in Streamlit.",
        )

        video_offset_s = st.number_input(
            "Video offset relative to EEG/EMG, seconds",
            value=float(metadata.get("video_offset_s", 0.0)),
            step=1.0,
            key="qc_video_offset_s",
            help="Positive values start the video later. Negative values start it earlier.",
        )

        save_video_to_metadata = st.checkbox(
            "Save video path and offset to metadata",
            value=True,
            key="qc_save_video_metadata",
        )

        if st.button("Save/update video settings"):
            if video_file_input and not Path(video_file_input).exists():
                st.error(f"Video file not found: {video_file_input}")
            else:
                metadata["video_file"] = video_file_input
                metadata["video_offset_s"] = float(video_offset_s)

                if save_video_to_metadata:
                    metadata_file.write_text(json.dumps(metadata, indent=2))
                    st.success("Video settings saved to metadata.json.")
                else:
                    st.info("Video settings updated for this session only.")

        if not video_file_input:
            st.info("No video selected.")
        elif not Path(video_file_input).exists():
            st.warning(f"Video file does not exist: {video_file_input}")
        else:
            video_start_s = max(0.0, float(start_min) * 60 + float(video_offset_s))

            st.write(
                f"**Video start:** {video_start_s:.1f} s "
                f"for signal window {start_min:.1f}–{end_min:.1f} min"
            )

            suffix = Path(video_file_input).suffix.lower()

            if suffix in [".mp4", ".mov", ".m4v", ".webm"]:
                st.video(video_file_input, start_time=int(video_start_s))
            else:
                st.warning(
                    f"{suffix} may not play in the browser. "
                    "Convert it to .mp4 or use a short MP4 clip for QC."
                )

    st.markdown("---")

    try:
        fig = make_qc_plot(
            recording_dir=recording_dir,
            start_min=start_min,
            window_min=window_min,
        )

        st.plotly_chart(fig, width='stretch')

        st.caption(
            "Use the Plotly range slider at the bottom of the figure to slide through the full recording. "
            "All panels move together."
        )


    except Exception as e:
        st.error(f"Could not make QC plot: {repr(e)}")

    with st.expander("Layer 1 summary"):
        l1 = read_csv_fast(layer1_file)
        st.write(l1["layer1_label"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
        st.dataframe(l1.head(1000), width='stretch')


# =============================================================================
# TAB ABOUT
# =============================================================================



# =============================================================================
# TAB 3 — SOMNOTATE
# =============================================================================
with tab_somnotate:
    st.subheader("Layer 2 — Somnotate Wake/NREM/REM")

    st.write(
        "Choose whether you want to use an existing trained Somnotate model "
        "or train a new model from manually scored recordings."
    )

    project_root_som = st.text_input(
        "Project root",
        default_project,
        key="som_project_root",
    )

    manifest_som = load_manifest(project_root_som)

    if manifest_som is None:
        st.info("No recordings found yet. Import a .mat recording first.")
        st.stop()

    recording_ids_all = manifest_som["recording_id"].astype(str).tolist()

    som_mode = st.radio(
        "What do you want to do?",
        [
            "Use existing trained model",
            "Train new Somnotate model",
        ],
        horizontal=True,
    )

    st.markdown("### Somnotate installation")

    s1, s2 = st.columns(2)

    with s1:
        somnotate_root = st.text_input(
            "Somnotate repository folder",
            str(Path.home() / "somnotate"),
            help=(
                "You can provide the main cloned Somnotate folder. "
                "The app will automatically find example_pipeline inside it."
            ),
        )

    with s2:
        somnotate_python = st.text_input(
            "Somnotate Python executable",
            str(Path.home() / "anaconda3" / "envs" / "somnotate_env" / "bin" / "python"),
        )

    target_fs = st.number_input(
        "Somnotate target sampling rate, Hz",
        min_value=1.0,
        value=512.0,
        step=1.0,
        help="Use the sampling rate expected by the model. Your current model expects 512 Hz.",
    )

    # ------------------------------------------------------------------
    # OPTION A — USE EXISTING MODEL
    # ------------------------------------------------------------------
    if som_mode == "Use existing trained model":
        st.markdown("---")
        st.markdown("### Use existing trained model")

        recording_id_score = st.selectbox(
            "Recording to score",
            recording_ids_all,
            key="som_existing_recording",
        )

        model_file = st.text_input(
            "Trained Somnotate model file",
            "/Users/margaridaseabra/Library/CloudStorage/OneDrive-UniversityofCopenhagen/PD-Katia/Data/prepared_data/models/final_models/final_WT_week2_week21_model_512hz.pickle",
        )

        st.caption(
            "This will prepare the selected recording for Somnotate, run preprocessing, "
            "score Wake/NREM/REM, compute probabilities, and import the results into the QC viewer."
        )

        if st.button("Run Somnotate using existing model", type="primary"):
            cmd = [
                sys.executable,
                "pipelines/10_somnotate_layer.py",
                "use-existing-model",
                "--project-root", str(project_root_som),
                "--recording-ids", str(recording_id_score),
                "--somnotate-root", str(somnotate_root),
                "--somnotate-python", str(somnotate_python),
                "--model-file", str(model_file),
                "--target-fs", str(target_fs),
                "--prepare",
                "--preprocess",
                "--score",
                "--probabilities",
                "--import-results",
            ]

            with st.spinner("Running Somnotate with existing model..."):
                result = run_command(cmd)

            if result.returncode == 0:
                st.success("Somnotate scoring complete. Go to QC viewer.")
                st.code(result.stdout)
            else:
                st.error("Somnotate scoring failed.")
                st.code(result.stdout)
                st.code(result.stderr)

        rec_dir = Path(
            manifest_som[
                manifest_som["recording_id"].astype(str) == recording_id_score
            ].iloc[0]["recording_dir"]
        )

        som_results = rec_dir / "somnotate" / "somnotate_results_timeseries.csv"

        if som_results.exists():
            st.markdown("### Current Somnotate results")
            som = pd.read_csv(som_results)

            st.write(
                som["somnotate_state"]
                .value_counts(normalize=True)
                .mul(100)
                .round(1)
                .astype(str) + "%"
            )

            prob_cols = [c for c in som.columns if c.startswith("somnotate_P_")]
            if prob_cols:
                st.write("Probability columns:", prob_cols)

    # ------------------------------------------------------------------
    # OPTION B — TRAIN NEW MODEL
    # ------------------------------------------------------------------
    else:
        st.markdown("---")
        st.markdown("### Train new Somnotate model")

        st.write(
            "Training recordings must have manual scoring imported from the `.mat` file. "
            "Test recordings are optional, but useful to immediately evaluate the new model."
        )

        training_ready = []

        for _, rr in manifest_som.iterrows():
            rec_id = str(rr["recording_id"])
            rec_dir = Path(rr["recording_dir"])
            manual_file = rec_dir / "manual_scoring_aligned.csv"

            if manual_file.exists():
                training_ready.append(rec_id)

        if not training_ready:
            st.warning(
                "No training-ready recordings found. Import at least one `.mat` file "
                "that contains manual scoring."
            )
        else:
            train_recordings = st.multiselect(
                "Training recordings",
                training_ready,
                default=training_ready[:1],
            )

            test_recordings = st.multiselect(
                "Test recordings",
                recording_ids_all,
                default=[],
                help="Optional. The trained model will be applied to these recordings after training.",
            )

            model_name = st.text_input(
                "New model name",
                "lab_somnotate_model_v1",
            )

            st.caption(
                "This will prepare training recordings, run Somnotate preprocessing, "
                "train a model, and optionally score the test recordings."
            )

            if st.button("Train Somnotate model", type="primary"):
                if not train_recordings:
                    st.error("Select at least one training recording.")
                else:
                    cmd = [
                        sys.executable,
                        "pipelines/10_somnotate_layer.py",
                        "train-model",
                        "--project-root", str(project_root_som),
                        "--train-recording-ids", ",".join(train_recordings),
                        "--test-recording-ids", ",".join(test_recordings),
                        "--somnotate-root", str(somnotate_root),
                        "--somnotate-python", str(somnotate_python),
                        "--model-name", str(model_name),
                        "--target-fs", str(target_fs),
                        "--prepare",
                        "--preprocess",
                    ]

                    with st.spinner("Training Somnotate model..."):
                        result = run_command(cmd)

                    if result.returncode == 0:
                        st.success("Somnotate model trained.")
                        st.code(result.stdout)
                    else:
                        st.error("Somnotate training failed.")
                        st.code(result.stdout)
                        st.code(result.stderr)

            st.markdown("### Existing trained models in this project")

            model_dir = Path(project_root_som) / "somnotate_models"

            if model_dir.exists():
                models = sorted(model_dir.glob("*.pickle"))

                if models:
                    st.dataframe(
                        pd.DataFrame({
                            "model_file": [str(m) for m in models],
                            "modified": [pd.Timestamp(m.stat().st_mtime, unit="s") for m in models],
                        }),
                        width='stretch',
                    )
                else:
                    st.info("No trained models found yet.")
            else:
                st.info("No somnotate_models folder found yet.")




# =============================================================================
# TAB 4 — REVIEW / EDIT SCORING
# =============================================================================
with tab_review:
    st.subheader("Active review / edit final scoring")

    components.html(
        """
        <script>
        const parentDoc = window.parent.document;

        if (!window.parent.__sleep_stage_qc_review_undo_v3) {
            window.parent.__sleep_stage_qc_review_undo_v3 = true;

            window.parent.addEventListener("keydown", function(e) {
                const isUndo = (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z";

                if (!isUndo) {
                    return;
                }

                e.preventDefault();
                e.stopPropagation();

                const buttons = Array.from(parentDoc.querySelectorAll("button"));
                const undoButton = buttons.find(
                    b => b.innerText && b.innerText.includes("Undo last action")
                );

                if (undoButton) {
                    undoButton.click();
                }
            }, true);
        }
        </script>
        """,
        height=0,
    )


    components.html(
        """
        <script>
        const parentDoc = window.parent.document;

        if (!window.parent.__sleep_stage_qc_undo_shortcut_installed_v2) {
            window.parent.__sleep_stage_qc_undo_shortcut_installed_v2 = true;

            window.parent.addEventListener("keydown", function(e) {
                const isUndo = (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z";

                if (!isUndo) {
                    return;
                }

                const active = parentDoc.activeElement;
                const tag = active ? active.tagName.toLowerCase() : "";

                // Avoid hijacking text editing inside inputs/textareas.
                if (tag === "input" || tag === "textarea") {
                    return;
                }

                e.preventDefault();

                const buttons = Array.from(parentDoc.querySelectorAll("button"));
                const undoButton = buttons.find(
                    b => b.innerText && b.innerText.includes("Undo last action")
                );

                if (undoButton) {
                    undoButton.click();
                }
            });
        }
        </script>
        """,
        height=0,
    )


    # Keyboard shortcut: Ctrl+Z on Windows/Linux or Cmd+Z on Mac.
    # This triggers the visible "Undo last pending edit" button.
    components.html(
        """
        <script>
        const parentDoc = window.parent.document;

        if (!window.parent.__sleep_stage_qc_undo_shortcut_installed) {
            window.parent.__sleep_stage_qc_undo_shortcut_installed = true;

            window.parent.addEventListener("keydown", function(e) {
                const isUndo = (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z";

                if (!isUndo) {
                    return;
                }

                e.preventDefault();

                const buttons = Array.from(parentDoc.querySelectorAll("button"));
                const undoButton = buttons.find(
                    b => b.innerText && b.innerText.trim().includes("Undo last pending edit")
                );

                if (undoButton) {
                    undoButton.click();
                }
            });
        }
        </script>
        """,
        height=0,
    )


    st.write(
        "Inspect suspicious periods, approve model/manual bouts, manually label selected periods, "
        "and export the final scoring as CSV or MAT."
    )

    project_root_review = st.text_input(
        "Project root",
        default_project,
        key="review_project_root",
    )

    manifest_review = load_manifest(project_root_review)

    if manifest_review is None:
        st.info("No recordings found yet.")
        st.stop()

    recording_ids_review = manifest_review["recording_id"].astype(str).tolist()

    recording_id_review = st.selectbox(
        "Recording",
        recording_ids_review,
        key="review_recording",
    )

    row_review = manifest_review[
        manifest_review["recording_id"].astype(str) == recording_id_review
    ].iloc[0]

    recording_dir_review = Path(row_review["recording_dir"])
    metadata_review = json.loads((recording_dir_review / "metadata.json").read_text())
    duration_min_review = float(metadata_review["duration_s"]) / 60
    max_start_review = max(0.0, duration_min_review - 0.5)

    queue_file = recording_dir_review / "review_queue.csv"
    final_file = recording_dir_review / "final_scoring.csv"

    # If the Statistics/Dissociation tab sent an event here, jump the Review viewer to it.
    if "pending_dissociation_jump" in st.session_state:
        jump = st.session_state.pop("pending_dissociation_jump")

        jump_start = float(jump.get("start_min", 0.0))
        jump_end = float(jump.get("end_min", jump_start + 1.0))

        jump_start = max(0.0, jump_start)
        jump_end = min(duration_min_review, max(jump_start + 0.5, jump_end))

        st.session_state.review_mode = "Manual interval"
        st.session_state.review_manual_start_min = float(jump_start)
        st.session_state.review_manual_window_min = float(jump_end - jump_start)
        st.session_state.selected_label_start_min = float(jump_start)
        st.session_state.selected_label_end_min = float(jump_end)
        st.session_state.selected_period_slider_revision = (
            st.session_state.get("selected_period_slider_revision", 0) + 1
        )

        st.success(
            f"Loaded dissociation event {jump.get('event_id', '')}: "
            f"{jump_start:.2f}–{jump_end:.2f} min | {jump.get('reason', '')}"
        )

    # ------------------------------------------------------------------
    # Dissociation queue navigation
    # ------------------------------------------------------------------
    st.markdown("### Dissociation review queue")

    dissociation_analysis_dir = recording_dir_review / "dissociation_analysis"
    dissociation_events_file = dissociation_analysis_dir / "dissociation_events.csv"

    dq1, dq2 = st.columns([1, 3])

    with dq1:
        dissociation_threshold_review = st.number_input(
            "Minimum dissociation index",
            min_value=0.05,
            max_value=0.95,
            value=0.20,
            step=0.05,
            key="review_dissociation_threshold",
            help=(
                "Lower values show more candidate events. "
                "Higher values show only stronger model disagreements."
            ),
        )

    with dq2:
        st.caption(
            "Use this queue to jump through moments where Manual, Somnotate, Layer 1, "
            "confidence, and EMG features suggest the strongest disagreement."
        )

    build_dissociation_col, queue_info_col = st.columns([1, 3])

    with build_dissociation_col:
        if st.button("Build / refresh dissociation queue", key="build_dissociation_queue_review"):
            diss_script = Path("pipelines/30_dissociation_analysis.py")

            if not diss_script.exists():
                st.error(
                    "pipelines/30_dissociation_analysis.py was not found. "
                    "Create the dissociation analysis pipeline first."
                )
            else:
                cmd = [
                    sys.executable,
                    str(diss_script),
                    "--project-root", str(project_root_review),
                    "--recording-id", str(recording_id_review),
                    "--threshold", str(float(dissociation_threshold_review)),
                ]

                with st.spinner("Computing dissociation events..."):
                    result = run_command(cmd)

                if result.returncode == 0:
                    st.success("Dissociation queue built.")
                    st.code(result.stdout)
                    st.rerun()
                else:
                    st.error("Dissociation analysis failed.")
                    st.code(result.stdout)
                    st.code(result.stderr)

    if not dissociation_events_file.exists():
        with queue_info_col:
            st.info(
                "No dissociation queue found yet. Click **Build / refresh dissociation queue**."
            )
    else:
        diss_events = pd.read_csv(dissociation_events_file)

        if len(diss_events) == 0:
            with queue_info_col:
                st.info("Dissociation queue exists, but no events passed the threshold.")
        else:
            if "review_dissociation_event_index" not in st.session_state:
                st.session_state.review_dissociation_event_index = 0

            st.session_state.review_dissociation_event_index = max(
                0,
                min(
                    int(st.session_state.review_dissociation_event_index),
                    len(diss_events) - 1,
                ),
            )

            current_diss_event = diss_events.iloc[int(st.session_state.review_dissociation_event_index)]

            with queue_info_col:
                st.success(
                    f"Loaded dissociation queue: {len(diss_events)} events. "
                    f"Current rank #{int(current_diss_event['rank'])}, "
                    f"index={float(current_diss_event['max_dissociation_index']):.2f}"
                )

            with st.expander("Jump through dissociation events", expanded=True):
                event_descriptions = diss_events.apply(
                    lambda r: (
                        f"#{int(r['rank'])} | {r['event_id']} | "
                        f"{r['start_min']:.2f}–{r['end_min']:.2f} min | "
                        f"index={r['max_dissociation_index']:.2f} | "
                        f"{r['main_reason']}"
                    ),
                    axis=1,
                ).tolist()

                nav1, nav2, nav3, nav4 = st.columns([1, 1, 4, 1])

                with nav1:
                    if st.button("◀ Previous", key="previous_dissociation_event"):
                        st.session_state.review_dissociation_event_index = max(
                            0,
                            int(st.session_state.review_dissociation_event_index) - 1,
                        )

                        ev = diss_events.iloc[int(st.session_state.review_dissociation_event_index)]

                        set_review_window_around_event(
                            ev,
                            duration_min_review=duration_min_review,
                            fixed_window_min=15.0,
                        )
                        st.rerun()

                with nav2:
                    if st.button("Next ▶", key="next_dissociation_event"):
                        st.session_state.review_dissociation_event_index = min(
                            len(diss_events) - 1,
                            int(st.session_state.review_dissociation_event_index) + 1,
                        )

                        ev = diss_events.iloc[int(st.session_state.review_dissociation_event_index)]

                        set_review_window_around_event(
                            ev,
                            duration_min_review=duration_min_review,
                            fixed_window_min=15.0,
                        )
                        st.rerun()

                with nav3:
                    selected_event_desc = st.selectbox(
                        "Dissociation event",
                        event_descriptions,
                        index=int(st.session_state.review_dissociation_event_index),
                        key="review_dissociation_event_selector",
                    )
                    selected_event_index = event_descriptions.index(selected_event_desc)

                with nav4:
                    if st.button("Load event", key="load_dissociation_event"):
                        st.session_state.review_dissociation_event_index = int(selected_event_index)

                        ev = diss_events.iloc[int(selected_event_index)]

                        set_review_window_around_event(
                            ev,
                            duration_min_review=duration_min_review,
                            fixed_window_min=15.0,
                        )
                        st.rerun()

                ev = diss_events.iloc[int(st.session_state.review_dissociation_event_index)]

                st.caption(
                    f"Current event: **{ev['event_id']}** | "
                    f"{float(ev['start_min']):.2f}–{float(ev['end_min']):.2f} min | "
                    f"max index={float(ev['max_dissociation_index']):.2f} | "
                    f"mean index={float(ev['mean_dissociation_index']):.2f} | "
                    f"{ev['main_reason']}"
                )

                if "states_at_peak" in ev:
                    st.caption(f"States at peak: {ev['states_at_peak']}")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Prepare review
    # ------------------------------------------------------------------
    st.markdown("### 3. Inspect QC plot and choose selected period")

    st.caption(
        "Use the normal QC plot for inspection. You can select a manual period with the red slider "
        "or Plotly box-select. If you box-select over a Manual / Layer 1 / Somnotate scoring row, "
        "the app selects that full bout."
    )

    # Safety fallback: make sure the selected review window exists before plotting.
    # This prevents NameError if the Review tab was patched and the interval-selection
    # block is not reached before the QC plot section.
    if "selected_start_min" not in locals():
        selected_start_min = float(st.session_state.get("review_manual_start_min", 0.0))

    if "selected_window_min" not in locals():
        selected_window_min = float(st.session_state.get("review_manual_window_min", min(15.0, duration_min_review)))

    selected_start_min = max(0.0, min(float(selected_start_min), float(duration_min_review)))
    selected_window_min = max(0.5, min(float(selected_window_min), float(duration_min_review)))

    if selected_start_min + selected_window_min > duration_min_review:
        selected_window_min = max(0.5, duration_min_review - selected_start_min)

    selected_end_min = float(selected_start_min) + float(selected_window_min)

    # --------------------------------------------------------------
    # App-level window navigation
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # Window navigation: this is the main way to slide through the recording
    # while keeping raw EEG/EMG quality.
    # --------------------------------------------------------------
    st.markdown("#### Window navigation")

    if "review_window_slider_revision" not in st.session_state:
        st.session_state.review_window_slider_revision = 0

    fixed_window_min = st.number_input(
        "Window length, min",
        min_value=1.0,
        max_value=min(120.0, float(duration_min_review)),
        value=float(st.session_state.get("review_manual_window_min", 15.0)),
        step=1.0,
        key="review_fixed_window_length_min",
        help="EEG/EMG are loaded raw only for this visible window.",
    )

    fixed_window_min = min(float(fixed_window_min), float(duration_min_review))
    max_window_start = max(0.0, float(duration_min_review) - float(fixed_window_min))

    current_window_start = float(st.session_state.get("review_manual_start_min", selected_start_min))
    current_window_start = max(0.0, min(current_window_start, max_window_start))

    nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([1, 1, 5, 1, 1])

    with nav_c1:
        if st.button("◀ 15 min", key="review_back_15_min"):
            st.session_state.review_manual_start_min = max(0.0, current_window_start - 15.0)
            st.session_state.review_manual_window_min = float(fixed_window_min)
            st.session_state.review_window_slider_revision += 1
            st.rerun()

    with nav_c2:
        if st.button("◀ 5 min", key="review_back_5_min"):
            st.session_state.review_manual_start_min = max(0.0, current_window_start - 5.0)
            st.session_state.review_manual_window_min = float(fixed_window_min)
            st.session_state.review_window_slider_revision += 1
            st.rerun()

    with nav_c3:
        selected_start_min = st.slider(
            "Window start, min",
            min_value=0.0,
            max_value=float(max_window_start),
            value=float(current_window_start),
            step=0.25,
            key=f"review_window_start_slider_{st.session_state.review_window_slider_revision}",
        )

    with nav_c4:
        if st.button("5 min ▶", key="review_forward_5_min"):
            st.session_state.review_manual_start_min = min(max_window_start, current_window_start + 5.0)
            st.session_state.review_manual_window_min = float(fixed_window_min)
            st.session_state.review_window_slider_revision += 1
            st.rerun()

    with nav_c5:
        if st.button("15 min ▶", key="review_forward_15_min"):
            st.session_state.review_manual_start_min = min(max_window_start, current_window_start + 15.0)
            st.session_state.review_manual_window_min = float(fixed_window_min)
            st.session_state.review_window_slider_revision += 1
            st.rerun()

    selected_window_min = float(fixed_window_min)
    selected_end_min = min(float(duration_min_review), float(selected_start_min) + float(selected_window_min))

    st.session_state.review_manual_start_min = float(selected_start_min)
    st.session_state.review_manual_window_min = float(selected_window_min)

    st.caption(
        f"Visible raw EEG/EMG window: {selected_start_min:.2f}–{selected_end_min:.2f} min "
        f"({selected_window_min:.1f} min)."
    )

    # --------------------------------------------------------------
    # Safety: define selected-period defaults after window navigation
    # --------------------------------------------------------------
    if "selected_start_min" not in locals():
        selected_start_min = float(st.session_state.get("review_manual_start_min", 0.0))

    if "selected_window_min" not in locals():
        selected_window_min = float(st.session_state.get("review_manual_window_min", 15.0))

    selected_start_min = max(0.0, min(float(selected_start_min), float(duration_min_review)))
    selected_window_min = max(0.1, min(float(selected_window_min), float(duration_min_review)))

    selected_end_min = min(
        float(duration_min_review),
        float(selected_start_min) + float(selected_window_min),
    )

    if selected_end_min <= selected_start_min:
        selected_end_min = min(float(duration_min_review), selected_start_min + 0.5)

    default_label_start = float(
        st.session_state.get("selected_label_start_min", selected_start_min)
    )
    default_label_end = float(
        st.session_state.get("selected_label_end_min", selected_end_min)
    )

    default_label_start = max(
        float(selected_start_min),
        min(float(default_label_start), float(selected_end_min)),
    )
    default_label_end = max(
        float(selected_start_min),
        min(float(default_label_end), float(selected_end_min)),
    )

    if default_label_end <= default_label_start:
        default_label_end = min(float(selected_end_min), default_label_start + 0.5)

    # Safety: initialize slider revision keys before using them in widget keys.
    if "selected_period_slider_revision" not in st.session_state:
        st.session_state.selected_period_slider_revision = 0

    if "review_window_slider_revision" not in st.session_state:
        st.session_state.review_window_slider_revision = 0

    selected_period = st.slider(
        "Selected period to label, min",
        min_value=float(selected_start_min),
        max_value=float(selected_end_min),
        value=(float(default_label_start), float(default_label_end)),
        step=0.05,
        key=f"selected_period_range_slider_{st.session_state.selected_period_slider_revision}",
    )

    st.session_state.selected_label_start_min = float(selected_period[0])
    st.session_state.selected_label_end_min = float(selected_period[1])

    # --------------------------------------------------------------
    # Compact decision panel: selected period + source/manual choice
    # --------------------------------------------------------------
    st.markdown("#### Decision for selected period")

    label_start_min_now = float(st.session_state.selected_label_start_min)
    label_end_min_now = float(st.session_state.selected_label_end_min)

    decision_c1, decision_c2, decision_c3 = st.columns([3, 1, 1])

    with decision_c1:
        st.info(
            f"Selected period: **{label_start_min_now:.2f}–{label_end_min_now:.2f} min** "
            f"({(label_end_min_now - label_start_min_now) * 60:.1f} s)"
        )

    with decision_c2:
        if st.button("Use full visible window"):
            st.session_state.selected_label_start_min = float(selected_start_min)
            st.session_state.selected_label_end_min = float(selected_end_min)
            st.session_state.selected_period_slider_revision = (
                st.session_state.get("selected_period_slider_revision", 0) + 1
            )
            st.rerun()

    with decision_c3:
        if st.button("↶ Undo last action", key="undo_last_review_action_button"):
            ok, msg = undo_last_review_action(
                recording_dir=recording_dir_review,
                recording_id=recording_id_review,
            )

            if ok:
                st.success(msg)
            else:
                st.info(msg)

    compact_notes = st.text_input(
        "Notes for this edit",
        value="review edit",
        key="compact_selected_period_notes",
    )

    source_cols = st.columns(3)

    with source_cols[0]:
        if st.button("Accept Somnotate for selected period", key="accept_somnotate_selected"):
            ok, msg = apply_source_label_fast(
                recording_dir=recording_dir_review,
                recording_id=recording_id_review,
                start_s=label_start_min_now * 60,
                end_s=label_end_min_now * 60,
                source_name="Somnotate",
                notes=str(compact_notes),
            )
            if ok:
                shade_label = dominant_state_for_source_interval(
                    recording_dir=recording_dir_review,
                    source_name="Somnotate",
                    start_s=label_start_min_now * 60,
                    end_s=label_end_min_now * 60,
                )
                remember_review_shade(
                    recording_id=recording_id_review,
                    start_min=label_start_min_now,
                    end_min=label_end_min_now,
                    label=shade_label,
                    source="accepted_somnotate",
                )
                st.success(msg)
            else:
                st.error(msg)

    with source_cols[1]:
        if st.button("Accept Manual for selected period", key="accept_manual_selected"):
            ok, msg = apply_source_label_fast(
                recording_dir=recording_dir_review,
                recording_id=recording_id_review,
                start_s=label_start_min_now * 60,
                end_s=label_end_min_now * 60,
                source_name="Manual",
                notes=str(compact_notes),
            )
            if ok:
                shade_label = dominant_state_for_source_interval(
                    recording_dir=recording_dir_review,
                    source_name="Manual",
                    start_s=label_start_min_now * 60,
                    end_s=label_end_min_now * 60,
                )
                remember_review_shade(
                    recording_id=recording_id_review,
                    start_min=label_start_min_now,
                    end_min=label_end_min_now,
                    label=shade_label,
                    source="accepted_manual",
                )
                st.success(msg)
            else:
                st.error(msg)

    with source_cols[2]:
        if st.button("Accept Layer 1 for selected period", key="accept_layer1_selected"):
            ok, msg = apply_source_label_fast(
                recording_dir=recording_dir_review,
                recording_id=recording_id_review,
                start_s=label_start_min_now * 60,
                end_s=label_end_min_now * 60,
                source_name="Layer 1",
                notes=str(compact_notes),
            )
            if ok:
                shade_label = dominant_state_for_source_interval(
                    recording_dir=recording_dir_review,
                    source_name="Layer 1",
                    start_s=label_start_min_now * 60,
                    end_s=label_end_min_now * 60,
                )
                remember_review_shade(
                    recording_id=recording_id_review,
                    start_min=label_start_min_now,
                    end_min=label_end_min_now,
                    label=shade_label,
                    source="accepted_layer_1",
                )
                st.success(msg)
            else:
                st.error(msg)

    manual_cols = st.columns(5)
    quick_label_now = None

    with manual_cols[0]:
        if st.button("0 = Wake", key="compact_score_wake"):
            quick_label_now = "Wake"

    with manual_cols[1]:
        if st.button("1 = NREM", key="compact_score_nrem"):
            quick_label_now = "NREM"

    with manual_cols[2]:
        if st.button("2 = REM", key="compact_score_rem"):
            quick_label_now = "REM"

    with manual_cols[3]:
        if st.button("Uncertain", key="compact_score_uncertain"):
            quick_label_now = "Uncertain"

    with manual_cols[4]:
        if st.button("Artifact", key="compact_score_artifact"):
            quick_label_now = "Artifact"

    if quick_label_now is not None:
        if label_end_min_now <= label_start_min_now:
            st.error("Selected end must be after selected start.")
        else:
            ok, msg = apply_manual_label_fast(
                recording_dir=recording_dir_review,
                recording_id=recording_id_review,
                start_s=label_start_min_now * 60,
                end_s=label_end_min_now * 60,
                label=quick_label_now,
                notes=str(compact_notes),
            )

            st.success(msg) if ok else st.error(msg)

    # --------------------------------------------------------------
    # Compact export panel near review controls
    # --------------------------------------------------------------
    with st.expander("Export scoring", expanded=False):
        manual_file_for_export = recording_dir_review / "manual_scoring_aligned.csv"
        has_manual_for_export = manual_file_for_export.exists()

        st.caption(
            "Export the current reviewed scoring. MAT exports include scoring, state labels, "
            "sources, epoch times, and epoch ids."
        )

        if not has_manual_for_export:
            st.info(
                "No manual scoring was provided. Export will use the scoring produced/reviewed in the app."
            )

            if st.button("Export app scoring as CSV and MAT", type="primary", key="top_export_app_only_no_manual"):
                ok, msg, csv_out, mat_out = export_scoring_from_app(
                    recording_dir=recording_dir_review,
                    recording_id=recording_id_review,
                    mode="app",
                )

                if ok:
                    st.success(msg)
                    st.session_state.last_export_csv = str(csv_out)
                    st.session_state.last_export_mat = str(mat_out)
                else:
                    st.error(msg)

        else:
            st.info(
                "Manual scoring exists. You can export app scoring only, or a mixed version: "
                "app-reviewed periods + original manual scoring elsewhere."
            )

            ex1, ex2 = st.columns(2)

            with ex1:
                if st.button("Export app scoring only", type="primary", key="top_export_app_only"):
                    ok, msg, csv_out, mat_out = export_scoring_from_app(
                        recording_dir=recording_dir_review,
                        recording_id=recording_id_review,
                        mode="app",
                    )

                    if ok:
                        st.success(msg)
                        st.session_state.last_export_csv = str(csv_out)
                        st.session_state.last_export_mat = str(mat_out)
                    else:
                        st.error(msg)

            with ex2:
                if st.button("Export mixed manual + app scoring", type="primary", key="top_export_mixed"):
                    ok, msg, csv_out, mat_out = export_scoring_from_app(
                        recording_dir=recording_dir_review,
                        recording_id=recording_id_review,
                        mode="mixed_manual_app",
                    )

                    if ok:
                        st.success(msg)
                        st.session_state.last_export_csv = str(csv_out)
                        st.session_state.last_export_mat = str(mat_out)
                    else:
                        st.error(msg)

        export_dir_review = recording_dir_review / "exports"

        if export_dir_review.exists():
            export_files = (
                sorted(export_dir_review.glob(f"{recording_id_review}_*.csv"))
                + sorted(export_dir_review.glob(f"{recording_id_review}_*.mat"))
            )

            if export_files:
                st.markdown("**Available exports**")

                for f in export_files:
                    mime = "text/csv" if f.suffix.lower() == ".csv" else "application/octet-stream"

                    st.download_button(
                        label=f"Download {f.name}",
                        data=f.read_bytes(),
                        file_name=f.name,
                        mime=mime,
                        key=f"top_download_export_{recording_id_review}_{f.name}",
                    )


    # --------------------------------------------------------------
    # Clear legends for scoring colors and probability traces
    # --------------------------------------------------------------
    st.markdown("#### Legend")

    st.markdown(
        """
        <div style="display:flex; gap:18px; flex-wrap:wrap; align-items:center; margin-bottom:6px;">
          <b>Scoring colors:</b>
          <span><span style="display:inline-block;width:14px;height:14px;background:#1f77b4;border-radius:3px;margin-right:5px;"></span>Wake</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:#f7c6d9;border-radius:3px;margin-right:5px;border:1px solid #ddd;"></span>Layer 1 Sleep</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:#ff7f0e;border-radius:3px;margin-right:5px;"></span>NREM</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:#2ca02c;border-radius:3px;margin-right:5px;"></span>REM</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:#9e9e9e;border-radius:3px;margin-right:5px;"></span>Uncertain</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:#000000;border-radius:3px;margin-right:5px;"></span>Artifact</span>
        </div>

        <div style="display:flex; gap:18px; flex-wrap:wrap; align-items:center; margin-bottom:12px;">
          <b>Probability traces:</b>
          <span><span style="display:inline-block;width:24px;border-top:2px dashed #1f77b4;margin-right:5px;vertical-align:middle;"></span>Layer 1 P(Wake)</span>
          <span><span style="display:inline-block;width:24px;border-top:2px dashed #ff7f0e;margin-right:5px;vertical-align:middle;"></span>Layer 1 P(Sleep)</span>
          <span><span style="display:inline-block;width:24px;border-top:2px dotted #9e9e9e;margin-right:5px;vertical-align:middle;"></span>Layer 1 uncertainty</span>
          <span><span style="display:inline-block;width:24px;border-top:2px solid #1f77b4;margin-right:5px;vertical-align:middle;"></span>Somnotate P(Wake)</span>
          <span><span style="display:inline-block;width:24px;border-top:2px solid #ff7f0e;margin-right:5px;vertical-align:middle;"></span>Somnotate P(NREM)</span>
          <span><span style="display:inline-block;width:24px;border-top:2px solid #2ca02c;margin-right:5px;vertical-align:middle;"></span>Somnotate P(REM)</span>
          <span><span style="display:inline-block;width:24px;border-top:2px solid #9e9e9e;margin-right:5px;vertical-align:middle;"></span>Somnotate uncertainty</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        fig_review = get_cached_review_base_qc_plot(
            recording_dir=recording_dir_review,
            start_min=selected_start_min,
            window_min=selected_window_min,
        )

        # Red outline = pending selected period.
        fig_review.add_vrect(
            x0=float(selected_period[0]),
            x1=float(selected_period[1]),
            fillcolor="rgba(255, 0, 0, 0.04)",
            line_width=1,
            line_color="rgba(255, 0, 0, 0.75)",
            row="all",
            col=1,
        )

        # Add invisible selection helpers.
        # Row 1 helpers allow selecting a scoring bout using box-select.
        layer1_for_selection = read_csv_fast(recording_dir_review / "layer1_wake_sleep.csv") if "read_csv_fast" in globals() else pd.read_csv(recording_dir_review / "layer1_wake_sleep.csv")
        layer1_for_selection["time_min"] = layer1_for_selection["t0_s"] / 60
        selection_x = layer1_for_selection["time_min"].to_numpy(dtype=float)

        scoring_row_y = {
            "Manual": "Manual",
            "Layer 1": "Layer 1",
            "Somnotate": "Somnotate",
        }

        for source_name, y_name in scoring_row_y.items():
            # Only add helpers for existing sources.
            if source_name == "Manual" and not (recording_dir_review / "manual_scoring_aligned.csv").exists():
                continue
            if source_name == "Somnotate" and not (recording_dir_review / "somnotate" / "somnotate_results_timeseries.csv").exists():
                continue

            fig_review.add_trace(
                go.Scatter(
                    x=selection_x,
                    y=np.array([y_name] * len(selection_x), dtype=object),
                    mode="markers",
                    marker=dict(size=6, color="rgba(0,0,0,0.01)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{source_name} selection helper",
                    customdata=np.array([source_name] * len(selection_x), dtype=object),
                ),
                row=1,
                col=1,
            )

        # Signal-row helpers allow manual box-select.
        for selection_row, selection_y in [(2, 0.0), (3, 0.0), (4, 0.0), (5, 0.5)]:
            fig_review.add_trace(
                go.Scattergl(
                    x=selection_x,
                    y=np.full(len(selection_x), selection_y, dtype=float),
                    mode="markers",
                    marker=dict(size=5, color="rgba(0,0,0,0.01)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name="manual selection helper",
                ),
                row=selection_row,
                col=1,
            )

        fig_review.update_layout(
            dragmode="select",
            clickmode="event+select",
            modebar_add=["select2d", "lasso2d", "pan2d", "zoom2d", "resetScale2d"],
        )

        # Raw-window mode: show only the window that was loaded.
        # Otherwise EEG/EMG are blank outside the loaded interval.
        fig_review.update_xaxes(
            range=[float(selected_start_min), float(selected_end_min)],
            rangeslider=dict(visible=False),
        )

        # Preserve zoom/pan/range-slider state across reruns.
        fig_review.update_layout(
            uirevision=f"{recording_id_review}_{selected_start_min:.3f}_{selected_window_min:.3f}"
        )

        # Add immediate in-session feedback from recent edits without rebuilding
        # the base QC figure.
        fig_review = add_recent_review_shades_to_fig(fig_review, recording_id_review)

        # --------------------------------------------------------------
        # Custom Plotly range-slider setup
        # --------------------------------------------------------------
        # Important: do NOT call fig_review.update_xaxes(rangeslider=...)
        # globally, because that creates a grey range slider inside every subplot.
        #
        # Instead:
        #   1. clamp all subplots to the current visible raw-window
        #   2. add one invisible full-duration anchor to the bottom panel
        #   3. enable the range slider only on the bottom x-axis
        try:
            # Clamp all visible axes to the currently loaded raw window.
            fig_review.update_xaxes(
                range=[float(selected_start_min), float(selected_end_min)],
                rangeslider=dict(visible=False),
            )

            # Add invisible full-duration anchor on the bottom subplot, so the
            # bottom range slider knows the total recording duration.
            fig_review.add_trace(
                go.Scattergl(
                    x=[0.0, float(duration_min_review)],
                    y=[0.0, 0.0],
                    mode="lines",
                    opacity=0.0,
                    hoverinfo="skip",
                    showlegend=False,
                    name="full-duration-anchor-bottom",
                ),
                row=5,
                col=1,
            )

            # Enable only the bottom x-axis range slider.
            # For the current 5-row QC figure, the bottom axis is xaxis5.
            if "xaxis5" in fig_review.layout:
                fig_review.layout.xaxis5.rangeslider = dict(visible=True)
                fig_review.layout.xaxis5.range = [
                    float(selected_start_min),
                    float(selected_end_min),
                ]
            else:
                # Fallback: enable only the last x-axis found.
                xaxis_names = sorted(
                    [k for k in fig_review.layout if str(k).startswith("xaxis")],
                    key=lambda z: int(str(z).replace("xaxis", "") or "1"),
                )
                if xaxis_names:
                    last_axis = xaxis_names[-1]
                    fig_review.layout[last_axis].rangeslider = dict(visible=True)
                    fig_review.layout[last_axis].range = [
                        float(selected_start_min),
                        float(selected_end_min),
                    ]

        except Exception as e:
            st.warning(f"Could not configure custom range slider cleanly: {e}")

        # Stable key for the Review/Edit Plotly chart.
        # It must be defined before st.plotly_chart().
        if "review_plot_key" not in locals():
            review_plot_key = (
                f"review_qc_box_select_"
                f"{recording_id_review}_"
                f"{float(selected_start_min):.3f}_"
                f"{float(selected_window_min):.3f}"
            )

        # Hide Plotly's crowded trace legend; use explicit legends above instead.
        fig_review.update_layout(showlegend=False)

        # Convert through Plotly JSON encoder so NumPy arrays become JSON-safe.
        fig_review_json_str = pio.to_json(fig_review, validate=False)

        interaction_event = plotly_relayout_viewer(
            fig_json_str=fig_review_json_str,
            height=980,
            key=review_plot_key,
        )

        selection_state = None

        if interaction_event is not None:
            if isinstance(interaction_event, dict):
                event_type = interaction_event.get("event_type", "")

                if event_type == "relayout":
                    new_x0 = float(interaction_event.get("x0"))
                    new_x1 = float(interaction_event.get("x1"))

                    # Clamp and keep a reasonable visible window.
                    new_x0 = max(0.0, min(new_x0, float(duration_min_review)))
                    new_x1 = max(new_x0 + 0.1, min(new_x1, float(duration_min_review)))

                    new_window = new_x1 - new_x0

                    # Update the app's visible window from Plotly slider/pan/zoom.
                    st.session_state.review_manual_start_min = float(new_x0)
                    st.session_state.review_manual_window_min = float(new_window)

                    # Keep selected period inside the new visible window if needed.
                    cur_sel0 = float(st.session_state.get("selected_label_start_min", new_x0))
                    cur_sel1 = float(st.session_state.get("selected_label_end_min", new_x1))

                    if cur_sel0 < new_x0 or cur_sel1 > new_x1:
                        st.session_state.selected_label_start_min = float(new_x0)
                        st.session_state.selected_label_end_min = float(min(new_x1, new_x0 + 1.0))

                    st.session_state.review_window_slider_revision = (
                        st.session_state.get("review_window_slider_revision", 0) + 1
                    )
                    st.session_state.selected_period_slider_revision = (
                        st.session_state.get("selected_period_slider_revision", 0) + 1
                    )

                    st.rerun()

                elif event_type == "selection":
                    sel_x0 = float(interaction_event.get("x0"))
                    sel_x1 = float(interaction_event.get("x1"))

                    sel_x0 = max(float(selected_start_min), min(sel_x0, float(selected_end_min)))
                    sel_x1 = max(float(selected_start_min), min(sel_x1, float(selected_end_min)))

                    if sel_x1 > sel_x0:
                        st.session_state.selected_label_start_min = float(sel_x0)
                        st.session_state.selected_label_end_min = float(sel_x1)
                        st.session_state.selected_period_slider_revision = (
                            st.session_state.get("selected_period_slider_revision", 0) + 1
                        )
                        st.rerun()

        parsed = parse_review_plot_selection(selection_state)

        if parsed is not None:
            if parsed["kind"] == "manual_range":
                plot_start_min = max(float(selected_start_min), min(float(parsed["start_min"]), float(selected_end_min)))
                plot_end_min = max(float(selected_start_min), min(float(parsed["end_min"]), float(selected_end_min)))

                if plot_end_min > plot_start_min:
                    changed = (
                        abs(float(st.session_state.get("selected_label_start_min", -999)) - plot_start_min) > 0.01
                        or abs(float(st.session_state.get("selected_label_end_min", -999)) - plot_end_min) > 0.01
                    )

                    st.session_state.selected_label_start_min = float(plot_start_min)
                    st.session_state.selected_label_end_min = float(plot_end_min)

                    if changed:
                        st.session_state.selected_period_slider_revision = (
                            st.session_state.get("selected_period_slider_revision", 0) + 1
                        )
                        # No explicit st.rerun() here.
                        # Plotly selection already triggered this run, and the next run
                        # will pre-process the selection before rendering the controls.

            elif parsed["kind"] == "scoring_bout":
                clicked_bout = find_bout_from_clicked_scoring(
                    recording_dir_review,
                    parsed["source"],
                    parsed["time_min"],
                )

                if clicked_bout is not None:
                    st.session_state.clicked_bout_source = parsed["source"]
                    st.session_state.clicked_bout_state = str(clicked_bout["state"])
                    st.session_state.clicked_bout_start_min = float(clicked_bout["start_min"])
                    st.session_state.clicked_bout_end_min = float(clicked_bout["end_min"])
                    st.session_state.clicked_bout_start_s = float(clicked_bout["start_s"])
                    st.session_state.clicked_bout_end_s = float(clicked_bout["end_s"])
                    st.session_state.clicked_bout_id = str(clicked_bout["bout_id"])

                    st.session_state.selected_label_start_min = float(clicked_bout["start_min"])
                    st.session_state.selected_label_end_min = float(clicked_bout["end_min"])
                    st.session_state.selected_period_slider_revision = (
                        st.session_state.get("selected_period_slider_revision", 0) + 1
                    )
                    # No explicit st.rerun() here.

        if "clicked_bout_start_min" in st.session_state:
            st.success(
                f"Selected scoring bout: "
                f"{st.session_state.clicked_bout_source} "
                f"{st.session_state.clicked_bout_state} | "
                f"{st.session_state.clicked_bout_start_min:.2f}–"
                f"{st.session_state.clicked_bout_end_min:.2f} min"
            )

            b1, b2 = st.columns(2)

            with b1:
                if st.button("Use selected bout as period"):
                    st.session_state.selected_label_start_min = float(st.session_state.clicked_bout_start_min)
                    st.session_state.selected_label_end_min = float(st.session_state.clicked_bout_end_min)
                    st.rerun()

            with b2:
                if st.button("Approve selected scoring bout", type="primary"):
                    source = st.session_state.clicked_bout_source
                    mode, label = mode_for_scoring_source(source)

                    result = approve_interval_from_app(
                        project_root=project_root_review,
                        recording_id=recording_id_review,
                        final_file=final_file,
                        start_s=float(st.session_state.clicked_bout_start_s),
                        end_s=float(st.session_state.clicked_bout_end_s),
                        mode=mode,
                        label=label,
                        notes=f"approved selected scoring bout {st.session_state.clicked_bout_id} from {source}",
                    )

                    if result.returncode == 0:
                        st.success("Selected scoring bout approved.")
                        st.code(result.stdout)
                        st.rerun()
                    else:
                        st.error("Could not approve selected scoring bout.")
                        st.code(result.stdout)
                        st.code(result.stderr)

        st.caption(
            "Use Plotly box-select over the scoring rows to select a whole bout, "
            "or over the raw traces/probability panel to select a manual interval."
        )

    except Exception as e:
        st.error(f"Could not plot selected interval: {repr(e)}")
        st.exception(e)


    st.markdown("---")
    st.markdown("### Video QC")

    st.write(
        "Use this to inspect the behavior/video corresponding to the current review window "
        "or the selected period you are about to label."
    )

    metadata_path_review = recording_dir_review / "metadata.json"

    try:
        metadata_for_video = json.loads(metadata_path_review.read_text())
    except Exception:
        metadata_for_video = {}

    v1, v2 = st.columns([3, 1])

    with v1:
        video_file_input = st.text_input(
            "Video file path",
            value=str(metadata_for_video.get("video_file", "")),
            key=f"review_video_file_{recording_id_review}",
            help="Recommended formats: .mp4, .mov, .m4v or .webm. AVI may not play directly in Streamlit.",
        )

    with v2:
        video_offset_s = st.number_input(
            "Video offset, seconds",
            value=float(metadata_for_video.get("video_offset_s", 0.0)),
            step=0.5,
            key=f"review_video_offset_{recording_id_review}",
            help=(
                "Offset between signal time and video time. "
                "Use positive values if the video needs to start later than the EEG/EMG time."
            ),
        )

    save_v1, save_v2, save_v3 = st.columns(3)

    with save_v1:
        if st.button("Save video settings", key=f"save_review_video_settings_{recording_id_review}"):
            if video_file_input and not Path(video_file_input).exists():
                st.error(f"Video file not found: {video_file_input}")
            else:
                metadata_for_video["video_file"] = str(video_file_input)
                metadata_for_video["video_offset_s"] = float(video_offset_s)
                metadata_path_review.write_text(json.dumps(metadata_for_video, indent=2))
                st.success("Video settings saved to metadata.json.")

    with save_v2:
        video_start_mode = st.radio(
            "Start video at",
            ["Selected period", "Visible review window"],
            horizontal=False,
            key=f"review_video_start_mode_{recording_id_review}",
        )

    with save_v3:
        st.write("")

    if not video_file_input:
        st.info("No video selected yet.")
    elif not Path(video_file_input).exists():
        st.warning(f"Video file does not exist: {video_file_input}")
    else:
        if video_start_mode == "Selected period":
            video_signal_start_s = float(st.session_state.get("selected_label_start_min", selected_start_min)) * 60
            video_signal_end_s = float(st.session_state.get("selected_label_end_min", selected_end_min)) * 60
        else:
            video_signal_start_s = float(selected_start_min) * 60
            video_signal_end_s = float(selected_end_min) * 60

        video_start_s = max(0.0, video_signal_start_s + float(video_offset_s))
        video_end_s = max(video_start_s, video_signal_end_s + float(video_offset_s))

        st.caption(
            f"Signal interval: {video_signal_start_s:.1f}–{video_signal_end_s:.1f} s | "
            f"Video starts at: {video_start_s:.1f} s"
        )

        suffix = Path(video_file_input).suffix.lower()

        if suffix in [".mp4", ".mov", ".m4v", ".webm"]:
            st.video(video_file_input, start_time=int(video_start_s))
        else:
            st.warning(
                f"{suffix} may not play in the browser. Convert the file to .mp4 for Streamlit playback."
            )


# =============================================================================
# TAB 5 — MODEL STATISTICS / DISSOCIATION
# =============================================================================
with tab_stats:
    st.subheader("Model statistics and dissociation analysis")

    st.write(
        "This section compares Manual, Somnotate, Layer 1, and App scoring, "
        "then ranks periods where the models disagree most strongly."
    )

    stats_project_root = st.text_input(
        "Project root",
        default_project,
        key="stats_project_root",
    )

    stats_manifest = load_manifest(stats_project_root)

    if stats_manifest is None:
        st.info("No recordings found yet.")
        st.stop()

    stats_recording_ids = stats_manifest["recording_id"].astype(str).tolist()

    stats_recording_id = st.selectbox(
        "Recording",
        stats_recording_ids,
        key="stats_recording_id",
    )

    stats_row = stats_manifest[
        stats_manifest["recording_id"].astype(str) == stats_recording_id
    ].iloc[0]

    stats_recording_dir = Path(stats_row["recording_dir"])

    threshold = st.slider(
        "Dissociation event threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.20,
        step=0.05,
        help="Lower values produce more candidate events. Higher values show only stronger disagreements.",
    )

    if st.button("Build / refresh dissociation analysis", type="primary"):
        cmd = [
            sys.executable,
            "pipelines/30_dissociation_analysis.py",
            "--project-root", str(stats_project_root),
            "--recording-id", str(stats_recording_id),
            "--threshold", str(threshold),
        ]

        with st.spinner("Computing model disagreement and dissociation events..."):
            result = run_command(cmd)

        if result.returncode == 0:
            st.success("Dissociation analysis complete.")
            st.code(result.stdout)
        else:
            st.error("Dissociation analysis failed.")
            st.code(result.stdout)
            st.code(result.stderr)

    analysis_dir = stats_recording_dir / "dissociation_analysis"
    pair_file = analysis_dir / "dissociation_pairwise_summary.csv"
    state_file = analysis_dir / "dissociation_state_summary.csv"
    events_file = analysis_dir / "dissociation_events.csv"
    epochs_file = analysis_dir / "dissociation_epochs.csv"

    if not analysis_dir.exists():
        st.info("No dissociation analysis found yet. Click Build / refresh dissociation analysis.")
    else:
        if pair_file.exists():
            pair = pd.read_csv(pair_file)

            st.markdown("### Pairwise disagreement")

            if len(pair):
                m1, m2, m3 = st.columns(3)

                with m1:
                    st.metric("Compared pairs", len(pair))

                with m2:
                    st.metric(
                        "Highest disagreement %",
                        f"{pair['percent_disagree'].max():.1f}%" if pair["percent_disagree"].notna().any() else "NA",
                    )

                with m3:
                    total_dis = int(pair["n_disagree_epochs"].sum()) if "n_disagree_epochs" in pair.columns else 0
                    st.metric("Total disagreement epochs", total_dis)

                fig_pair = go.Figure()
                fig_pair.add_bar(
                    x=pair["pair"],
                    y=pair["percent_disagree"],
                    text=pair["percent_disagree"].round(1),
                )
                fig_pair.update_layout(
                    height=350,
                    title="Percent disagreement by model pair",
                    xaxis_title="Model pair",
                    yaxis_title="Disagreement (%)",
                    margin=dict(l=40, r=20, t=60, b=120),
                )
                st.plotly_chart(fig_pair, width="stretch")

                st.dataframe(pair, width="stretch")

        if state_file.exists():
            state = pd.read_csv(state_file)

            st.markdown("### Which states disagree most?")

            if len(state):
                fig_state = go.Figure()

                for pair_name in state["pair"].dropna().unique():
                    sub = state[state["pair"] == pair_name].copy()
                    fig_state.add_bar(
                        x=sub["reference_state"],
                        y=sub["percent_disagree_within_state"],
                        name=pair_name,
                    )

                fig_state.update_layout(
                    height=400,
                    barmode="group",
                    title="Disagreement by reference state",
                    xaxis_title="Reference state",
                    yaxis_title="Disagreement within state (%)",
                    margin=dict(l=40, r=20, t=60, b=60),
                )
                st.plotly_chart(fig_state, width="stretch")

                st.dataframe(state, width="stretch")

        if events_file.exists():
            events = pd.read_csv(events_file)

            st.markdown("### Dissociation events ranked by severity")

            if len(events) == 0:
                st.info("No dissociation events found at the selected threshold.")
            else:
                st.dataframe(
                    events[
                        [
                            "rank",
                            "event_id",
                            "start_min",
                            "end_min",
                            "duration_s",
                            "max_dissociation_index",
                            "mean_dissociation_index",
                            "main_reason",
                            "states_at_peak",
                        ]
                    ],
                    width="stretch",
                    height=350,
                )

                event_options = events.apply(
                    lambda r: (
                        f"#{int(r['rank'])} | {r['event_id']} | "
                        f"{r['start_min']:.2f}–{r['end_min']:.2f} min | "
                        f"index={r['max_dissociation_index']:.2f} | "
                        f"{r['main_reason']}"
                    ),
                    axis=1,
                ).tolist()

                selected_event_desc = st.selectbox(
                    "Select dissociation event to review",
                    event_options,
                    key="stats_selected_dissociation_event",
                )

                selected_event = events.iloc[event_options.index(selected_event_desc)]

                if st.button("Send selected event to Review/Edit tab"):
                    st.session_state.pending_dissociation_jump = {
                        "start_min": float(selected_event["start_min"]),
                        "end_min": float(selected_event["end_min"]),
                        "event_id": str(selected_event["event_id"]),
                        "reason": str(selected_event["main_reason"]),
                    }

                    st.success(
                        "Selected event sent to Review/Edit. "
                        "Open the Review/Edit tab to inspect it."
                    )

        with st.expander("Download analysis files", expanded=False):
            for f in [pair_file, state_file, events_file, epochs_file]:
                if f.exists():
                    st.download_button(
                        label=f"Download {f.name}",
                        data=f.read_bytes(),
                        file_name=f.name,
                        mime="text/csv",
                        key=f"download_dissociation_{stats_recording_id}_{f.name}",
                    )



with tab_about:
    st.subheader("App logic")

    st.markdown(
        """
        **Overview**

        Sleep Stage QC v2 is designed to inspect, compare, correct, and export sleep-stage scoring from EEG/EMG recordings.
        It combines raw signal visualization, manual scoring, Layer 1 Wake/Sleep classification, Somnotate Wake/NREM/REM scoring,
        model confidence, disagreement detection, and final manual review.

        The main idea is that the user should be able to load a recording, inspect the EEG/EMG and model predictions,
        identify suspicious moments, correct the scoring if needed, and export a final scoring file.

        ---

        **1. Import `.mat` + Layer 1**

        This section is used to create a standardized project structure from the original recording files.

        The user provides a preprocessed `.mat` file or equivalent recording input containing the EEG and EMG signals.
        The app extracts the relevant signals, saves them into the project folder, computes basic epoch-level features,
        and runs the first classification layer.

        Layer 1 is a simple Wake/Sleep classifier. Its goal is not to distinguish NREM from REM.
        Instead, it provides a broad first pass that separates periods that look like Wake from periods that look like Sleep.

        Expected outputs include `eeg.npy`, `emg.npy`, `metadata.json`, `epoch_features.csv`, `layer1_wake_sleep.csv`,
        and optional aligned manual scoring if provided.

        Layer 1 labels are interpreted as:

        - **Wake**: the model thinks the animal is awake.
        - **Sleep**: the model thinks the animal is asleep, but does not distinguish NREM from REM.
        - **Uncertain**: the model confidence is low or the state is ambiguous.

        ---

        **2. QC viewer**

        The QC viewer is the main visual inspection tool. It shows the recording signals and scoring layers together in one synchronized view.

        The plot contains scoring layers, raw EEG, raw EMG, EMG RMS z-score, and a probability panel.
        The scoring rows can include Manual, Layer 1, Somnotate, and Final/App scoring when available.

        The state colors are:

        - **Wake**: blue.
        - **Layer 1 Sleep**: light pink.
        - **NREM**: orange.
        - **REM**: green.
        - **Uncertain**: grey.
        - **Artifact**: black or dark shading.

        The raw EEG panel shows cortical activity for the selected time window.
        The raw EMG panel shows muscle activity and is useful for distinguishing Wake from Sleep and for detecting REM periods with abnormal muscle activation.

        The EMG RMS z-score summarizes muscle tone or burst activity at the epoch level.
        The probability panel shows Layer 1 and Somnotate confidence over time, which is useful for finding transition periods and unstable predictions.

        ---

        **3. Somnotate**

        The Somnotate section supports two workflows.

        **Use an existing trained model**

        This option is for users who already have a trained Somnotate model, either trained previously by themselves or provided by someone else.

        In this workflow, the user provides the Somnotate repository path, the Somnotate Python environment, the trained model file,
        and the recordings that should be scored.

        The app prepares the selected recordings for Somnotate, runs the external Somnotate scoring pipeline, imports the resulting
        Wake/NREM/REM predictions, and displays them together with EEG, EMG, manual scoring, and Layer 1.

        This is the fastest workflow when a reliable model already exists.

        **Train a new Somnotate model**

        This option is for users who want to create their own Somnotate model from manually scored recordings.

        Training recordings must have manual scoring available. These recordings are used by Somnotate to learn Wake/NREM/REM
        classification from the EEG/EMG signal features.

        The user can also provide optional test recordings. Test recordings are not required for training, but they are useful to
        immediately evaluate how the newly trained model performs on recordings that were not part of the training set.

        In this workflow, the app helps organize and launch the external Somnotate pipeline, but Somnotate itself is still run outside
        the app logic. The user must provide the Somnotate repository folder, the Somnotate Python executable/environment,
        the training recordings, optionally test recordings, and a name for the new model.

        After training, the new model is saved in the project’s Somnotate model folder and can later be reused as an existing trained model.

        **Important note**

        The app is not replacing Somnotate. It is acting as a wrapper and visualization/QC interface around Somnotate.

        Somnotate training and scoring depend on the external Somnotate code and environment. The app’s role is to prepare compatible
        inputs, call the appropriate Somnotate scripts, import the outputs, and make the results easier to inspect and compare.

        ---

        **4. Review / Edit scoring**

        The Review/Edit section is where the user actively inspects suspicious periods and decides the final scoring.

        The user can select a period using the selected-period slider, box-selecting a region on the QC plot, clicking or selecting a dissociation event,
        jumping through the dissociation review queue, or using the visible review window as the selected period.

        Available actions include accepting Somnotate scoring, accepting manual scoring, accepting Layer 1 scoring,
        or directly assigning a manual label such as Wake, NREM, REM, Uncertain, or Artifact.

        The app keeps an undo history for recent review actions. If a scoring decision was wrong, the user can undo the last action and restore the previous scoring.

        After a period is approved or manually labelled, the app can show colored shading over the EEG/EMG plot to indicate that the period has been reviewed or modified.

        ---

        **5. Model statistics / dissociation**

        This section quantifies how much the different scoring sources disagree.

        It compares available sources such as Manual scoring, Layer 1 Wake/Sleep scoring, Somnotate Wake/NREM/REM scoring, and App/final scoring.

        The goal is to help the user find the most interesting or suspicious periods, instead of manually scanning the full recording from beginning to end.

        The app computes pairwise disagreement, state-specific disagreement, a dissociation index, and ranked dissociation events.

        The dissociation index increases when scoring sources disagree, confidence is low, REM-like periods have high EMG, or a model output is missing or uncertain.

        Neighboring high-dissociation epochs are grouped into dissociation events, which the Review/Edit section can use as a review queue.

        ---

        **Export scoring**

        The app can export the reviewed scoring as CSV and MAT files.

        There are two main export modes: app scoring only, or mixed manual + app scoring when original manual scoring exists.

        App scoring only exports the scoring currently produced inside the app.
        Mixed manual + app scoring uses app-reviewed periods where the user intervened and keeps original manual scoring elsewhere.

        The exported files include epoch times, state labels, numeric state codes, source information, and review status where available.

        ---

        **Recommended workflow**

        1. Import the recording and run Layer 1.
        2. Attach Somnotate outputs, if available.
        3. Open the QC viewer and inspect the EEG/EMG and scoring.
        4. Build the dissociation analysis.
        5. Go to Review/Edit scoring.
        6. Jump through high-dissociation events.
        7. Accept Somnotate, manual, Layer 1, or assign a manual label.
        8. Use undo if a decision was wrong.
        9. Export app scoring only or mixed manual + app scoring.
        10. Keep the exported CSV/MAT together with the original recording metadata for downstream analysis.

        ---

        **Important interpretation notes**

        Layer 1 and Somnotate have different purposes.

        Layer 1 is a broad Wake/Sleep classifier and should not be interpreted as a REM detector.
        Somnotate is a more detailed Wake/NREM/REM classifier, but uncertain or conflicting periods should still be visually checked.

        Manual scoring is not automatically assumed to be perfect.
        In this project, some REM-like or RBD-like moments may have been manually or automatically scored as Wake because of EMG activation.
        Therefore, disagreement between manual scoring, Somnotate, and Layer 1 can be biologically meaningful.

        High EMG during REM-like periods is not automatically an artifact.
        It should be reviewed carefully because it may be one of the relevant phenomena in RBD/PD-related analysis.

        The final scoring should always be interpreted as a reviewed decision, not simply as the output of one model.

        ---

        **References**

        - Layer 1: an unsupervised Gaussian mixture model (GMM) Wake/Sleep screen developed for this project.
        - Somnotate: [Brodersen PJN, Alfonsa H, Krone LB, Blanco-Duque C, Fisk AS, Flaherty SJ, et al. Somnotate: A probabilistic sleep stage classifier for studying vigilance state transitions. PLOS Computational Biology (2024).](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011793)
        - Somnotate code: [github.com/paulbrodersen/somnotate](https://github.com/paulbrodersen/somnotate)
        - This project was developed by Margarida Seabra Gomes
        """
    )
