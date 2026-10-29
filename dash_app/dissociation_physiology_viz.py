"""Physiology graphics: longitudinal traces, within-recording contrasts, event drill-down."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FEATURES = [
    ("eeg_delta_z", "Delta log-power z", "#467CB4"),
    ("eeg_theta_delta_ratio_z", "Theta / delta log-ratio z", "#9254A4"),
    ("eeg_sigma_z", "Sigma log-power z", "#DA8837"),
    ("emg_rms_z", "EMG log-RMS z", "#239C94"),
    ("emg_abs_p95_z", "EMG log-|P95| z", "#5C778E"),
    ("photometry_mean_z", "Photometry epoch-mean z", "#CE5B73"),
]
FEATURE_LABELS = {key: label for key, label, _ in FEATURES}


def _numbers(df, key):
    if key not in df:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float)


def _valid_features(df):
    if df is None or df.empty:
        return []
    return [(key, title, color) for key, title, color in FEATURES
            if key in df and np.isfinite(_numbers(df, key)).any()]


def _thin_indices(t, y, max_points=2200):
    """Keep within-bin minima/maxima to avoid hiding narrow physiology peaks."""
    finite = np.isfinite(t) & np.isfinite(y)
    valid = np.flatnonzero(finite)
    if not len(valid):
        return valid
    if len(valid) <= max_points:
        return valid
    step = max(1, int(math.ceil(len(valid) / max(1, max_points // 2))))
    chosen = []
    for j in range(0, len(valid), step):
        group = valid[j:j + step]
        chosen.extend(sorted(set((group[int(np.argmin(y[group]))],
                                  group[int(np.argmax(y[group]))]))))
    return np.asarray(chosen, dtype=int)


def physiology_overview_figure(epochs, events=None):
    """Epoch physiology throughout recording, with core event intervals shaded."""
    keys = _valid_features(epochs)
    if not keys:
        return None
    t = _numbers(epochs, "t0_s") / 3600.0
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[label for _, label, _ in keys])
    for i, (key, label, color) in enumerate(keys, 1):
        y = _numbers(epochs, key)
        idx = _thin_indices(t, y)
        fig.add_trace(go.Scattergl(
            x=t[idx], y=y[idx], mode="lines", line=dict(color=color, width=1.3),
            name=label, showlegend=False,
            hovertemplate="%{x:.2f} hours<br>%{y:.2f} z<extra>" + label + "</extra>",
        ), row=i, col=1)
        fig.update_yaxes(title_text="z", row=i, col=1, zeroline=True)
    if events is not None and len(events):
        for _, ev in events.head(50).iterrows():
            left = float(ev.get("core_start_s", ev["start_s"])) / 3600.0
            right = float(ev.get("core_end_s", ev["end_s"])) / 3600.0
            if right > left:
                fig.add_vrect(x0=left, x1=right, fillcolor="rgba(230,76,80,0.15)",
                              line_width=0, layer="below", row="all", col=1)
    fig.update_xaxes(title_text="Time since recording start (hours)", row=len(keys), col=1)
    fig.update_layout(
        title="Physiology across recording (red = first 50 ranked event cores)",
        template="plotly_white", height=max(380, 165 * len(keys)),
        showlegend=False, margin=dict(l=70, r=30, t=85, b=50),
        hovermode="x",
    )
    return fig


def event_contrast_heatmap(event_physiology, events=None, max_events=30):
    """Core-event median minus unflagged, same-state background median."""
    if event_physiology is None or event_physiology.empty:
        return None
    frame = event_physiology.copy()
    if events is not None and len(events) and "rank" in events:
        frame = frame.merge(
            events[["event_id", "rank"]], on="event_id", how="left",
            validate="one_to_one", suffixes=("", "_new"),
        ).sort_values("rank", na_position="last")
    frame = frame.head(max_events)
    cols = [(key, label) for key, label, _ in FEATURES
            if "delta_" + key in frame and
            np.isfinite(_numbers(frame, "delta_" + key)).any()]
    if not cols:
        return None
    values = np.column_stack([_numbers(frame, "delta_" + col) for col, _ in cols])
    raw = np.column_stack([_numbers(frame, "event_" + col) for col, _ in cols])
    y = []
    for i, (_, ev) in enumerate(frame.iterrows()):
        rank = ev.get("rank", i + 1)
        prefix = f"#{int(rank)}" if pd.notna(rank) else f"#{i + 1}"
        center_m = (float(ev["core_start_s"]) + float(ev["core_end_s"])) / 120.0
        state = str(ev.get("reference_state", "") or "No state")
        y.append(f"{prefix} · {center_m:.1f} min · {state}")
    hovertext = np.empty(values.shape, dtype=object)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            n = int(frame.iloc[i].get("reference_n_epochs", 0))
            val = values[i, j]
            rawval = raw[i, j]
            hovertext[i, j] = (
                f"Event: {frame.iloc[i]['event_id']}<br>"
                f"Same-state reference: {n} eligible epochs<br>"
                f"Core median: {rawval:.2f} z<br>"
                f"Difference from reference: {val:+.2f} z"
            ) if np.isfinite(val) else "Insufficient matched background"
    fig = go.Figure(go.Heatmap(
        z=values, x=[label for _, label in cols], y=y,
        zmin=-3, zmax=3, zmid=0, colorscale="RdBu",
        customdata=hovertext, hoverongaps=False,
        hovertemplate="%{y}<br>%{x}<br>%{customdata}<extra></extra>",
        colorbar=dict(title="Δ z"),
    ))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(tickangle=-18, automargin=True)
    fig.update_layout(title="Event core vs unflagged same-state epochs (descriptive Δ z)",
                      template="plotly_white",
                      height=max(380, 26 * len(frame) + 190),
                      margin=dict(l=135, r=35, t=80, b=135))
    return fig


def event_detail_figure(epochs, event_row, context_s=45):
    """Zoom into physiology surrounding one event, keeping core distinct from padding."""
    if epochs is None or epochs.empty or event_row is None:
        return None
    core_start = float(event_row.get("core_start_s", event_row["start_s"]))
    core_end = float(event_row.get("core_end_s", event_row["end_s"]))
    start = max(0.0, core_start - context_s)
    end = core_end + context_s
    t0 = _numbers(epochs, "t0_s")
    t1 = _numbers(epochs, "t1_s")
    part = epochs[(t1 > start) & (t0 < end)]
    if part.empty:
        return None
    keys = _valid_features(part)
    if not keys:
        return None
    time_min = (_numbers(part, "t0_s") + _numbers(part, "t1_s")) / 120.0
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True,
                        subplot_titles=[label for _, label, _ in keys],
                        vertical_spacing=0.065)
    for i, (key, label, color) in enumerate(keys, 1):
        y = _numbers(part, key)
        fig.add_trace(go.Scatter(
            x=time_min, y=y, mode="lines+markers",
            line=dict(color=color, width=1.4), marker=dict(size=3),
            name=label, showlegend=False,
            hovertemplate="%{x:.2f} min<br>%{y:.2f} z<extra>" + label + "</extra>",
        ), row=i, col=1)
        fig.add_hline(y=0, line=dict(color="#CBD5E1", dash="dot", width=1),
                      row=i, col=1)
        fig.add_vrect(x0=core_start / 60, x1=core_end / 60,
                      fillcolor="rgba(230,76,80,0.15)", line_width=0,
                      row=i, col=1)
        fig.update_yaxes(title_text="z", row=i, col=1)
    fig.update_xaxes(title_text="Time since recording start (minutes)",
                     row=len(keys), col=1)
    fig.update_layout(title="Physiology near selected event (red = core)",
                      template="plotly_white", height=max(350, len(keys) * 165),
                      margin=dict(l=65, r=20, t=85, b=55), hovermode="x")
    return fig


def event_metric_rows(physiology_row):
    """Table values in units and with missing background represented explicitly."""
    if physiology_row is None:
        return []
    rows = []
    for name, label, _ in FEATURES:
        core = physiology_row.get("event_" + name)
        base = physiology_row.get("background_" + name)
        delta = physiology_row.get("delta_" + name)
        if pd.isna(core):
            continue
        rows.append({
            "Feature": label,
            "Core median (z)": round(float(core), 2),
            "Same-state reference (z)": round(float(base), 2) if pd.notna(base) else "Not available",
            "Difference (z)": round(float(delta), 2) if pd.notna(delta) else "Not available",
        })
    return rows
