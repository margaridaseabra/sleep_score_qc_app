"""Plotly figures for biologically interpretable sleep-scoring dissociation QC.

These are descriptive visualizations of recording epochs, NOT inferential
statistics or validation of any individual scoring source as ground truth.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

STATES = ("Wake", "NREM", "REM")
STATE_PALETTE = {"Wake": "#3178BD", "NREM": "#EA8C27", "REM": "#30A46C",
                 "Undefined": "#94A3B8", "Artifact": "#64748B"}
REASON_NAMES = {
    "manual_vs_somnotate_3state": "Manual vs Somnotate",
    "final_vs_somnotate_3state": "Final vs Somnotate",
    "final_vs_manual_3state": "Final vs Manual",
    "layer1_vs_somnotate_ws": "Layer 1 vs Somnotate (Wake/Sleep)",
    "layer1_vs_manual_ws": "Layer 1 vs Manual (Wake/Sleep)",
    "low_somnotate_confidence": "Low Somnotate confidence",
    "low_layer1_confidence": "Low Layer 1 confidence",
    "REM_with_high_EMG": "REM with high EMG",
}


def _numeric(values):
    return pd.to_numeric(values, errors="coerce")


def _state_values(df, column):
    if column not in df:
        return pd.Series("Undefined", index=df.index)
    return df[column].fillna("Undefined").astype(str).str.strip()


def _layout(fig, title, height=350):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white", height=height,
        margin=dict(l=75, r=30, t=65, b=60),
        font=dict(size=12), hovermode="closest",
    )
    return fig


def confusion_figure(epochs, row_col, col_col, row_label, col_label):
    """Percentages are conditional on the row label; all comparable epochs."""
    if epochs is None or not {row_col, col_col}.issubset(epochs.columns):
        return None
    rows = _state_values(epochs, row_col)
    cols = _state_values(epochs, col_col)
    valid = rows.isin(STATES) & cols.isin(STATES)
    if not valid.any():
        return None
    count = pd.crosstab(rows[valid], cols[valid]).reindex(
        index=list(STATES), columns=list(STATES), fill_value=0,
    )
    denominator = count.sum(axis=1).to_numpy(dtype=float)
    number = count.to_numpy(dtype=int)
    percentage = np.divide(
        number * 100.0, denominator[:, None],
        out=np.zeros_like(number, dtype=float),
        where=denominator[:, None] > 0,
    )
    labels = np.empty(number.shape, dtype=object)
    for i in range(3):
        for j in range(3):
            labels[i, j] = f"{percentage[i, j]:.1f}%<br>n={number[i, j]:,}"
    fig = go.Figure(go.Heatmap(
        z=percentage, x=list(STATES), y=list(STATES),
        customdata=number,
        colorscale=[[0, "#F4F7FB"], [0.5, "#7AADD5"], [1, "#225C8E"]],
        zmin=0, zmax=100,
        text=labels, texttemplate="%{text}",
        colorbar=dict(title="Row %", ticksuffix="%"),
        hovertemplate=f"{row_label}: %{{y}}<br>{col_label}: %{{x}}"
                      "<br>%{z:.1f}% of row (n=%{customdata})<extra></extra>",
    ))
    fig.update_xaxes(title_text=col_label, side="bottom")
    fig.update_yaxes(title_text=row_label, autorange="reversed")
    fig = _layout(fig, f"{row_label} → {col_label}: all comparable epochs", 390)
    return fig


def score_timeline_figure(epochs, events, threshold=0.20, max_points=2400):
    """Max/min-preserving downsampling keeps narrow high-scoring episodes visible."""
    required = {"t0_s", "dissociation_index"}
    if epochs is None or not required.issubset(epochs.columns) or len(epochs) == 0:
        return None
    time_s = _numeric(epochs["t0_s"]).to_numpy(dtype=float)
    score = _numeric(epochs["dissociation_index"]).to_numpy(dtype=float)
    good = np.isfinite(time_s) & np.isfinite(score)
    time_s, score = time_s[good], score[good]
    if not len(score):
        return None

    # Preserve both extremes of each interval, in their original time order.
    step = max(1, int(math.ceil(len(score) / max(1, max_points // 2))))
    selected = []
    for k in range(0, len(score), step):
        fragment = score[k:k + step]
        indices = sorted(set([k + int(np.argmin(fragment)), k + int(np.argmax(fragment))]))
        selected.extend(indices)
    idx = np.asarray(selected, dtype=int)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time_s[idx] / 3600.0, y=score[idx],
        mode="lines", line=dict(color="#63748B", width=1.4),
        name="Review-priority index",
        hovertemplate="Time: %{x:.2f} h<br>Index: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=float(threshold), line_color="#D85737", line_dash="dash",
                  annotation_text=f"Threshold {threshold:.2f}", annotation_position="top left")

    if events is not None and len(events) and "max_dissociation_index" in events:
        centers = (
            _numeric(events.get("core_start_s", events["start_s"])) +
            _numeric(events.get("core_end_s", events["end_s"]))
        ) / 7200.0
        reasons = events.get("main_reason", pd.Series("", index=events.index)).fillna("").astype(str)
        labels = [REASON_NAMES.get(x, x.replace("_", " ")) for x in reasons]
        event_ids = events.get("event_id", pd.Series("", index=events.index)).astype(str)
        fig.add_trace(go.Scatter(
            x=centers, y=_numeric(events["max_dissociation_index"]),
            mode="markers", name="Flagged episodes",
            marker=dict(color="#D85737", size=7, opacity=0.8),
            customdata=np.array(list(zip(event_ids, labels)), dtype=object),
            hovertemplate="Event %{customdata[0]}<br>%{customdata[1]}"
                          "<br>Time: %{x:.2f} h<br>Peak index: %{y:.3f}<extra></extra>",
        ))
    fig.update_xaxes(title_text="Time since recording start (hours)", rangemode="tozero")
    fig.update_yaxes(title_text="Review-priority index (0–1)", range=[0, 1.03])
    return _layout(fig, "When do scoring discrepancies occur?", 360)


def hourly_burden_figure(epochs, threshold=0.20):
    """Percent of eligible epochs flagged; an epoch is not an independent animal."""
    if epochs is None or not {"t0_s", "dissociation_index"}.issubset(epochs.columns):
        return None
    frame = epochs[["t0_s", "dissociation_index"]].copy()
    frame["t0_s"] = _numeric(frame["t0_s"])
    frame["dissociation_index"] = _numeric(frame["dissociation_index"])
    if "evidence_available" in epochs.columns:
        evidence = epochs["evidence_available"]
        if evidence.dtype == object:
            frame["eligible"] = evidence.astype(str).str.lower().isin(["true", "1"])
        else:
            frame["eligible"] = evidence.fillna(False).astype(bool)
    else:
        frame["eligible"] = frame["dissociation_index"].notna()
    frame = frame[np.isfinite(frame["t0_s"]) & frame["eligible"]].copy()
    if frame.empty:
        return None
    frame["hour"] = (frame["t0_s"] // 3600).astype(int)
    frame["flagged"] = frame["dissociation_index"] >= float(threshold)
    counts = frame.groupby("hour").agg(eligible=("flagged", "size"), flagged=("flagged", "sum"))
    counts["rate"] = 100.0 * counts["flagged"] / counts["eligible"]
    fig = go.Figure(go.Bar(
        x=counts.index.to_numpy() + 0.5, y=counts["rate"].to_numpy(),
        width=0.78, marker=dict(color="#4D91BC"),
        customdata=np.stack([counts["flagged"].to_numpy(), counts["eligible"].to_numpy()], axis=1),
        hovertemplate="Recording hour: %{x:.1f}<br>Flagged: %{customdata[0]} / %{customdata[1]}"
                      "<br>% eligible epochs: %{y:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(title_text="Time since recording start (hours)")
    fig.update_yaxes(title_text="Eligible epochs above threshold (%)", rangemode="tozero")
    return _layout(fig, "Where is review effort concentrated?", 330)


def event_raster_figure(events):
    """Core event duration (not extra video/QC padding) by the principal trigger."""
    if events is None or not len(events) or "main_reason" not in events:
        return None
    start = _numeric(events.get("core_start_s", events["start_s"])) / 3600.0
    end = _numeric(events.get("core_end_s", events["end_s"])) / 3600.0
    reasons = events["main_reason"].fillna("Unspecified").astype(str)
    labels = [REASON_NAMES.get(x, x.replace("_", " ") or "Unspecified") for x in reasons]
    frame = pd.DataFrame({"start": start, "end": end, "reason": labels})
    frame["id"] = events.get("event_id", pd.Series("", index=events.index)).astype(str).values
    frame["peak"] = _numeric(events["max_dissociation_index"]).values
    fig = go.Figure()
    colors = ["#3178BD", "#EA8C27", "#30A46C", "#995FD6", "#DB5C59",
              "#17A2A4", "#64748B", "#AF7C43"]
    for i, (reason, group) in enumerate(frame.groupby("reason", sort=False)):
        xs = []
        ys = []
        for _, row in group.iterrows():
            xs.extend([row["start"], row["end"], None])
            ys.extend([reason, reason, None])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=reason,
            line=dict(width=7, color=colors[i % len(colors)]),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=(group["start"] + group["end"]) / 2, y=[reason] * len(group),
            mode="markers", name=reason, showlegend=False,
            marker=dict(size=9, color=colors[i % len(colors)]),
            customdata=np.stack([group["id"], group["peak"], (group["end"] - group["start"]) * 3600], axis=1),
            hovertemplate="Event %{customdata[0]}<br>" + reason +
                          "<br>Peak: %{customdata[1]:.2f}<br>Core duration: %{customdata[2]:.1f} s<extra></extra>",
        ))
    fig.update_xaxes(title_text="Time since recording start (hours)", rangemode="tozero")
    fig.update_yaxes(title_text="Principal reason", automargin=True, type="category")
    return _layout(fig, "Which types of events happen, and how long do they last?",
                   max(290, 125 + 48 * frame["reason"].nunique()))


def final_coverage_figure(epochs):
    """Separate missing Final labels from genuine label disagreements."""
    if epochs is None or "final_state" not in epochs:
        return None
    state = _state_values(epochs, "final_state")
    final_valid = state.isin(STATES)
    counts = [int(final_valid.sum()), int((~final_valid).sum())]
    fig = go.Figure(go.Bar(
        x=["Defined Final", "Undefined / uncertain / artifact"],
        y=counts, marker_color=["#30A46C", "#94A3B8"],
        text=[f"{c:,} ({100*c/max(len(state),1):.1f}%)" for c in counts],
        textposition="auto",
        hovertemplate="%{x}: %{y:,} epochs<extra></extra>",
    ))
    fig.update_yaxes(title_text="Epochs", rangemode="tozero")
    return _layout(fig, "How much Final scoring is available?", 310)
