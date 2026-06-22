
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, callback_context, Patch, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import h5py
except Exception:
    h5py = None

try:
    from scipy.io import loadmat, savemat
except Exception:
    loadmat = None
    savemat = None


APP_DIR = Path(__file__).resolve().parents[1]
PIPELINES_DIR = APP_DIR / "pipelines"
SOMNOTATE_MODELS_DIR = APP_DIR / "somnotate_models"

# -----------------------------------------------------------------------------
# Colours and labels shared across the Dash app
# -----------------------------------------------------------------------------
STATE_COLORS = {
    "Wake": "#1f77b4",
    "Layer 1 Sleep": "#f7c6d9",
    "NREM": "#ff7f0e",
    "REM": "#2ca02c",
    "Uncertain": "#9e9e9e",
    "Undefined": "#9e9e9e",
    "Artifact": "#000000",
}

STATE_TO_CODE = {
    "Artifact": -2,
    "Undefined": -1,
    "Uncertain": -1,
    "Wake": 0,
    "NREM": 1,
    "REM": 2,
    "Sleep": 1,
    "Layer 1 Sleep": 3,
}

# Separate display code so Layer 1 Sleep can be pink while real NREM remains orange.
DISPLAY_CODE_TO_COLOR = {
    -2: STATE_COLORS["Artifact"],
    -1: STATE_COLORS["Uncertain"],
    0: STATE_COLORS["Wake"],
    1: STATE_COLORS["NREM"],
    2: STATE_COLORS["REM"],
    3: STATE_COLORS["Layer 1 Sleep"],
}

FINAL_EXPORT_CODE = {
    "Wake": 0,
    "NREM": 1,
    "REM": 2,
    "Uncertain": -1,
    "Undefined": -1,
    "Artifact": -2,
    "Sleep": 1,
    "Layer 1 Sleep": 1,
}


def discrete_colorscale():
    # zmin=-2, zmax=3. Each integer gets a small interval in the colorscale.
    zmin, zmax = -2, 3
    def pos(v):
        return (v - zmin) / (zmax - zmin)
    eps = 0.0001
    scale = []
    for code in [-2, -1, 0, 1, 2, 3]:
        color = DISPLAY_CODE_TO_COLOR[code]
        left = max(0.0, pos(code - 0.49))
        right = min(1.0, pos(code + 0.49))
        scale.append([left, color])
        scale.append([right - eps, color])
    return scale


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd or APP_DIR),
            text=True,
            capture_output=True,
        )
        out = ""
        if p.stdout:
            out += p.stdout
        if p.stderr:
            out += "\n[stderr]\n" + p.stderr
        return p.returncode, out.strip()
    except Exception as e:
        return 999, repr(e)


def as_path(x: str | Path | None) -> Path:
    return Path(str(x or "")).expanduser().resolve()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def safe_mat_keys(mat_file: str | Path) -> list[str]:
    p = Path(mat_file).expanduser()
    if not p.exists():
        return []
    # MATLAB v7.3 HDF5 files
    if h5py is not None:
        try:
            with h5py.File(p, "r") as f:
                return sorted(list(f.keys()))
        except Exception:
            pass
    # Classic .mat files
    if loadmat is not None:
        try:
            d = loadmat(p, variable_names=None)
            return sorted([k for k in d.keys() if not k.startswith("__")])
        except Exception:
            pass
    return []


def load_manifest(project_root: str | Path | None) -> pd.DataFrame | None:
    if not project_root:
        return None
    p = Path(project_root).expanduser()
    manifest = p / "recordings_manifest.csv"
    if not manifest.exists():
        return None
    try:
        return pd.read_csv(manifest)
    except Exception:
        return None


def recording_dir_from_manifest(project_root: str | Path, recording_id: str) -> Path:
    manifest = load_manifest(project_root)
    if manifest is not None and len(manifest):
        m = manifest[manifest["recording_id"].astype(str) == str(recording_id)]
        if len(m) and "recording_dir" in m.columns:
            return Path(m.iloc[0]["recording_dir"]).expanduser().resolve()
    return Path(project_root).expanduser().resolve() / "recordings" / str(recording_id)


def available_recordings(project_root: str | Path | None) -> list[dict[str, str]]:
    manifest = load_manifest(project_root)
    if manifest is None or len(manifest) == 0 or "recording_id" not in manifest.columns:
        return []
    return [{"label": str(x), "value": str(x)} for x in manifest["recording_id"].astype(str).tolist()]


def available_models() -> list[dict[str, str]]:
    if not SOMNOTATE_MODELS_DIR.exists():
        return []
    models = sorted(SOMNOTATE_MODELS_DIR.glob("*.pickle"))
    return [{"label": p.name, "value": str(p)} for p in models]


def state_display_codes(labels: list[str] | np.ndarray, row_name: str = "") -> np.ndarray:
    out = []
    for x in labels:
        sx = str(x)
        if row_name == "Layer 1" and sx == "Sleep":
            out.append(3)
        else:
            out.append(STATE_TO_CODE.get(sx, -1))
    return np.asarray(out, dtype=float)


def labels_at_epoch_midpoints(epoch_df: pd.DataFrame, source_df: pd.DataFrame, label_col: str, default="Undefined") -> np.ndarray:
    mids = (epoch_df["t0_s"].to_numpy(float) + epoch_df["t1_s"].to_numpy(float)) / 2.0
    src = source_df.copy()
    if not {"t0_s", "t1_s", label_col}.issubset(src.columns):
        return np.full(len(epoch_df), default, dtype=object)
    src = src.dropna(subset=["t0_s", "t1_s"]).sort_values("t0_s")
    rows = src[["t0_s", "t1_s", label_col]].to_numpy(object)
    out = np.full(len(epoch_df), default, dtype=object)
    j = 0
    for i, mid in enumerate(mids):
        while j < len(rows) and float(rows[j][1]) <= mid:
            j += 1
        if j < len(rows):
            t0 = float(rows[j][0]); t1 = float(rows[j][1])
            if t0 <= mid < t1:
                out[i] = str(rows[j][2])
    return out


def downsample_npy_window(npy_path: Path, fs: float, start_s: float, end_s: float, max_points=70000) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(npy_path, mmap_mode="r")
    i0 = max(0, int(np.floor(start_s * fs)))
    i1 = min(len(x), int(np.ceil(end_s * fs)))
    if i1 <= i0:
        return np.array([]), np.array([])
    n = i1 - i0
    if n <= max_points:
        idx = np.arange(i0, i1)
    else:
        step = int(np.ceil(n / max_points))
        idx = np.arange(i0, i1, step)
    return idx / fs / 60.0, np.asarray(x[idx], dtype=float)


def robust_range(x: np.ndarray, low=1, high=99, pad=0.08):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return None
    a, b = np.percentile(x, [low, high])
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return None
    p = (b - a) * pad
    return [a - p, b + p]


# -----------------------------------------------------------------------------
# Recording loading
# -----------------------------------------------------------------------------
def ensure_final_scoring(recording_dir: Path, recording_id: str) -> Path:
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
    init = []
    for x in layer1["layer1_label"].fillna("Undefined").astype(str):
        if x == "Wake":
            init.append("Wake")
        elif x == "Sleep":
            init.append("NREM")
        else:
            init.append("Undefined")
    out["final_state"] = init
    out["final_code"] = [FINAL_EXPORT_CODE.get(x, -1) for x in init]
    out["final_source"] = "initial_layer1_dash"
    out["review_status"] = "not_reviewed"
    out["review_notes"] = ""
    out.to_csv(final_file, index=False)
    return final_file


def load_recording(project_root: str | Path, recording_id: str) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    recording_dir = recording_dir_from_manifest(project_root, recording_id)
    metadata = read_json(recording_dir / "metadata.json")
    fs = float(metadata["sampling_rate_hz"])
    duration_s = float(metadata["duration_s"])
    layer1 = pd.read_csv(recording_dir / "layer1_wake_sleep.csv")
    manual_file = recording_dir / "manual_scoring_aligned.csv"
    manual = pd.read_csv(manual_file) if manual_file.exists() else None
    som_file = recording_dir / "somnotate" / "somnotate_results_timeseries.csv"
    som = pd.read_csv(som_file) if som_file.exists() else None
    features_file = recording_dir / "epoch_features.csv"
    features = pd.read_csv(features_file) if features_file.exists() else None
    final_file = ensure_final_scoring(recording_dir, recording_id)
    final = pd.read_csv(final_file)
    return {
        "project_root": project_root,
        "recording_id": str(recording_id),
        "recording_dir": recording_dir,
        "metadata": metadata,
        "fs": fs,
        "duration_s": duration_s,
        "layer1": layer1,
        "manual": manual,
        "som": som,
        "features": features,
        "final": final,
    }


def scoring_rows_for_window(rec: dict[str, Any], start_min: float, end_min: float):
    layer1 = rec["layer1"].copy()
    mask = (layer1["t0_s"].astype(float) < end_min * 60.0) & (layer1["t1_s"].astype(float) > start_min * 60.0)
    epoch_df = layer1.loc[mask, ["t0_s", "t1_s"]].copy()
    if len(epoch_df) == 0:
        return [], [], [], []
    rows, names, labels_for_hover = [], [], []
    if rec["manual"] is not None:
        labels = labels_at_epoch_midpoints(epoch_df, rec["manual"], "manual_state")
        rows.append(state_display_codes(labels, "Manual")); names.append("Manual"); labels_for_hover.append(labels)
    labels = layer1.loc[mask, "layer1_label"].fillna("Uncertain").astype(str).to_numpy()
    rows.append(state_display_codes(labels, "Layer 1")); names.append("Layer 1"); labels_for_hover.append(labels)
    if rec["som"] is not None:
        labels = labels_at_epoch_midpoints(epoch_df, rec["som"], "somnotate_state")
        rows.append(state_display_codes(labels, "Somnotate")); names.append("Somnotate"); labels_for_hover.append(labels)
    labels = labels_at_epoch_midpoints(epoch_df, rec["final"], "final_state")
    rows.append(state_display_codes(labels, "Final")); names.append("Final"); labels_for_hover.append(labels)
    x = ((epoch_df["t0_s"].to_numpy(float) + epoch_df["t1_s"].to_numpy(float)) / 2.0) / 60.0
    return x, rows, names, labels_for_hover


def find_photometry(rec: dict[str, Any]) -> tuple[Path, float, str] | None:
    rd = rec["recording_dir"]
    meta = rec.get("metadata", {})
    candidates = []
    for key in ["photometry_file", "ach_file", "ne_file"]:
        if key in meta and meta[key]:
            candidates.append(rd / str(meta[key]))
            candidates.append(Path(str(meta[key])))
    for name in ["ne.npy", "ach.npy", "photometry.npy", "fiber_photometry.npy"]:
        candidates.append(rd / name)
    for p in candidates:
        try:
            if p.exists():
                fs = float(meta.get("photometry_sampling_rate_hz") or meta.get("ne_frequency") or rec["fs"])
                return p, fs, p.stem
        except Exception:
            pass
    return None


def make_review_figure(project_root: str, recording_id: str, start_min: float, window_min: float,
                       show_photometry=True, max_points=70000):
    rec = load_recording(project_root, recording_id)
    duration_min = rec["duration_s"] / 60.0
    end_min = min(duration_min, float(start_min) + float(window_min))
    fs = rec["fs"]
    t_eeg, eeg = downsample_npy_window(rec["recording_dir"] / "eeg.npy", fs, start_min * 60, end_min * 60, max_points=max_points)
    t_emg, emg = downsample_npy_window(rec["recording_dir"] / "emg.npy", fs, start_min * 60, end_min * 60, max_points=max_points)
    phot = find_photometry(rec) if show_photometry else None
    has_phot = phot is not None
    nrows = 5 if has_phot else 4
    row_heights = [0.16, 0.27, 0.22, 0.20, 0.15] if has_phot else [0.18, 0.30, 0.25, 0.27]
    titles = ["Scoring rows", "EEG", "EMG", "Probabilities / features"]
    if has_phot:
        titles.append("ACh / fiber photometry")
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.035,
                        subplot_titles=titles)
    sx, rows, names, hlabels = scoring_rows_for_window(rec, start_min, end_min)
    if len(rows):
        z = np.vstack(rows)
        custom = np.vstack(hlabels)
        fig.add_trace(go.Heatmap(
            x=sx, y=names, z=z, customdata=custom, zmin=-2, zmax=3, colorscale=discrete_colorscale(), showscale=False,
            hovertemplate="Time=%{x:.2f} min<br>Layer=%{y}<br>State=%{customdata}<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(go.Scattergl(x=t_eeg, y=eeg, mode="lines", name="Raw EEG", line=dict(width=1)), row=2, col=1)
    yrg = robust_range(eeg)
    if yrg: fig.update_yaxes(range=yrg, row=2, col=1)
    fig.add_trace(go.Scattergl(x=t_emg, y=emg, mode="lines", name="Raw EMG", line=dict(width=1)), row=3, col=1)
    yrg = robust_range(emg)
    if yrg: fig.update_yaxes(range=yrg, row=3, col=1)
    layer1 = rec["layer1"].copy()
    layer1["time_min"] = layer1["t0_s"].astype(float) / 60.0
    lm = (layer1["time_min"] >= start_min) & (layer1["time_min"] <= end_min)
    for col, label, dash in [
        ("layer1_P_Wake", "Layer 1 P(Wake)", "dash"),
        ("layer1_P_Sleep", "Layer 1 P(Sleep)", "dash"),
        ("layer1_uncertainty", "Layer 1 uncertainty", "dot"),
    ]:
        if col in layer1.columns:
            fig.add_trace(go.Scatter(x=layer1.loc[lm, "time_min"], y=layer1.loc[lm, col], mode="lines", name=label, line=dict(dash=dash)), row=4, col=1)
    if rec["som"] is not None:
        som = rec["som"].copy()
        if "time_min" not in som.columns:
            som["time_min"] = som["t0_s"].astype(float) / 60.0
        sm = (som["time_min"] >= start_min) & (som["time_min"] <= end_min)
        for col, label in [
            ("somnotate_P_Wake", "Somnotate P(Wake)"),
            ("somnotate_P_NREM", "Somnotate P(NREM)"),
            ("somnotate_P_REM", "Somnotate P(REM)"),
            ("somnotate_uncertainty", "Somnotate uncertainty"),
        ]:
            if col in som.columns:
                fig.add_trace(go.Scatter(x=som.loc[sm, "time_min"], y=som.loc[sm, col], mode="lines", name=label), row=4, col=1)
    if rec["features"] is not None:
        feat = rec["features"].copy()
        if "time_min" not in feat.columns:
            feat["time_min"] = feat["t0_s"].astype(float) / 60.0
        fm = (feat["time_min"] >= start_min) & (feat["time_min"] <= end_min)
        for col in ["emg_rms_z", "emg_rms_zscore", "log_emg_rms"]:
            if col in feat.columns:
                fig.add_trace(go.Scatter(x=feat.loc[fm, "time_min"], y=feat.loc[fm, col], mode="lines", name=col, opacity=0.55), row=4, col=1)
                break
    if has_phot:
        p, pfs, label = phot
        t_p, y_p = downsample_npy_window(p, pfs, start_min * 60, end_min * 60, max_points=max_points)
        fig.add_trace(go.Scattergl(x=t_p, y=y_p, mode="lines", name=label, line=dict(width=1)), row=5, col=1)
        yrg = robust_range(y_p)
        if yrg: fig.update_yaxes(range=yrg, row=5, col=1)
        fig.update_xaxes(title_text="Time (min)", row=5, col=1)
    else:
        fig.update_xaxes(title_text="Time (min)", row=4, col=1)
    fig.update_yaxes(range=[-0.5, 1.05], row=4, col=1)
    fig.update_layout(
        height=920,
        margin=dict(l=75, r=25, t=85, b=45),
        hovermode="x unified",
        dragmode="select",
        uirevision=f"{recording_id}-{start_min}-{window_min}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        selectdirection="h",
    )
    fig.update_xaxes(range=[start_min, end_min])
    return fig


# -----------------------------------------------------------------------------
# Editing and export
# -----------------------------------------------------------------------------
def record_undo_snapshot(recording_dir: Path, final: pd.DataFrame, mask: pd.Series, action: str):
    udir = recording_dir / "review_undo_stack"
    udir.mkdir(exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    snap = udir / f"undo_{ts}.csv"
    previous = final.loc[mask].copy()
    previous.to_csv(snap, index=False)
    registry = udir / "undo_registry.csv"
    row = pd.DataFrame([{"snapshot_file": str(snap), "action": action, "created_at": pd.Timestamp.now().isoformat(), "is_active": True}])
    if registry.exists():
        reg = pd.read_csv(registry)
        reg = pd.concat([reg, row], ignore_index=True)
    else:
        reg = row
    reg.to_csv(registry, index=False)


def apply_manual_label(project_root: str, recording_id: str, start_min: float, end_min: float, label: str):
    rec = load_recording(project_root, recording_id)
    final_file = ensure_final_scoring(rec["recording_dir"], rec["recording_id"])
    final = pd.read_csv(final_file)
    start_s, end_s = float(start_min) * 60.0, float(end_min) * 60.0
    mask = (final["t0_s"].astype(float) < end_s) & (final["t1_s"].astype(float) > start_s)
    if int(mask.sum()) == 0:
        return False, "No epochs found in selected interval."
    record_undo_snapshot(rec["recording_dir"], final, mask, f"manual {label}")
    final.loc[mask, "final_state"] = label
    final.loc[mask, "final_code"] = FINAL_EXPORT_CODE.get(label, -1)
    final.loc[mask, "final_source"] = "dash_manual"
    final.loc[mask, "review_status"] = "reviewed"
    final.loc[mask, "review_notes"] = "dash edit"
    final.to_csv(final_file, index=False)
    return True, f"Saved {label} for {int(mask.sum())} epochs."


def apply_source_label(project_root: str, recording_id: str, start_min: float, end_min: float, source_name: str):
    rec = load_recording(project_root, recording_id)
    final_file = ensure_final_scoring(rec["recording_dir"], rec["recording_id"])
    final = pd.read_csv(final_file)
    start_s, end_s = float(start_min) * 60.0, float(end_min) * 60.0
    mask = (final["t0_s"].astype(float) < end_s) & (final["t1_s"].astype(float) > start_s)
    if int(mask.sum()) == 0:
        return False, "No epochs found in selected interval."
    epoch_df = final[["t0_s", "t1_s"]].copy()
    if source_name == "Manual":
        if rec["manual"] is None: return False, "Manual scoring not found."
        source_labels = labels_at_epoch_midpoints(epoch_df, rec["manual"], "manual_state")
        final_source = "dash_accept_manual"
    elif source_name == "Somnotate":
        if rec["som"] is None: return False, "Somnotate scoring not found."
        source_labels = labels_at_epoch_midpoints(epoch_df, rec["som"], "somnotate_state")
        final_source = "dash_accept_somnotate"
    elif source_name == "Layer 1":
        source_labels = []
        for x in rec["layer1"]["layer1_label"].fillna("Undefined").astype(str):
            source_labels.append("Wake" if x == "Wake" else "NREM" if x == "Sleep" else "Undefined")
        source_labels = np.asarray(source_labels, dtype=object)
        final_source = "dash_accept_layer1"
    else:
        return False, f"Unknown source: {source_name}"
    record_undo_snapshot(rec["recording_dir"], final, mask, f"accept {source_name}")
    selected = np.asarray(source_labels, dtype=object)[mask.to_numpy()]
    final.loc[mask, "final_state"] = selected
    final.loc[mask, "final_code"] = [FINAL_EXPORT_CODE.get(str(x), -1) for x in selected]
    final.loc[mask, "final_source"] = final_source
    final.loc[mask, "review_status"] = "reviewed"
    final.loc[mask, "review_notes"] = "dash source approval"
    final.to_csv(final_file, index=False)
    return True, f"Accepted {source_name} for {int(mask.sum())} epochs."


def undo_last_action(project_root: str, recording_id: str):
    rec = load_recording(project_root, recording_id)
    udir = rec["recording_dir"] / "review_undo_stack"
    reg_file = udir / "undo_registry.csv"
    final_file = rec["recording_dir"] / "final_scoring.csv"
    if not reg_file.exists(): return False, "No undo history."
    reg = pd.read_csv(reg_file)
    active = reg[reg.get("is_active", True).astype(bool)].copy() if "is_active" in reg.columns else reg.copy()
    if len(active) == 0: return False, "No active undo snapshot."
    idx = active.index[-1]
    snap = Path(active.loc[idx, "snapshot_file"])
    if not snap.exists(): return False, "Undo snapshot file is missing."
    previous = pd.read_csv(snap)
    final = pd.read_csv(final_file)
    if "epoch_id" not in previous.columns or "epoch_id" not in final.columns:
        return False, "Cannot undo: epoch_id missing."
    prev_idx = previous.set_index("epoch_id", drop=False)
    final_idx = final.set_index("epoch_id", drop=False)
    common = final_idx.index.intersection(prev_idx.index)
    cols = [c for c in previous.columns if c in final.columns]
    final_idx.loc[common, cols] = prev_idx.loc[common, cols]
    final = final_idx.sort_index().reset_index(drop=True)
    final.to_csv(final_file, index=False)
    reg.loc[idx, "is_active"] = False
    reg.loc[idx, "undone_at"] = pd.Timestamp.now().isoformat()
    reg.to_csv(reg_file, index=False)
    return True, f"Undid: {reg.loc[idx, 'action']}"


def export_final(project_root: str, recording_id: str):
    rec = load_recording(project_root, recording_id)
    final_file = ensure_final_scoring(rec["recording_dir"], rec["recording_id"])
    final = pd.read_csv(final_file)
    out_dir = rec["recording_dir"] / "exports"
    out_dir.mkdir(exist_ok=True)
    csv_out = out_dir / f"{recording_id}_final_scoring_dash.csv"
    mat_out = out_dir / f"{recording_id}_final_scoring_dash.mat"
    final.to_csv(csv_out, index=False)
    if savemat is not None:
        savemat(mat_out, {"scoring": final["final_code"].to_numpy(dtype=np.int16), "t0_s": final["t0_s"].to_numpy(float), "t1_s": final["t1_s"].to_numpy(float)})
        return True, f"Exported:\n{csv_out}\n{mat_out}"
    return True, f"Exported CSV:\n{csv_out}\nMAT export skipped because scipy.io.savemat is unavailable."


def patch_scoring_heatmap(project_root: str, recording_id: str, window_data: dict[str, Any]):
    rec = load_recording(project_root, recording_id)
    start = float(window_data.get("start_min", 0.0))
    wmin = float(window_data.get("window_min", 15.0))
    end = min(rec["duration_s"] / 60.0, start + wmin)
    sx, rows, names, labels = scoring_rows_for_window(rec, start, end)
    patched = Patch()
    if len(rows):
        patched["data"][0]["z"] = np.vstack(rows).tolist()
        patched["data"][0]["y"] = names
        patched["data"][0]["customdata"] = np.vstack(labels).tolist()
    return patched


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
def legend_bar():
    items = [("Wake", STATE_COLORS["Wake"]), ("Layer 1 Sleep", STATE_COLORS["Layer 1 Sleep"]), ("NREM", STATE_COLORS["NREM"]), ("REM", STATE_COLORS["REM"]), ("Uncertain/Undefined", STATE_COLORS["Uncertain"]), ("Artifact", STATE_COLORS["Artifact"])]
    return html.Div([
        html.B("Scoring colours: "),
        *[html.Span([html.Span(style={"display":"inline-block","width":"13px","height":"13px","backgroundColor":c,"border":"1px solid #bbb","marginRight":"4px"}), name], style={"marginRight":"14px"}) for name, c in items]
    ], style={"fontSize":"13px", "margin":"4px 0 8px 0"})


app = Dash(__name__, suppress_callback_exceptions=True, title="Sleep Stage QC v2 Dash")

app.layout = html.Div(style={"fontFamily": "Arial", "padding": "12px"}, children=[
    html.H2("Sleep Stage QC v2 — Dash migration"),
    dcc.Store(id="project-root-store"),
    dcc.Store(id="recording-id-store"),
    dcc.Store(id="window-store", data={"start_min": 0.0, "window_min": 15.0}),
    dcc.Store(id="selected-interval-store"),
    dcc.Store(id="manifest-refresh", data=0),
    html.Div(style={"display":"flex","gap":"8px","alignItems":"center","marginBottom":"10px"}, children=[
        html.Label("Project root:"),
        dcc.Input(id="project-root-input", type="text", value=str(Path.home()/"Desktop"/"SleepStageQC_v2_Project"), style={"width":"650px"}),
        html.Button("Load project", id="load-project", n_clicks=0),
        html.Div(id="project-status", style={"fontWeight":"bold"}),
    ]),
    dcc.Tabs(id="main-tabs", value="tab-review", children=[
        dcc.Tab(label="1. Import .mat + Layer 1", value="tab-import"),
        dcc.Tab(label="2. QC / Review", value="tab-review"),
        dcc.Tab(label="3. Somnotate", value="tab-somnotate"),
        dcc.Tab(label="4. Dissociation", value="tab-stats"),
        dcc.Tab(label="About", value="tab-about"),
    ]),
    html.Div(id="tab-content", style={"paddingTop":"12px"}),
])


# -----------------------------------------------------------------------------
# Render tab contents
# -----------------------------------------------------------------------------
@app.callback(
    Output("project-root-store", "data"),
    Output("project-status", "children"),
    Input("load-project", "n_clicks"),
    State("project-root-input", "value"),
    prevent_initial_call=True,
)
def set_project_root(n, root):
    if not root:
        return no_update, "Enter a project root."
    p = as_path(root)
    p.mkdir(parents=True, exist_ok=True)
    msg = f"Project loaded: {p}"
    if not (p/"recordings_manifest.csv").exists():
        msg += "  | No manifest yet. Import a recording first."
    return str(p), msg


@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"), State("project-root-store", "data"), Input("manifest-refresh", "data"))
def render_tab(tab, project_root, _refresh):
    rec_options = available_recordings(project_root)
    if tab == "tab-import":
        return html.Div([
            html.H3("Import preprocessed .mat recording"),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"10px"}, children=[
                html.Div([html.Label(".mat file"), dcc.Input(id="mat-file", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Recording ID"), dcc.Input(id="import-recording-id", type="text", value="test_recording", style={"width":"100%"})]),
                html.Div([html.Label("Sampling rate Hz"), dcc.Input(id="import-fs", type="number", value=1017.2526, style={"width":"100%"})]),
                html.Div([html.Label("Mouse ID"), dcc.Input(id="import-mouse-id", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Group"), dcc.Input(id="import-group", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Condition"), dcc.Input(id="import-condition", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Week"), dcc.Input(id="import-week", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Epoch sec"), dcc.Input(id="import-epoch-sec", type="number", value=1.0, style={"width":"100%"})]),
            ]),
            html.Button("Detect .mat variables", id="detect-mat", n_clicks=0, style={"marginTop":"10px"}),
            html.Div(id="mat-keys-status", style={"whiteSpace":"pre-wrap", "margin":"8px 0"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"10px"}, children=[
                html.Div([html.Label("EEG variable"), dcc.Input(id="eeg-key", type="text", value="EEG", style={"width":"100%"})]),
                html.Div([html.Label("EMG variable"), dcc.Input(id="emg-key", type="text", value="EMG", style={"width":"100%"})]),
                html.Div([html.Label("Optional scoring variable"), dcc.Input(id="scoring-key", type="text", style={"width":"100%"})]),
            ]),
            html.Label("Manual scoring code map"),
            dcc.Textarea(id="code-map", value='{"0":"Wake","1":"NREM","2":"REM","15":"Wake","-1":"Undefined"}', style={"width":"100%", "height":"70px"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(3, 1fr)","gap":"8px", "marginTop":"10px"}, children=[
                html.Button("1. Import .mat", id="btn-import-mat", n_clicks=0),
                html.Button("2. Compute epoch features", id="btn-compute-features", n_clicks=0),
                html.Button("3. Run Layer 1 Wake/Sleep", id="btn-run-layer1", n_clicks=0),
            ]),
            html.Pre(id="import-log", style={"whiteSpace":"pre-wrap", "background":"#f7f7f7", "padding":"8px", "maxHeight":"360px", "overflow":"auto"}),
            html.H4("Current recordings"),
            html.Div(id="manifest-table-import")
        ])
    if tab == "tab-review":
        return html.Div([
            html.H3("QC / Review"),
            html.Div(style={"display":"flex","gap":"8px","alignItems":"center"}, children=[
                html.Label("Recording:"),
                dcc.Dropdown(id="recording-dropdown", options=rec_options, value=rec_options[0]["value"] if rec_options else None, style={"width":"360px"}),
                html.Button("Load recording", id="load-recording", n_clicks=0),
                html.Div(id="load-status"),
            ]),
            legend_bar(),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 3fr 1fr 1fr","gap":"6px","alignItems":"center", "margin":"6px 0"}, children=[
                html.Button("◀ 15 min", id="back-15"), html.Button("◀ 5 min", id="back-5"), html.Div(id="window-label", style={"textAlign":"center","fontWeight":"bold"}), html.Button("5 min ▶", id="forward-5"), html.Button("15 min ▶", id="forward-15"),
            ]),
            dcc.Graph(id="qc-graph", config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False, "modeBarButtonsToAdd": ["select2d", "pan2d", "zoom2d", "resetScale2d"]}),
            html.Div(id="selected-interval-label", style={"fontWeight":"bold", "color":"#1357c8", "margin":"8px 0"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(6, 1fr)","gap":"6px"}, children=[
                html.Button("1 = Wake", id="score-wake"), html.Button("2 = NREM", id="score-nrem"), html.Button("3 = REM", id="score-rem"), html.Button("S = Somnotate", id="score-somnotate"), html.Button("L = Layer 1", id="score-layer1"), html.Button("M = Manual", id="score-manual"),
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"6px", "marginTop":"8px"}, children=[
                html.Button("Undo last action", id="btn-undo"), html.Button("Export final scoring", id="btn-export"), html.Div("Keyboard shortcuts: 1, 2, 3, s, l, m")
            ]),
            html.Div(id="score-status", style={"marginTop":"8px", "whiteSpace":"pre-wrap"}),
        ])
    if tab == "tab-somnotate":
        models = available_models()
        return html.Div([
            html.H3("Somnotate"),
            html.P("Somnotate itself is external. This tab calls the pipeline scripts and then imports the results into the project."),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"10px"}, children=[
                html.Div([html.Label("Recording IDs, comma-separated"), dcc.Input(id="som-recording-ids", type="text", value=",".join([o["value"] for o in rec_options[:1]]), style={"width":"100%"})]),
                html.Div([html.Label("Target fs"), dcc.Input(id="som-target-fs", type="number", value=512.0, style={"width":"100%"})]),
                html.Div([html.Label("Somnotate repository path"), dcc.Input(id="som-root", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Somnotate conda env"), dcc.Input(id="som-conda-env", type="text", value="somnotate_env", style={"width":"100%"})]),
                html.Div([html.Label("Optional Somnotate Python executable"), dcc.Input(id="som-python", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Existing model"), dcc.Dropdown(id="som-model-file", options=models, value=models[0]["value"] if models else None)]),
            ]),
            html.H4("Use existing model"),
            dcc.Checklist(id="som-existing-steps", options=[{"label":x,"value":x} for x in ["prepare","preprocess","score","probabilities","import-results"]], value=["prepare","preprocess","score","probabilities","import-results"], inline=True),
            html.Button("Run existing-model workflow", id="btn-som-existing", n_clicks=0),
            html.H4("Train new model"),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"10px"}, children=[
                html.Div([html.Label("Train recording IDs"), dcc.Input(id="som-train-ids", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Test recording IDs, optional"), dcc.Input(id="som-test-ids", type="text", style={"width":"100%"})]),
                html.Div([html.Label("New model name"), dcc.Input(id="som-model-name", type="text", value="my_somnotate_model", style={"width":"100%"})]),
            ]),
            dcc.Checklist(id="som-train-steps", options=[{"label":x,"value":x} for x in ["prepare","preprocess"]], value=["prepare","preprocess"], inline=True),
            html.Button("Train new model", id="btn-som-train", n_clicks=0),
            html.H4("Import already existing local results"),
            html.Button("Import Somnotate results", id="btn-som-import-results", n_clicks=0),
            html.Pre(id="som-log", style={"whiteSpace":"pre-wrap", "background":"#f7f7f7", "padding":"8px", "maxHeight":"420px", "overflow":"auto"}),
        ])
    if tab == "tab-stats":
        return html.Div([
            html.H3("Model statistics / dissociation"),
            html.Div(style={"display":"flex","gap":"8px","alignItems":"center"}, children=[
                html.Label("Recording:"), dcc.Dropdown(id="stats-recording", options=rec_options, value=rec_options[0]["value"] if rec_options else None, style={"width":"360px"}),
                html.Label("Threshold:"), dcc.Input(id="diss-threshold", type="number", value=0.20, step=0.05, style={"width":"100px"}),
                html.Button("Run dissociation analysis", id="btn-run-diss", n_clicks=0),
            ]),
            html.Div(id="diss-log", style={"whiteSpace":"pre-wrap", "margin":"8px 0"}),
            html.H4("Pairwise disagreement"), html.Div(id="diss-pairwise"),
            html.H4("State disagreement"), html.Div(id="diss-state"),
            html.H4("Dissociation events"), html.Div(id="diss-events"),
        ])
    return html.Div([
        html.H3("About"),
        html.P("This Dash version is a migration of the Sleep Stage QC v2 Streamlit app. The first goal is to make the Review/QC viewer fast and stable."),
        html.Ul([
            html.Li("Layer 1 is an unsupervised GMM-based Wake/Sleep layer using mainly EMG-derived features."),
            html.Li("Somnotate provides Wake/NREM/REM scoring and can be used with existing trained models or trained from manually scored recordings."),
            html.Li("The Review tab updates only the scoring row when possible, avoiding expensive redraws of EEG/EMG."),
            html.Li("Shortcuts: 1 Wake, 2 NREM, 3 REM, s Somnotate, l Layer 1, m Manual."),
        ]),
    ])


# -----------------------------------------------------------------------------
# Import callbacks
# -----------------------------------------------------------------------------
@app.callback(Output("mat-keys-status", "children"), Input("detect-mat", "n_clicks"), State("mat-file", "value"), prevent_initial_call=True)
def detect_mat_vars(n, mat_file):
    keys = safe_mat_keys(mat_file)
    if not keys:
        return "Could not read variables. Check path/file."
    return "Detected variables:\n" + ", ".join(keys)


@app.callback(Output("manifest-table-import", "children"), Input("manifest-refresh", "data"), State("project-root-store", "data"))
def show_manifest_table(refresh, project_root):
    manifest = load_manifest(project_root)
    if manifest is None or len(manifest)==0:
        return "No recordings found yet."
    return dcc.Graph(figure=go.Figure(data=[go.Table(header=dict(values=list(manifest.columns)), cells=dict(values=[manifest[c] for c in manifest.columns]))]).update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10)))


@app.callback(
    Output("import-log", "children"), Output("manifest-refresh", "data"),
    Input("btn-import-mat", "n_clicks"), Input("btn-compute-features", "n_clicks"), Input("btn-run-layer1", "n_clicks"),
    State("project-root-store", "data"), State("mat-file", "value"), State("import-recording-id", "value"), State("import-fs", "value"), State("eeg-key", "value"), State("emg-key", "value"), State("scoring-key", "value"), State("import-epoch-sec", "value"), State("import-mouse-id", "value"), State("import-group", "value"), State("import-condition", "value"), State("import-week", "value"), State("code-map", "value"), State("manifest-refresh", "data"),
    prevent_initial_call=True,
)
def run_import_pipeline(n1,n2,n3,project_root,mat_file,rec_id,fs,eeg_key,emg_key,scoring_key,epoch_sec,mouse_id,group,condition,week,code_map,refresh):
    if not project_root or not rec_id:
        return "Load project and enter recording ID first.", refresh
    trigger = callback_context.triggered_id
    if trigger == "btn-import-mat":
        cmd = [sys.executable, str(PIPELINES_DIR/"01_import_mat_recording.py"), "--mat-file", str(mat_file), "--project-root", str(project_root), "--recording-id", str(rec_id), "--eeg-key", str(eeg_key), "--emg-key", str(emg_key), "--fs", str(fs), "--epoch-sec", str(epoch_sec), "--code-map", str(code_map or "{}"), "--mouse-id", str(mouse_id or ""), "--group", str(group or ""), "--condition", str(condition or ""), "--week", str(week or "")]
        if scoring_key: cmd += ["--scoring-key", str(scoring_key)]
    elif trigger == "btn-compute-features":
        cmd = [sys.executable, str(PIPELINES_DIR/"02_compute_epoch_features.py"), "--project-root", str(project_root), "--recording-id", str(rec_id), "--epoch-sec", str(epoch_sec)]
    elif trigger == "btn-run-layer1":
        cmd = [sys.executable, str(PIPELINES_DIR/"03_layer1_emg_wake_sleep.py"), "--project-root", str(project_root), "--recording-id", str(rec_id), "--epoch-sec", str(epoch_sec)]
    else:
        return no_update, refresh
    code, out = run_command(cmd)
    return f"$ {' '.join(cmd)}\n\n{out}", (refresh or 0) + 1


# -----------------------------------------------------------------------------
# Review callbacks
# -----------------------------------------------------------------------------
@app.callback(
    Output("recording-id-store", "data"), Output("qc-graph", "figure"), Output("window-label", "children"), Output("load-status", "children"), Output("window-store", "data", allow_duplicate=True), Output("selected-interval-store", "data", allow_duplicate=True),
    Input("load-recording", "n_clicks"), State("project-root-store", "data"), State("recording-dropdown", "value"), prevent_initial_call=True,
)
def load_recording_cb(n, project_root, recording_id):
    if not project_root or not recording_id:
        return no_update, no_update, no_update, "Load a project and choose a recording.", no_update, no_update
    try:
        rec = load_recording(project_root, recording_id)
        window = {"start_min": 0.0, "window_min": 15.0}
        fig = make_review_figure(project_root, recording_id, 0.0, 15.0)
        end = min(rec["duration_s"] / 60.0, 15.0)
        return recording_id, fig, f"Window: 0.00–{end:.2f} min", f"Loaded {recording_id}", window, None
    except Exception as e:
        return no_update, no_update, no_update, f"Could not load: {e}", no_update, no_update


@app.callback(
    Output("window-store", "data", allow_duplicate=True),
    Input("back-15", "n_clicks"), Input("back-5", "n_clicks"), Input("forward-5", "n_clicks"), Input("forward-15", "n_clicks"),
    State("window-store", "data"), State("project-root-store", "data"), State("recording-id-store", "data"), prevent_initial_call=True,
)
def navigate(n1,n2,n3,n4, window, project_root, recording_id):
    if not project_root or not recording_id:
        return no_update
    trig = callback_context.triggered_id
    delta = {"back-15":-15, "back-5":-5, "forward-5":5, "forward-15":15}.get(trig, 0)
    rec = load_recording(project_root, recording_id)
    duration_min = rec["duration_s"] / 60.0
    wmin = float((window or {}).get("window_min", 15.0))
    old = float((window or {}).get("start_min", 0.0))
    new = max(0.0, min(max(0.0, duration_min-wmin), old+delta))
    return {"start_min": new, "window_min": wmin}


@app.callback(Output("qc-graph", "figure", allow_duplicate=True), Output("window-label", "children", allow_duplicate=True), Output("selected-interval-store", "data", allow_duplicate=True), Input("window-store", "data"), State("project-root-store", "data"), State("recording-id-store", "data"), prevent_initial_call=True)
def update_window(window, project_root, recording_id):
    if not project_root or not recording_id: return no_update, no_update, no_update
    rec = load_recording(project_root, recording_id)
    start = float(window.get("start_min", 0.0)); wmin = float(window.get("window_min", 15.0)); end = min(rec["duration_s"]/60.0, start+wmin)
    return make_review_figure(project_root, recording_id, start, wmin), f"Window: {start:.2f}–{end:.2f} min", None


@app.callback(Output("selected-interval-store", "data"), Output("selected-interval-label", "children"), Output("qc-graph", "figure", allow_duplicate=True), Input("qc-graph", "selectedData"), State("qc-graph", "figure"), prevent_initial_call=True)
def update_selection(selected, fig):
    if not selected: return no_update, no_update, no_update
    x0 = x1 = None
    if "range" in selected:
        r = selected["range"]
        if "x" in r: x0, x1 = r["x"]
        else:
            for k, v in r.items():
                if str(k).lower().startswith("x") and isinstance(v, list) and len(v)>=2:
                    x0, x1 = v[0], v[1]; break
    if x0 is None and selected.get("points"):
        xs = [p.get("x") for p in selected["points"] if "x" in p]
        if len(xs)>=2: x0, x1 = min(xs), max(xs)
    if x0 is None or x1 is None: return no_update, "Could not read selected interval.", no_update
    x0 = float(x0); x1 = float(x1)
    if x1 < x0: x0, x1 = x1, x0
    if x1 <= x0: return no_update, "Invalid selected interval.", no_update
    patch = Patch()
    patch["layout"]["shapes"] = [{"type":"rect", "xref":"x", "yref":"paper", "x0":x0, "x1":x1, "y0":0, "y1":1, "fillcolor":"rgba(0,120,255,0.12)", "line":{"color":"rgba(0,90,220,0.95)", "width":2}, "layer":"above"}]
    dur = (x1-x0)*60
    return {"start_min":x0, "end_min":x1}, f"Selected interval: {x0:.2f}–{x1:.2f} min ({dur:.1f} s)", patch


@app.callback(Output("score-status", "children"), Output("qc-graph", "figure", allow_duplicate=True), Input("score-wake", "n_clicks"), Input("score-nrem", "n_clicks"), Input("score-rem", "n_clicks"), Input("score-somnotate", "n_clicks"), Input("score-layer1", "n_clicks"), Input("score-manual", "n_clicks"), Input("btn-undo", "n_clicks"), Input("btn-export", "n_clicks"), State("selected-interval-store", "data"), State("project-root-store", "data"), State("recording-id-store", "data"), State("window-store", "data"), prevent_initial_call=True)
def score_or_export(*args):
    selected, project_root, recording_id, window = args[-4], args[-3], args[-2], args[-1]
    if not project_root or not recording_id: return "No recording loaded.", no_update
    trig = callback_context.triggered_id
    if trig == "btn-undo":
        ok, msg = undo_last_action(project_root, recording_id)
        return msg, patch_scoring_heatmap(project_root, recording_id, window) if ok else no_update
    if trig == "btn-export":
        ok, msg = export_final(project_root, recording_id)
        return msg, no_update
    if not selected:
        return "Select an interval first. The app will not score the full visible window automatically.", no_update
    start = float(selected["start_min"]); end = float(selected["end_min"])
    if trig == "score-wake": ok, msg = apply_manual_label(project_root, recording_id, start, end, "Wake")
    elif trig == "score-nrem": ok, msg = apply_manual_label(project_root, recording_id, start, end, "NREM")
    elif trig == "score-rem": ok, msg = apply_manual_label(project_root, recording_id, start, end, "REM")
    elif trig == "score-somnotate": ok, msg = apply_source_label(project_root, recording_id, start, end, "Somnotate")
    elif trig == "score-layer1": ok, msg = apply_source_label(project_root, recording_id, start, end, "Layer 1")
    elif trig == "score-manual": ok, msg = apply_source_label(project_root, recording_id, start, end, "Manual")
    else: return "Unknown action.", no_update
    return msg, patch_scoring_heatmap(project_root, recording_id, window) if ok else no_update


# -----------------------------------------------------------------------------
# Somnotate callbacks
# -----------------------------------------------------------------------------
@app.callback(Output("som-log", "children"), Input("btn-som-existing", "n_clicks"), Input("btn-som-train", "n_clicks"), Input("btn-som-import-results", "n_clicks"), State("project-root-store", "data"), State("som-recording-ids", "value"), State("som-target-fs", "value"), State("som-root", "value"), State("som-conda-env", "value"), State("som-python", "value"), State("som-model-file", "value"), State("som-existing-steps", "value"), State("som-train-ids", "value"), State("som-test-ids", "value"), State("som-model-name", "value"), State("som-train-steps", "value"), prevent_initial_call=True)
def run_somnotate(n_exist, n_train, n_import, project_root, rec_ids, target_fs, som_root, som_env, som_py, model_file, steps, train_ids, test_ids, model_name, train_steps):
    if not project_root: return "Load project first."
    trig = callback_context.triggered_id
    base = [sys.executable, str(PIPELINES_DIR/"10_somnotate_layer.py")]
    if trig == "btn-som-existing":
        cmd = base + ["use-existing-model", "--project-root", str(project_root), "--recording-ids", str(rec_ids or ""), "--somnotate-root", str(som_root or ""), "--somnotate-conda-env", str(som_env or "somnotate_env"), "--model-file", str(model_file or ""), "--target-fs", str(target_fs or 512)]
        if som_py: cmd += ["--somnotate-python", str(som_py)]
        for s in steps or []: cmd += [f"--{s}"]
    elif trig == "btn-som-train":
        cmd = base + ["train-model", "--project-root", str(project_root), "--train-recording-ids", str(train_ids or ""), "--test-recording-ids", str(test_ids or ""), "--somnotate-root", str(som_root or ""), "--somnotate-conda-env", str(som_env or "somnotate_env"), "--model-name", str(model_name or "model"), "--target-fs", str(target_fs or 512)]
        if som_py: cmd += ["--somnotate-python", str(som_py)]
        for s in train_steps or []: cmd += [f"--{s}"]
    elif trig == "btn-som-import-results":
        cmd = base + ["import-results", "--project-root", str(project_root), "--recording-ids", str(rec_ids or "")]
    else:
        return no_update
    code, out = run_command(cmd)
    return f"$ {' '.join(cmd)}\n\n{out}"


# -----------------------------------------------------------------------------
# Dissociation callbacks
# -----------------------------------------------------------------------------
def table_from_csv(path: Path, max_rows=50):
    if not path.exists(): return "Not found."
    df = pd.read_csv(path)
    if len(df) > max_rows: df = df.head(max_rows)
    return dcc.Graph(figure=go.Figure(data=[go.Table(header=dict(values=list(df.columns)), cells=dict(values=[df[c] for c in df.columns]))]).update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10)))


@app.callback(Output("diss-log", "children"), Output("diss-pairwise", "children"), Output("diss-state", "children"), Output("diss-events", "children"), Input("btn-run-diss", "n_clicks"), State("project-root-store", "data"), State("stats-recording", "value"), State("diss-threshold", "value"), prevent_initial_call=True)
def run_diss(n, project_root, recording_id, threshold):
    if not project_root or not recording_id: return "Load project and choose recording.", no_update, no_update, no_update
    cmd = [sys.executable, str(PIPELINES_DIR/"30_dissociation_analysis.py"), "--project-root", str(project_root), "--recording-id", str(recording_id), "--threshold", str(threshold or 0.2)]
    code, out = run_command(cmd)
    analysis = recording_dir_from_manifest(project_root, recording_id) / "dissociation_analysis"
    return out, table_from_csv(analysis/"dissociation_pairwise_summary.csv"), table_from_csv(analysis/"dissociation_state_summary.csv"), table_from_csv(analysis/"dissociation_events.csv")


# -----------------------------------------------------------------------------
# Keyboard shortcuts
# -----------------------------------------------------------------------------
app.clientside_callback(
    """
    function(id) {
        if (window.__sleepDashShortcutsInstalled) { return window.dash_clientside.no_update; }
        window.__sleepDashShortcutsInstalled = true;
        document.addEventListener("keydown", function(e) {
            const tag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
            if (tag === "input" || tag === "textarea" || tag === "select") { return; }
            if (e.ctrlKey || e.metaKey || e.altKey) { return; }
            const map = {"1":"score-wake", "2":"score-nrem", "3":"score-rem", "s":"score-somnotate", "l":"score-layer1", "m":"score-manual"};
            const key = e.key.toLowerCase();
            if (map[key]) { e.preventDefault(); const btn = document.getElementById(map[key]); if (btn) { btn.click(); } }
        }, true);
        return window.dash_clientside.no_update;
    }
    """,
    Output("project-status", "data-shortcuts"),
    Input("main-tabs", "value"),
)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
