
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote
from typing import Any

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, callback_context, Patch, no_update, dash_table
from flask import abort, request, send_file
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

try:
    from scipy.signal import spectrogram as scipy_spectrogram
except Exception:
    scipy_spectrogram = None

try:
    import pyedflib
except Exception:
    pyedflib = None


APP_DIR = Path(__file__).resolve().parents[1]
PIPELINES_DIR = APP_DIR / "pipelines"


DEFAULT_PROJECT_ROOT = "/Users/margaridaseabra/Desktop/Margarida-batch2-june26"


def PInput(*args, **kwargs):
    """Persistent Dash input so fields do not reset after callbacks/tab switches."""
    kwargs.setdefault("persistence", True)
    kwargs.setdefault("persistence_type", "local")
    if kwargs.get("id") == "project-root-input":
        kwargs.setdefault("value", DEFAULT_PROJECT_ROOT)
    return dcc.Input(*args, **kwargs)


def PDropdown(*args, **kwargs):
    kwargs.setdefault("persistence", True)
    kwargs.setdefault("persistence_type", "local")
    return dcc.Dropdown(*args, **kwargs)


def PTextarea(*args, **kwargs):
    kwargs.setdefault("persistence", True)
    kwargs.setdefault("persistence_type", "local")
    return dcc.Textarea(*args, **kwargs)

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


RAW_TRACE_COLOR = "black"
EMG_RMS_COLOR = "rgba(90,90,90,0.75)"

PROB_TRACE_COLORS = {
    "Layer 1 P(Wake)": STATE_COLORS["Wake"],
    "Layer 1 P(Sleep)": STATE_COLORS["Layer 1 Sleep"],
    "Layer 1 uncertainty": STATE_COLORS["Uncertain"],
    "Somnotate P(Wake)": STATE_COLORS["Wake"],
    "Somnotate P(NREM)": STATE_COLORS["NREM"],
    "Somnotate P(REM)": STATE_COLORS["REM"],
    "Somnotate uncertainty": STATE_COLORS["Uncertain"],
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
    """
    True discrete heatmap colours for scoring rows.

    Codes:
        -2 = Artifact
        -1 = Uncertain / Undefined
         0 = Wake
         1 = NREM
         2 = REM
         3 = Layer 1 Sleep

    Important: use zmin=-2.5 and zmax=3.5 in the Heatmap.
    """
    bins = [
        (-2, STATE_COLORS["Artifact"]),
        (-1, STATE_COLORS["Uncertain"]),
        (0, STATE_COLORS["Wake"]),
        (1, STATE_COLORS["NREM"]),
        (2, STATE_COLORS["REM"]),
        (3, STATE_COLORS["Layer 1 Sleep"]),
    ]

    # Boundaries halfway between integer codes, normalized from -2.5 to 3.5.
    zmin = -2.5
    zmax = 3.5

    def norm(x):
        return (x - zmin) / (zmax - zmin)

    scale = []

    for code, color in bins:
        left = norm(code - 0.5)
        right = norm(code + 0.5)
        scale.append([max(0.0, left), color])
        scale.append([min(1.0, right), color])

    return scale



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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def video_url_for_path(video_file: str | Path | None) -> str | None:
    """Return a local Dash/Flask URL for a video path.

    Browser video elements usually cannot reliably read arbitrary local file://
    paths, so the app serves the selected file through a local Flask route.
    The video itself remains on the user's computer; it is not copied to GitHub.
    """
    if not video_file:
        return None
    return "/_local_video?path=" + quote(str(Path(str(video_file)).expanduser()))


def video_format_message(video_file: str | Path | None) -> str:
    if not video_file:
        return "No video selected yet."
    suffix = Path(str(video_file)).suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        return "Video selected. MP4/MOV playback should work in most browsers."
    if suffix == ".avi":
        return "AVI selected. The path is saved, but browser playback may fail. Convert to MP4 if it does not play."
    return f"Video selected with extension '{suffix}'. MP4 is recommended for browser playback."


def video_panel_children(video_file: str | Path | None, offset_s: float | int | str | None = 0.0):
    if not video_file:
        return html.Div("No video file saved for this recording yet.", className="app-subtitle")

    p = Path(str(video_file)).expanduser()
    exists = p.exists()
    suffix = p.suffix.lower()

    messages = []
    if not exists:
        messages.append(html.Div(f"Video file not found: {p}", className="status-line"))
    elif suffix == ".avi":
        messages.append(html.Div(
            "AVI path saved. If the .avi file is not loading in this browser, "
            "try another browser or convert the video to .mp4 using the helper command below, "
            "then save the new .mp4 path instead.",
            className="status-line",
        ))
    elif suffix not in {".mp4", ".m4v", ".mov"}:
        messages.append(html.Div(
            f"Unsupported or unusual video extension ({suffix}). MP4 is recommended.",
            className="status-line",
        ))

    if not exists:
        return html.Div(messages)

    return html.Div(children=messages + [
        html.Video(
            id="qc-video-player",
            src=video_url_for_path(p),
            controls=True,
            preload="metadata",
            style={"width": "100%", "maxHeight": "420px", "background": "#000", "borderRadius": "10px"},
        ),
        html.Div(
            f"Video offset: {float(offset_s or 0):.3f} s. Video time = recording time - offset.",
            className="app-subtitle",
            style={"marginTop": "6px"},
        ),
    ])


def load_video_metadata(project_root: str | Path | None, recording_id: str | None) -> tuple[str, float]:
    if not project_root or not recording_id:
        return "", 0.0
    try:
        rd = recording_dir_from_manifest(project_root, recording_id)
        meta = read_json(rd / "metadata.json")
        return str(meta.get("video_file", "") or ""), float(meta.get("video_offset_s", 0.0) or 0.0)
    except Exception:
        return "", 0.0


def save_video_metadata(project_root: str | Path, recording_id: str, video_file: str, video_offset_s: float) -> tuple[bool, str]:
    try:
        rd = recording_dir_from_manifest(project_root, recording_id)
        meta_path = rd / "metadata.json"
        meta = read_json(meta_path)
        video_file = str(video_file or "").strip()
        meta["video_file"] = video_file
        meta["video_offset_s"] = float(video_offset_s or 0.0)
        write_json(meta_path, meta)

        if video_file and not Path(video_file).expanduser().exists():
            return True, f"Saved video settings, but file does not exist: {video_file}"
        return True, f"Saved video settings. {video_format_message(video_file)}"
    except Exception as e:
        return False, f"Could not save video settings: {type(e).__name__}: {e}"


def next_available_mp4_path(avi_path: Path) -> Path:
    """Return a non-overwriting MP4 path next to the AVI file."""
    base = avi_path.with_suffix(".mp4")
    if not base.exists():
        return base

    candidate = avi_path.with_name(f"{avi_path.stem}_converted.mp4")
    if not candidate.exists():
        return candidate

    for i in range(2, 1000):
        candidate = avi_path.with_name(f"{avi_path.stem}_converted_{i}.mp4")
        if not candidate.exists():
            return candidate

    raise RuntimeError("Could not choose a free MP4 output filename.")


def convert_avi_to_browser_mp4(video_file: str | Path) -> tuple[bool, str, str | None]:
    """Convert an AVI file to browser-friendly H.264 MP4 using ffmpeg.

    Returns (ok, message, output_path).
    """
    raw = str(video_file or "").strip()
    if not raw:
        return False, "Choose an AVI file first.", None

    avi_path = Path(raw).expanduser().resolve()
    if not avi_path.exists() or not avi_path.is_file():
        return False, f"AVI file not found: {avi_path}", None

    if avi_path.suffix.lower() != ".avi":
        return False, "Automatic conversion is only needed for .avi files. MP4/MOV can be saved directly.", None

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return (
            False,
            "ffmpeg was not found. Install it with: brew install ffmpeg",
            None,
        )

    out_path = next_available_mp4_path(avi_path)

    cmd = [
        ffmpeg,
        "-i", str(avi_path),
        "-map", "0:v:0",
        "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        str(out_path),
    ]

    try:
        p = subprocess.run(cmd, text=True, capture_output=True)
    except Exception as e:
        return False, f"Could not run ffmpeg: {type(e).__name__}: {e}", None

    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        if len(err) > 3500:
            err = err[-3500:]
        return False, f"ffmpeg conversion failed. Terminal output:\n{err}", None

    return True, f"Converted AVI to MP4 and saved new video path:\n{out_path}", str(out_path)


def safe_float(x, default=0.0) -> float:
    try:
        if x is None or x == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)


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


def safe_edf_info(edf_file: str | Path) -> str:
    """Return a compact EDF/BDF channel summary for the import tab."""
    p = Path(edf_file).expanduser()

    if pyedflib is None:
        return (
            "EDF/BDF support needs pyedflib, but pyedflib is not importable in this "
            "environment. Install it with:\n\n"
            "pip install pyedflib\n"
            "# or conda install -c conda-forge pyedflib"
        )

    if not p.exists():
        return f"EDF/BDF file not found: {p}"

    reader = None
    try:
        reader = pyedflib.EdfReader(str(p))
        labels = list(reader.getSignalLabels())
        freqs = np.asarray(reader.getSampleFrequencies(), dtype=float)
        n_samples = np.asarray(reader.getNSamples(), dtype=int)
        duration_s = float(getattr(reader, "file_duration", 0.0) or 0.0)
        if duration_s <= 0 and len(freqs) and np.all(freqs > 0):
            duration_s = float(np.nanmax(n_samples / freqs))

        lines = [
            f"Detected EDF/BDF file: {p.name}",
            f"Signals: {len(labels)} | duration: {duration_s:.2f} s ({duration_s / 60.0:.2f} min)",
            "",
            "Channels:",
        ]

        for i, label in enumerate(labels):
            fs = freqs[i] if i < len(freqs) else np.nan
            ns = n_samples[i] if i < len(n_samples) else 0
            lines.append(f"  {i}: {label}  |  fs={fs:g} Hz  |  samples={ns}")

        try:
            ann_onsets, ann_durations, ann_text = reader.readAnnotations()
            if len(ann_text):
                preview = ", ".join([str(x) for x in ann_text[:8]])
                lines += ["", f"Annotations: {len(ann_text)} found", f"First annotations: {preview}"]
        except Exception:
            pass

        lines += [
            "",
            "Use the channel label or index in the EEG/EMG fields below, then click Import recording.",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Could not read EDF/BDF file: {type(e).__name__}: {e}"
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass


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



def scale_series_to_unit(values):
    """
    Robustly scale a trace to 0–1 so it can sit on the probability axis
    without destroying the probability scale.
    """
    y = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)

    if y.size == 0 or np.all(~np.isfinite(y)):
        return y

    lo, hi = np.nanpercentile(y, [1, 99])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(y)
        hi = np.nanmax(y)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(y)

    y = np.clip(y, lo, hi)
    return (y - lo) / (hi - lo)


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


def compute_eeg_spectrogram_window(
    npy_path: Path,
    fs: float,
    start_s: float,
    end_s: float,
    max_freq_hz: float = 30.0,
    nperseg_s: float = 4.0,
    overlap_fraction: float = 0.75,
    max_time_bins: int = 900,
):
    """Compute an EEG spectrogram for the visible review window.

    Returns x in minutes, frequency in Hz, and log-power in dB. The function is
    intentionally windowed so it stays responsive in the Dash QC viewer.
    """
    if scipy_spectrogram is None:
        return None

    try:
        x = np.load(npy_path, mmap_mode="r")
        i0 = max(0, int(np.floor(start_s * fs)))
        i1 = min(len(x), int(np.ceil(end_s * fs)))
        if i1 <= i0:
            return None

        y = np.asarray(x[i0:i1], dtype=float)
        y = y[np.isfinite(y)] if np.any(~np.isfinite(y)) else y
        if len(y) < max(64, int(fs)):
            return None

        y = y - np.nanmedian(y)

        nperseg = int(max(64, round(float(nperseg_s) * float(fs))))
        nperseg = min(nperseg, len(y))
        noverlap = int(round(nperseg * float(overlap_fraction)))
        noverlap = min(max(0, noverlap), max(0, nperseg - 1))

        f, t, sxx = scipy_spectrogram(
            y,
            fs=float(fs),
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
            mode="psd",
        )

        fmask = (f >= 0.5) & (f <= float(max_freq_hz))
        if not np.any(fmask):
            return None

        f = f[fmask]
        z = sxx[fmask, :]
        z = 10.0 * np.log10(np.maximum(z, np.finfo(float).tiny))

        finite = z[np.isfinite(z)]
        if finite.size:
            lo, hi = np.percentile(finite, [5, 95])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                z = np.clip(z, lo, hi)

        t_min = (float(start_s) + t) / 60.0

        if z.shape[1] > max_time_bins:
            step = int(np.ceil(z.shape[1] / max_time_bins))
            z = z[:, ::step]
            t_min = t_min[::step]

        return t_min, f, z
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Recording loading
# -----------------------------------------------------------------------------
def ensure_final_scoring(recording_dir: Path, recording_id: str) -> Path:
    """
    Create final_scoring.csv if missing.

    Important:
    Final scoring starts EMPTY by default.
    The app should not automatically copy Layer 1, Somnotate, or Manual scoring into Final.
    The reviewer must explicitly add labels.
    """
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

    # Empty by default
    out["final_state"] = "Undefined"
    out["final_code"] = -1
    out["final_source"] = "empty_default"
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
                fs = float(meta.get("ach_sampling_rate_hz") or meta.get("photometry_sampling_rate_hz") or meta.get("ne_frequency") or rec["fs"])
                return p, fs, p.stem
        except Exception:
            pass
    return None



def make_review_figure(
    project_root: str,
    recording_id: str,
    start_min: float,
    window_min: float,
    show_photometry=True,
    max_points=70000,
):
    rec = load_recording(project_root, recording_id)

    duration_min = rec["duration_s"] / 60.0
    end_min = min(duration_min, float(start_min) + float(window_min))
    fs = rec["fs"]

    t_eeg, eeg = downsample_npy_window(
        rec["recording_dir"] / "eeg.npy",
        fs,
        start_min * 60,
        end_min * 60,
        max_points=max_points,
    )

    t_emg, emg = downsample_npy_window(
        rec["recording_dir"] / "emg.npy",
        fs,
        start_min * 60,
        end_min * 60,
        max_points=max_points,
    )

    phot = find_photometry(rec) if show_photometry else None
    has_phot = phot is not None

    # Panel order:
    # 1 scoring rows
    # 2 raw EEG
    # 3 EEG spectrogram
    # 4 raw EMG
    # 5 ACh / photometry if available
    # last probabilities / features
    spec_row = 3
    has_spec = scipy_spectrogram is not None

    if has_phot:
        nrows = 6
        score_row = 1
        eeg_row = 2
        spec_row = 3
        emg_row = 4
        ach_row = 5
        prob_row = 6
        row_heights = [0.12, 0.20, 0.20, 0.18, 0.15, 0.15]
        titles = ["Scoring rows", "EEG", "EEG spectrogram (0.5–20 Hz)", "EMG", "ACh / fiber photometry", "Probabilities"]
    else:
        nrows = 5
        score_row = 1
        eeg_row = 2
        spec_row = 3
        emg_row = 4
        ach_row = None
        prob_row = 5
        row_heights = [0.13, 0.24, 0.24, 0.20, 0.19]
        titles = ["Scoring rows", "EEG", "EEG spectrogram (0.5–20 Hz)", "EMG", "Probabilities"]

    fig = make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=False,  # independent zoom per panel
        row_heights=row_heights,
        vertical_spacing=0.032,
        subplot_titles=titles,
    )

    # -----------------------------
    # Scoring rows
    # -----------------------------
    sx, rows, names, hlabels = scoring_rows_for_window(rec, start_min, end_min)

    if len(rows):
        z = np.vstack(rows)
        custom = np.vstack(hlabels)

        fig.add_trace(
            go.Heatmap(
                x=sx,
                y=names,
                z=z,
                customdata=custom,
                zmin=-2.5,
                zmax=3.5,
                colorscale=discrete_colorscale(),
                showscale=False,
                hovertemplate=(
                    "Time=%{x:.2f} min<br>"
                    "Layer=%{y}<br>"
                    "State=%{customdata}<extra></extra>"
                ),
            ),
            row=score_row,
            col=1,
        )

    # -----------------------------
    # Raw EEG — black
    # -----------------------------
    fig.add_trace(
        go.Scattergl(
            x=t_eeg,
            y=eeg,
            mode="lines",
            name="Raw EEG",
            line=dict(color=RAW_TRACE_COLOR, width=1),
        ),
        row=eeg_row,
        col=1,
    )

    yrg = robust_range(eeg)
    if yrg:
        fig.update_yaxes(range=yrg, row=eeg_row, col=1)

    # -----------------------------
    # EEG spectrogram
    # -----------------------------
    spec = compute_eeg_spectrogram_window(
        rec["recording_dir"] / "eeg.npy",
        fs,
        start_min * 60,
        end_min * 60,
        max_freq_hz=30.0,
    )

    if spec is not None:
        spec_t, spec_f, spec_z = spec
        fig.add_trace(
            go.Heatmap(
                x=spec_t,
                y=spec_f,
                z=spec_z,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="dB", len=0.18),
                name="EEG spectrogram",
                hovertemplate="Time=%{x:.2f} min<br>Frequency=%{y:.1f} Hz<br>Power=%{z:.1f} dB<extra></extra>",
            ),
            row=spec_row,
            col=1,
        )
        fig.update_yaxes(title_text="Hz", range=[0.5, 20.0], row=spec_row, col=1)
    else:
        fig.add_annotation(
            text="Spectrogram unavailable. Install scipy or check EEG signal length.",
            xref=f"x{spec_row}",
            yref=f"y{spec_row}",
            x=(start_min + end_min) / 2.0,
            y=15.0,
            showarrow=False,
            font=dict(size=12, color="#666"),
            row=spec_row,
            col=1,
        )
        fig.update_yaxes(title_text="Hz", range=[0.5, 20.0], row=spec_row, col=1)

    # -----------------------------
    # Raw EMG — black
    # -----------------------------
    fig.add_trace(
        go.Scattergl(
            x=t_emg,
            y=emg,
            mode="lines",
            name="Raw EMG",
            line=dict(color=RAW_TRACE_COLOR, width=1),
        ),
        row=emg_row,
        col=1,
    )

    yrg = robust_range(emg)
    if yrg:
        fig.update_yaxes(range=yrg, row=emg_row, col=1)

    # -----------------------------
    # ACh / photometry — black, before probabilities
    # -----------------------------
    if has_phot:
        p, pfs, label = phot

        t_p, y_p = downsample_npy_window(
            p,
            pfs,
            start_min * 60,
            end_min * 60,
            max_points=max_points,
        )

        fig.add_trace(
            go.Scattergl(
                x=t_p,
                y=y_p,
                mode="lines",
                name=label,
                line=dict(color=RAW_TRACE_COLOR, width=1),
            ),
            row=ach_row,
            col=1,
        )

        yrg = robust_range(y_p)
        if yrg:
            fig.update_yaxes(range=yrg, row=ach_row, col=1)

    # -----------------------------
    # Probability / features panel
    # -----------------------------
    layer1 = rec["layer1"].copy()
    layer1["time_min"] = layer1["t0_s"].astype(float) / 60.0
    lm = (layer1["time_min"] >= start_min) & (layer1["time_min"] <= end_min)

    layer1_traces = [
        (["layer1_P_Wake", "p_wake", "P_Wake"], "Layer 1 P(Wake)", "dash"),
        (["layer1_P_Sleep", "p_sleep", "P_Sleep"], "Layer 1 P(Sleep)", "dash"),
        (["layer1_uncertainty", "uncertainty"], "Layer 1 uncertainty", "dot"),
    ]

    for candidates, label, dash in layer1_traces:
        col = next((c for c in candidates if c in layer1.columns), None)

        if col is not None:
            fig.add_trace(
                go.Scatter(
                    x=layer1.loc[lm, "time_min"],
                    y=layer1.loc[lm, col],
                    mode="lines",
                    name=label,
                    line=dict(
                        dash=dash,
                        color=PROB_TRACE_COLORS.get(label, None),
                        width=2,
                    ),
                ),
                row=prob_row,
                col=1,
            )

    if rec["som"] is not None:
        som = rec["som"].copy()

        if "time_min" not in som.columns:
            som["time_min"] = som["t0_s"].astype(float) / 60.0

        sm = (som["time_min"] >= start_min) & (som["time_min"] <= end_min)

        som_traces = [
            (["somnotate_P_Wake", "p_wake", "P_Wake"], "Somnotate P(Wake)"),
            (["somnotate_P_NREM", "p_nrem", "P_NREM"], "Somnotate P(NREM)"),
            (["somnotate_P_REM", "p_rem", "P_REM"], "Somnotate P(REM)"),
            (["somnotate_uncertainty", "uncertainty"], "Somnotate uncertainty"),
        ]

        for candidates, label in som_traces:
            col = next((c for c in candidates if c in som.columns), None)

            if col is not None:
                fig.add_trace(
                    go.Scatter(
                        x=som.loc[sm, "time_min"],
                        y=som.loc[sm, col],
                        mode="lines",
                        name=label,
                        line=dict(
                            color=PROB_TRACE_COLORS.get(label, None),
                            width=2,
                        ),
                    ),
                    row=prob_row,
                    col=1,
                )

    # EMG RMS feature trace removed from probability panel for clarity.

    # -----------------------------
    # Light scoring-colour background over raw traces
    # -----------------------------
    raw_rows_for_background = [eeg_row, emg_row]
    if has_phot and ach_row is not None:
        raw_rows_for_background.append(ach_row)

    fig = add_scoring_background_to_raw_panels(
        fig,
        rec,
        start_min=start_min,
        end_min=end_min,
        raw_rows=raw_rows_for_background,
        source="Final",
    )

    # -----------------------------
    # Layout
    # -----------------------------
    fig.update_layout(
        height=1120,
        margin=dict(l=75, r=25, t=95, b=45),
        hovermode="x unified",
        dragmode="pan",
        uirevision=f"{recording_id}-{start_min}-{window_min}",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        selectdirection="h",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Independent x axes, same starting range.
    fig.update_xaxes(matches=None)

    for rr in range(1, nrows + 1):
        fig.update_xaxes(range=[start_min, end_min], row=rr, col=1)
        fig.update_yaxes(fixedrange=False, row=rr, col=1)

    fig.update_xaxes(title_text="Time (min)", row=nrows, col=1)

    # Probability panel should remain interpretable:
    # probabilities and scaled features live on 0–1.
    fig.update_yaxes(
        range=[-0.05, 1.05],
        title_text="Probability",
        row=prob_row,
        col=1,
    )

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



def reset_final_to_empty(project_root: str, recording_id: str):
    """Reset the full Final scoring row to Undefined/empty."""
    rec = load_recording(project_root, recording_id)
    final_file = ensure_final_scoring(rec["recording_dir"], rec["recording_id"])
    final = pd.read_csv(final_file)
    mask = pd.Series(True, index=final.index)
    record_undo_snapshot(rec["recording_dir"], final, mask, "reset final empty")
    final["final_state"] = "Undefined"
    final["final_code"] = -1
    final["final_source"] = "empty_reset"
    final["review_status"] = "not_reviewed"
    final["review_notes"] = ""
    final.to_csv(final_file, index=False)
    return True, f"Reset Final scoring to empty for {len(final)} epochs."

def fill_empty_final_with_somnotate(project_root: str, recording_id: str, export_after: bool = False):
    """Fill only empty/Undefined Final epochs with Somnotate labels.

    Existing reviewed labels are preserved. This is useful when the user wants
    an empty recording to start from Somnotate, while still keeping manual edits
    safe.
    """
    rec = load_recording(project_root, recording_id)
    if rec.get("som") is None:
        return False, "Somnotate scoring not found for this recording."

    final_file = ensure_final_scoring(rec["recording_dir"], rec["recording_id"])
    final = pd.read_csv(final_file)
    epoch_df = final[["t0_s", "t1_s"]].copy()
    source_labels = labels_at_epoch_midpoints(epoch_df, rec["som"], "somnotate_state")
    source_labels = np.asarray(source_labels, dtype=object)

    state = final.get("final_state", pd.Series("Undefined", index=final.index)).fillna("Undefined").astype(str)
    code = pd.to_numeric(final.get("final_code", pd.Series(-1, index=final.index)), errors="coerce").fillna(-1)
    empty_mask = state.isin(["", "Undefined", "Uncertain", "nan", "None"]) | (code < 0)

    valid_source = pd.Series(source_labels).fillna("Undefined").astype(str)
    valid_mask = ~valid_source.isin(["", "Undefined", "Uncertain", "nan", "None"])
    mask = empty_mask & valid_mask.to_numpy()

    if int(mask.sum()) == 0:
        return False, "No empty Final epochs with valid Somnotate labels were found."

    record_undo_snapshot(rec["recording_dir"], final, mask, "fill empty final with Somnotate")
    selected = source_labels[mask.to_numpy()]
    final.loc[mask, "final_state"] = selected
    final.loc[mask, "final_code"] = [FINAL_EXPORT_CODE.get(str(x), -1) for x in selected]
    final.loc[mask, "final_source"] = "dash_fill_empty_somnotate"
    final.loc[mask, "review_status"] = "auto_filled"
    final.loc[mask, "review_notes"] = "empty final filled from Somnotate"
    final.to_csv(final_file, index=False)

    msg = f"Filled {int(mask.sum())} empty Final epochs with Somnotate. Existing reviewed labels were preserved."
    if export_after:
        ok_export, export_msg = export_final(project_root, recording_id)
        msg += "\n" + export_msg
    return True, msg


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



def refresh_qc_figure_after_scoring(project_root, recording_id, window_data):
    """
    Redraw the full QC figure after scoring.

    This is needed because the faint colours over the raw traces are drawn
    from Final scoring. If we only patch the scoring row, the raw-signal
    background does not update.
    """
    if not project_root or not recording_id:
        return no_update

    window_data = window_data or {}
    start = float(window_data.get("start_min", 0.0))
    wmin = float(window_data.get("window_min", 15.0))

    return make_review_figure(project_root, recording_id, start, wmin)


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


app = Dash(__name__, suppress_callback_exceptions=True, title="Semi-automated sleep scoring QC app")


@app.server.route("/_local_video")
def serve_local_video():
    """Serve a local video file to the browser through Dash/Flask.

    This allows the Dash video player to display videos that live outside the
    repository/project folder. The app is intended for local lab use.
    """
    raw_path = request.args.get("path", "")
    if not raw_path:
        abort(404)

    path = Path(raw_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        abort(404)

    suffix = path.suffix.lower()
    if suffix not in {".mp4", ".m4v", ".mov", ".avi"}:
        abort(415)

    mimetype = {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
    }.get(suffix, "application/octet-stream")

    return send_file(path, mimetype=mimetype, conditional=True, as_attachment=False)


app.layout = html.Div(
    id="app-shell",
    className="theme-light",
    children=[
        dcc.Store(id="theme-store", data="light"),
        dcc.Store(id="project-root-store"),
        dcc.Store(id="recording-id-store"),
        dcc.Store(id="window-store", data={"start_min": 0.0, "window_min": 15.0}),
        dcc.Store(id="selected-interval-store"),
        dcc.Store(id="manifest-refresh", data=0),

        html.Div(
            className="app-header",
            children=[
                html.Div([
                    html.H1("Semi-automated sleep scoring QC app", className="app-title"),
                    html.Div(
                        "Interactive review, model comparison, dissociation QC and export for EEG/EMG sleep scoring.",
                        className="app-subtitle",
                    ),
                ]),
                html.Div([
                    html.Label("Theme"),
                    dcc.RadioItems(
                        id="theme-choice",
                        options=[{"label": "Light", "value": "light"}, {"label": "Dark", "value": "dark"}],
                        value="light",
                        inline=True,
                        persistence=True,
                        persistence_type="local",
                    ),
                ]),
            ],
        ),

        html.Div(className="card", children=[
            html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center"}, children=[
                html.Label("Project root:"),
                PInput(id="project-root-input", type="text", value=DEFAULT_PROJECT_ROOT, style={"width": "720px"}),
                html.Button("Load project", id="load-project", n_clicks=0),
            ]),
            html.Div(id="project-status", className="status-line"),
        ]),

        dcc.Tabs(id="main-tabs", value="tab-review", children=[
            dcc.Tab(label="1. Import .mat / EDF + Layer 1", value="tab-import"),
            dcc.Tab(label="2. QC / Review", value="tab-review"),
            dcc.Tab(label="3. Somnotate", value="tab-somnotate"),
            dcc.Tab(label="4. Dissociation", value="tab-stats"),
            dcc.Tab(label="About", value="tab-about"),
        ]),

        html.Div(
            className="qc-mode-bar",
            children=[
                html.Div("QC mouse mode:", style={"fontWeight": "700"}),
                html.Button("Pan / move recording", id="global-qc-mode-pan", n_clicks=0),
                html.Button("Select window for scoring", id="global-qc-mode-select-window", n_clicks=0),
                html.Div(
                    id="global-qc-mode-status",
                    className="status-line",
                    children="Pan is active by default. Press Select window for scoring, then drag on the QC plot.",
                ),
            ],
        ),
        html.Div(id="tab-content", style={"paddingTop": "12px"}),
    ],
)



# -----------------------------------------------------------------------------
# Validation layout for dynamic tabs
# -----------------------------------------------------------------------------
# Dash callbacks can reference components that are only visible in some tabs.
# This validation_layout tells Dash that these IDs are valid even when their tab
# is not currently rendered.
app.validation_layout = html.Div([
    dcc.Store(id="theme-store"),
    dcc.Store(id="project-root-store"),
    dcc.Store(id="recording-id-store"),
    dcc.Store(id="window-store"),
    dcc.Store(id="selected-interval-store"),
    dcc.Store(id="manifest-refresh"),
    dcc.RadioItems(id="theme-choice"),
    PInput(id="project-root-input"),
    html.Button(id="load-project"),
    html.Div(id="project-status"),
    dcc.Tabs(id="main-tabs"),
    html.Div(id="tab-content"),

    # Import tab
    PInput(id="mat-file"), PInput(id="import-recording-id"), PInput(id="import-fs"),
    PInput(id="import-mouse-id"), PInput(id="import-group"), PInput(id="import-condition"),
    PInput(id="import-week"), PInput(id="import-epoch-sec"),
    html.Button(id="detect-mat"), html.Div(id="mat-keys-status"),
    PInput(id="eeg-key"), PInput(id="emg-key"), PInput(id="ach-key"),
    PInput(id="eeg-fs-key"), PInput(id="ach-fs-key"), PInput(id="scoring-key"),
    PTextarea(id="code-map"),
    html.Button(id="btn-import-mat"), html.Button(id="btn-compute-features"), html.Button(id="btn-run-layer1"),
    html.Div(id="import-action-status"), html.Pre(id="import-log"), html.Div(id="manifest-table-import"),

    # QC tab
    PDropdown(id="recording-dropdown"), html.Button(id="load-recording"), html.Div(id="load-status"),
    html.Div(id="empty-qc-message"), html.Button(id="back-15"), html.Button(id="back-5"),
    html.Button(id="forward-5"), html.Button(id="forward-15"), html.Div(id="window-label"),
    html.Button(id="qc-refresh-diss-events"), html.Button(id="qc-prev-diss-event"),
    PDropdown(id="qc-diss-event-dropdown"), html.Button(id="qc-next-diss-event"), html.Div(id="qc-diss-event-status"),
    
                html.Div(className="qc-mode-bar", children=[
                    html.Div("Mouse mode:", style={"fontWeight": "700"}),
                    html.Button("Pan / move recording", id="qc-mode-pan", n_clicks=0),
                    html.Button("Select window for scoring", id="qc-mode-select-window", n_clicks=0),
                    html.Div(id="qc-mode-status", className="status-line"),
                ]),
dcc.Graph(id="qc-graph"),
    html.Button(id="qc-mode-pan"),
    html.Button(id="qc-mode-select-window"),
    html.Div(id="qc-mode-status"),
    dcc.RangeSlider(id="qc-window-range-slider"),
    html.Div(id="qc-window-range-label"),
    html.Div(id="selected-interval-label"),
    html.Button(id="score-wake"), html.Button(id="score-nrem"), html.Button(id="score-rem"),
    html.Button(id="score-somnotate"), html.Button(id="score-layer1"), html.Button(id="score-manual"),
    html.Button(id="score-window-somnotate"), html.Button(id="score-window-layer1"), html.Button(id="score-window-manual"),
    html.Button(id="btn-reset-final-empty"), html.Button(id="btn-undo"), html.Button(id="btn-export"),
    html.Button(id="btn-fill-empty-somnotate"), html.Button(id="btn-fill-empty-somnotate-export"), html.Button(id="btn-export-bottom"),
    html.Div(id="score-status"),
    PInput(id="video-file-input"), PInput(id="video-offset-input"), html.Button(id="save-video-settings"),
    html.Button(id="jump-video-window"), html.Button(id="jump-video-selected"), html.Button(id="convert-video-mp4"),
    html.Div(id="video-status"), html.Div(id="video-player-container"),
    dcc.Store(id="video-seek-store"), html.Div(id="video-seek-feedback"),

    # Somnotate tab
    PInput(id="som-recording-ids"), PInput(id="som-target-fs"), PInput(id="som-root"),
    PInput(id="som-conda-env"), PInput(id="som-python"), PDropdown(id="som-model-file"),
    dcc.Checklist(id="som-existing-steps"), html.Button(id="btn-som-existing"),
    PInput(id="som-train-ids"), PInput(id="som-test-ids"), PInput(id="som-model-name"),
    dcc.Checklist(id="som-train-steps"), html.Button(id="btn-som-train"), html.Button(id="btn-som-import-results"),
    html.Div(id="som-action-status"), html.Pre(id="som-log"),

    # Dissociation tab
    PDropdown(id="stats-recording"), PInput(id="diss-threshold"), html.Button(id="btn-run-diss"),
    html.Div(id="diss-action-status"), html.Div(id="diss-log"), html.Div(id="diss-pairwise"),
    html.Div(id="diss-state"), html.Div(id="diss-events"),
])



# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
@app.callback(
    Output("theme-store", "data"),
    Output("app-shell", "className"),
    Input("theme-choice", "value"),
)
def set_app_theme(theme):
    theme = theme or "light"
    if theme == "dark":
        return "dark", "theme-dark"
    return "light", "theme-light"


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
        return html.Div(className="card", children=[
            html.H3("Import recording (.mat or EDF/BDF)"),
            html.Div("Fields are persistent, so clicking buttons should not clear your paths.", className="app-subtitle"),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"10px"}, children=[
                html.Div([html.Label("Recording file (.mat, .edf, .bdf)"), PInput(id="mat-file", type="text", placeholder="/full/path/to/file.mat or /full/path/to/file.edf", style={"width":"100%"})]),
                html.Div([html.Label("Recording ID"), PInput(id="import-recording-id", type="text", value="test_recording", style={"width":"100%"})]),
                html.Div([html.Label("Sampling rate Hz (MAT fallback; EDF reads this automatically)"), PInput(id="import-fs", type="number", value=1017.2526, style={"width":"100%"})]),
                html.Div([html.Label("Mouse ID"), PInput(id="import-mouse-id", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Group"), PInput(id="import-group", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Condition"), PInput(id="import-condition", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Week"), PInput(id="import-week", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Epoch sec"), PInput(id="import-epoch-sec", type="number", value=1.0, style={"width":"100%"})]),
            ]),
            html.Button("Detect variables / EDF channels", id="detect-mat", n_clicks=0, style={"marginTop":"10px"}),
            html.Div(id="mat-keys-status", className="status-line", style={"whiteSpace":"pre-wrap"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"10px", "marginTop":"8px"}, children=[
                html.Div([html.Label("EEG variable or EDF channel"), PInput(id="eeg-key", type="text", value="eeg", style={"width":"100%"})]),
                html.Div([html.Label("EMG variable or EDF channel"), PInput(id="emg-key", type="text", value="emg", style={"width":"100%"})]),
                html.Div([html.Label("ACh / photometry variable or EDF channel, optional"), PInput(id="ach-key", type="text", value="ne", style={"width":"100%"})]),
                html.Div([html.Label("EEG sampling frequency variable, optional"), PInput(id="eeg-fs-key", type="text", value="eeg_frequency", style={"width":"100%"})]),
                html.Div([html.Label("ACh sampling frequency variable, optional"), PInput(id="ach-fs-key", type="text", value="ne_frequency", style={"width":"100%"})]),
                html.Div([html.Label("Optional scoring variable"), PInput(id="scoring-key", type="text", style={"width":"100%"})]),
            ]),
            html.Label("Manual scoring code map"),
            PTextarea(id="code-map", value='{"0":"Wake","1":"NREM","2":"REM","15":"Wake","-1":"Undefined"}', style={"width":"100%", "height":"70px"}),
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(3, 1fr)","gap":"8px", "marginTop":"10px"}, children=[
                html.Button("1. Import recording", id="btn-import-mat", n_clicks=0),
                html.Button("2. Compute epoch features", id="btn-compute-features", n_clicks=0),
                html.Button("3. Run Layer 1 Wake/Sleep", id="btn-run-layer1", n_clicks=0),
            ]),
            html.Div(id="import-action-status", className="status-line"),
            dcc.Loading(type="circle", children=html.Pre(id="import-log", className="log-box")),
            html.H4("Current recordings"),
            html.Div(id="manifest-table-import"),
        ])

    if tab == "tab-review":
        return html.Div([
            html.Div(className="card", children=[
                html.H3("QC / Review"),
                html.Div(style={"display":"flex","gap":"8px","alignItems":"center", "flexWrap":"wrap"}, children=[
                    html.Label("Recording:"),
                    PDropdown(id="recording-dropdown", options=rec_options, value=rec_options[0]["value"] if rec_options else None, style={"width":"360px"}),
                    html.Button("Load recording", id="load-recording", n_clicks=0),
                    html.Div(id="load-status", className="status-line"),
                ]),
                legend_bar(),
                html.Div(className="queue-box", children=[
                    html.H4("Dissociation review queue"),
                    html.Div("Run dissociation analysis first, then refresh here to jump through the most suspicious events.", className="app-subtitle"),
                    html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr 3fr 1fr", "gap":"8px", "alignItems":"center", "marginTop":"8px"}, children=[
                        html.Button("Refresh events", id="qc-refresh-diss-events", n_clicks=0),
                        html.Button("Previous", id="qc-prev-diss-event", n_clicks=0),
                        PDropdown(id="qc-diss-event-dropdown", options=[], placeholder="Choose dissociation event", style={"width":"100%"}),
                        html.Button("Next", id="qc-next-diss-event", n_clicks=0),
                    ]),
                    html.Div(id="qc-diss-event-status", className="status-line"),
                ]),

                html.Div(className="timeline-card", children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                        children=[
                            html.Div("Recording position", style={"fontWeight": "700"}),
                            html.Div("Drag the highlighted window to move through the recording.", className="app-subtitle"),
                        ],
                    ),
                    dcc.RangeSlider(
                        id="qc-window-range-slider",
                        min=0,
                        max=1,
                        step=0.25,
                        value=[0, 1],
                        allowCross=False,
                        disabled=True,
                        marks={0: "0", 1: "1"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(id="qc-window-range-label", className="status-line"),
                ]),

                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 3fr 1fr 1fr","gap":"6px","alignItems":"center", "margin":"8px 0"}, children=[
                    html.Button("◀ 15 min", id="back-15"), html.Button("◀ 5 min", id="back-5"),
                    html.Div(id="window-label", style={"textAlign":"center","fontWeight":"bold"}),
                    html.Button("5 min ▶", id="forward-5"), html.Button("15 min ▶", id="forward-15"),
                ]),

                html.Div(id="empty-qc-message", children=[html.H4("No recording loaded yet"), html.P("Load a project and choose a recording first.")], className="empty-panel"),
                dcc.Graph(id="qc-graph", style={"display":"none"}, config={"scrollZoom": False, "displayModeBar": True, "displaylogo": False, "modeBarButtonsToAdd": ["select2d", "pan2d", "zoom2d", "resetScale2d"]}),
                html.Div(id="selected-interval-label", className="status-line"),

                html.H4("Apply source to whole visible window"),
                html.Div(style={"display":"grid", "gridTemplateColumns":"repeat(3, 1fr)", "gap":"6px", "marginBottom":"12px"}, children=[
                    html.Button("Apply Somnotate to visible window", id="score-window-somnotate"),
                    html.Button("Apply Layer 1 to visible window", id="score-window-layer1"),
                    html.Button("Apply Manual to visible window", id="score-window-manual"),
                ]),

                html.Div(className="video-qc-card", children=[
                    html.H4("Video QC"),
                    html.Div(
                        "Optional: link an .mp4/.mov/.avi video to this recording. MP4 is the most reliable browser format. If an .avi file does not load in the browser, try another browser or convert the .avi video to .mp4 with the Terminal command below.",
                        className="app-subtitle",
                    ),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "3fr 1fr 1fr", "gap": "8px", "alignItems": "end", "marginTop": "8px"},
                        children=[
                            html.Div([html.Label("Video file path"), PInput(id="video-file-input", type="text", placeholder="/full/path/to/video.mp4 or .avi", style={"width": "100%"})]),
                            html.Div([html.Label("Video offset (s)"), PInput(id="video-offset-input", type="number", value=0.0, step=0.1, style={"width": "100%"})]),
                            html.Button("Save video", id="save-video-settings", n_clicks=0),
                        ],
                    ),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1.4fr 2fr", "gap": "8px", "alignItems": "center", "marginTop": "8px"},
                        children=[
                            html.Button("Jump video to window start", id="jump-video-window", n_clicks=0),
                            html.Button("Play selected video interval", id="jump-video-selected", n_clicks=0),
                            html.Button("Convert AVI to MP4 + save path", id="convert-video-mp4", n_clicks=0),
                            dcc.Loading(type="circle", children=html.Div(id="video-status", className="status-line")),
                        ],
                    ),
                    html.Div(
                        "During conversion, keep the app tab open. Large AVI files can take several minutes; the button shows a spinner while ffmpeg is running.",
                        className="app-subtitle",
                        style={"marginTop": "6px"},
                    ),
                    html.Div(id="video-player-container", style={"marginTop": "10px"}),
                    dcc.Store(id="video-seek-store"),
                    html.Div(id="video-seek-feedback", className="status-line"),
                    html.Details(children=[
                        html.Summary("AVI not loading? Convert AVI to MP4"),
                        html.Div(
                            "If your .avi file is not loading in this browser, click “Convert AVI to MP4 + save path” above. You can also do the same conversion manually in Terminal with this command. Replace videoname.avi and videoname.mp4 with your real file names or full paths.",
                            className="app-subtitle",
                            style={"marginTop": "6px", "marginBottom": "6px"},
                        ),
                        html.Pre(
                            "\n".join([
                                'ffmpeg -i "videoname.avi" \\',
                                '  -map 0:v:0 \\',
                                '  -an \\',
                                '  -c:v libx264 \\',
                                '  -pix_fmt yuv420p \\',
                                '  -preset fast \\',
                                '  -crf 23 \\',
                                '  -movflags +faststart \\',
                                '  "videoname.mp4"',
                            ]),
                            className="log-box",
                        ),
                    ]),
                ]),

                html.H4("Apply label to selected interval"),
                html.Div(style={"display":"grid", "gridTemplateColumns":"repeat(6, 1fr)", "gap":"6px"}, children=[
                    html.Button("1 = Wake", id="score-wake"), html.Button("2 = NREM", id="score-nrem"), html.Button("3 = REM", id="score-rem"),
                    html.Button("A = Somnotate", id="score-somnotate"), html.Button("L = Layer 1", id="score-layer1"), html.Button("M = Manual", id="score-manual"),
                ]),
                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr 2fr","gap":"6px", "marginTop":"8px"}, children=[
                    html.Button("Undo last action", id="btn-undo"),
                    html.Button("Export final scoring", id="btn-export"),
                    html.Button("Reset Final to empty", id="btn-reset-final-empty"),
                    html.Div("Shortcuts: P Pan, S Select window, 1 Wake, 2 NREM, 3 REM, A Somnotate/automatic, L Layer 1, M Manual"),
                ]),
                html.Div(className="card", style={"marginTop": "16px"}, children=[
                    html.H4("Final scoring utilities"),
                    html.Div(
                        "Use these at the end of review, or to initialise an empty Final row from Somnotate. Existing reviewed labels are not overwritten.",
                        className="app-subtitle",
                    ),
                    html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr 1fr", "gap":"8px", "marginTop":"8px"}, children=[
                        html.Button("Fill empty Final with Somnotate", id="btn-fill-empty-somnotate", n_clicks=0),
                        html.Button("Fill empty Final with Somnotate + export", id="btn-fill-empty-somnotate-export", n_clicks=0),
                        html.Button("Export final scoring", id="btn-export-bottom", n_clicks=0),
                    ]),
                ]),
                html.Div(id="score-status", className="status-line", style={"whiteSpace":"pre-wrap"}),
            ]),
        ])

    if tab == "tab-somnotate":
        models = available_models()
        return html.Div(className="card", children=[
            html.H3("Somnotate"),
            html.Div("These buttons call the external Somnotate pipeline. A spinner and command log will appear while each workflow runs.", className="app-subtitle"),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"10px"}, children=[
                html.Div([html.Label("Recording IDs, comma-separated"), PInput(id="som-recording-ids", type="text", value=",".join([o["value"] for o in rec_options[:1]]), style={"width":"100%"})]),
                html.Div([html.Label("Target fs"), PInput(id="som-target-fs", type="number", value=512.0, style={"width":"100%"})]),
                html.Div([html.Label("Somnotate repository path"), PInput(id="som-root", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Somnotate conda env"), PInput(id="som-conda-env", type="text", value="somnotate_env", style={"width":"100%"})]),
                html.Div([html.Label("Optional Somnotate Python executable"), PInput(id="som-python", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Existing model"), PDropdown(id="som-model-file", options=models, value=models[0]["value"] if models else None)]),
            ]),
            html.H4("Use existing model"),
            dcc.Checklist(id="som-existing-steps", options=[{"label":x,"value":x} for x in ["prepare","preprocess","score","probabilities","import-results"]], value=["prepare","preprocess","score","probabilities","import-results"], inline=True),
            html.Button("Run existing-model workflow", id="btn-som-existing", n_clicks=0),
            html.H4("Train new model"),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"10px"}, children=[
                html.Div([html.Label("Train recording IDs"), PInput(id="som-train-ids", type="text", style={"width":"100%"})]),
                html.Div([html.Label("Test recording IDs, optional"), PInput(id="som-test-ids", type="text", style={"width":"100%"})]),
                html.Div([html.Label("New model name"), PInput(id="som-model-name", type="text", value="my_somnotate_model", style={"width":"100%"})]),
            ]),
            dcc.Checklist(id="som-train-steps", options=[{"label":x,"value":x} for x in ["prepare","preprocess"]], value=["prepare","preprocess"], inline=True),
            html.Button("Train new model", id="btn-som-train", n_clicks=0),
            html.H4("Import already existing local results"),
            html.Button("Import Somnotate results", id="btn-som-import-results", n_clicks=0),
            html.Div(id="som-action-status", className="status-line"),
            dcc.Loading(type="circle", children=html.Pre(id="som-log", className="log-box")),
        ])

    if tab == "tab-stats":
        return html.Div(className="card", children=[
            html.H3("Dissociation review dashboard"),
            html.Div("Find where Layer 1, Somnotate, Manual and Final scoring disagree, then jump to those periods from the QC viewer.", className="app-subtitle"),
            html.Div(style={"display":"flex","gap":"8px","alignItems":"center", "flexWrap":"wrap"}, children=[
                html.Label("Recording:"), PDropdown(id="stats-recording", options=rec_options, value=rec_options[0]["value"] if rec_options else None, style={"width":"360px"}),
                html.Label("Threshold:"), PInput(id="diss-threshold", type="number", value=0.20, step=0.05, style={"width":"100px"}),
                html.Button("Run dissociation analysis", id="btn-run-diss", n_clicks=0),
            ]),
            html.Div(id="diss-action-status", className="status-line"),
            dcc.Loading(type="circle", children=html.Div(id="diss-log")),
            html.Div(id="diss-pairwise"),
            html.Div(id="diss-state", style={"display":"none"}),
            html.Div(id="diss-events", style={"display":"none"}),
        ])

    return html.Div(className="card", children=[
        html.H3("About"),
        html.P("This app supports semi-automated sleep scoring QC with manual review, Layer 1 Wake/Sleep, Somnotate comparison, dissociation event ranking, and export."),
        html.Ul([
            html.Li("Final scoring starts empty by default and is filled only when accepted/edited."),
            html.Li("Use selection mode to score a specific interval, or use window buttons to accept a source for the whole visible window."),
            html.Li("Run dissociation analysis, then use the QC review queue to jump through the most suspicious events."),
        ]),
    ])



# -----------------------------------------------------------------------------
# Immediate button feedback
# -----------------------------------------------------------------------------
@app.callback(
    Output("import-action-status", "children"),
    Input("detect-mat", "n_clicks"), Input("btn-import-mat", "n_clicks"),
    Input("btn-compute-features", "n_clicks"), Input("btn-run-layer1", "n_clicks"),
    prevent_initial_call=True,
)
def import_button_feedback(n_detect, n_import, n_features, n_layer1):
    trig = callback_context.triggered_id
    messages = {
        "detect-mat": "Reading variables / EDF channels...",
        "btn-import-mat": "Importing recording... this can take a moment.",
        "btn-compute-features": "Computing epoch features...",
        "btn-run-layer1": "Running Layer 1 Wake/Sleep...",
    }
    return messages.get(trig, "Working...")


@app.callback(
    Output("som-action-status", "children"),
    Input("btn-som-existing", "n_clicks"), Input("btn-som-train", "n_clicks"), Input("btn-som-import-results", "n_clicks"),
    prevent_initial_call=True,
)
def som_button_feedback(n_existing, n_train, n_import):
    trig = callback_context.triggered_id
    messages = {
        "btn-som-existing": "Running Somnotate existing-model workflow...",
        "btn-som-train": "Starting Somnotate model training workflow...",
        "btn-som-import-results": "Importing Somnotate results into the project...",
    }
    return messages.get(trig, "Working...")


@app.callback(
    Output("diss-action-status", "children"),
    Input("btn-run-diss", "n_clicks"),
    prevent_initial_call=True,
)
def diss_button_feedback(n):
    return "Running dissociation analysis..."


# -----------------------------------------------------------------------------
# Import callbacks
# -----------------------------------------------------------------------------
@app.callback(Output("mat-keys-status", "children"), Input("detect-mat", "n_clicks"), State("mat-file", "value"), prevent_initial_call=True)
def detect_mat_vars(n, mat_file):
    if mat_file is None or str(mat_file).strip() == "" or str(mat_file).strip().lower() == "none":
        return "Please paste the full path to a .mat, .edf, or .bdf file first."

    data_path = Path(str(mat_file)).expanduser()

    if not data_path.exists():
        return f"Recording file not found: {data_path}"

    suffix = data_path.suffix.lower()

    if suffix in {".edf", ".bdf"}:
        return safe_edf_info(str(data_path))

    if suffix != ".mat":
        return f"Unsupported file extension '{suffix}'. Use .mat, .edf, or .bdf."

    keys = safe_mat_keys(str(data_path))
    if not keys:
        return "Could not read variables. Check path/file."

    return "Detected MAT variables:\n" + ", ".join(keys)


@app.callback(Output("manifest-table-import", "children"), Input("manifest-refresh", "data"), State("project-root-store", "data"))
def show_manifest_table(refresh, project_root):
    manifest = load_manifest(project_root)
    if manifest is None or len(manifest)==0:
        return "No recordings found yet."
    return dcc.Graph(figure=go.Figure(data=[go.Table(header=dict(values=list(manifest.columns)), cells=dict(values=[manifest[c] for c in manifest.columns]))]).update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10)))


@app.callback(
    Output("import-log", "children"), Output("manifest-refresh", "data"),
    Input("btn-import-mat", "n_clicks"), Input("btn-compute-features", "n_clicks"), Input("btn-run-layer1", "n_clicks"),
    State("project-root-store", "data"), State("mat-file", "value"), State("import-recording-id", "value"), State("import-fs", "value"), State("eeg-key", "value"), State("emg-key", "value"), State("ach-key", "value"), State("eeg-fs-key", "value"), State("ach-fs-key", "value"), State("scoring-key", "value"), State("import-epoch-sec", "value"), State("import-mouse-id", "value"), State("import-group", "value"), State("import-condition", "value"), State("import-week", "value"), State("code-map", "value"), State("manifest-refresh", "data"),
    prevent_initial_call=True,
)
def run_import_pipeline(n1,n2,n3,project_root,mat_file,rec_id,fs,eeg_key,emg_key,ach_key,eeg_fs_key,ach_fs_key,scoring_key,epoch_sec,mouse_id,group,condition,week,code_map,refresh):
    if not project_root or not rec_id:
        return "Load project and enter recording ID first.", refresh
    trigger = callback_context.triggered_id
    if trigger == "btn-import-mat":
        if mat_file is None or str(mat_file).strip() == "" or str(mat_file).strip().lower() == "none":
            return (
                "Please paste the full path to a .mat, .edf, or .bdf file before pressing Import recording.\n\n"
                "Examples:\n"
                "/Users/margaridaseabra/Desktop/Margarida-batch2-june26/recordings/300526-m63-bas-1/my_recording.mat\n"
                "/Users/margaridaseabra/Desktop/Margarida-batch2-june26/recordings/300526-m63-bas-1/my_recording.edf",
                refresh,
            )

        data_path = Path(str(mat_file)).expanduser()

        if not data_path.exists():
            return f"Recording file not found:\n{data_path}", refresh

        suffix = data_path.suffix.lower()

        if suffix in {".edf", ".bdf"}:
            cmd = [
                sys.executable,
                str(PIPELINES_DIR/"01_import_edf_recording.py"),
                "--edf-file", str(data_path),
                "--project-root", str(project_root),
                "--recording-id", str(rec_id),
                "--eeg-channel", str(eeg_key or ""),
                "--emg-channel", str(emg_key or ""),
                "--epoch-sec", str(epoch_sec),
                "--annotation-map", str(code_map or "{}"),
                "--mouse-id", str(mouse_id or ""),
                "--group", str(group or ""),
                "--condition", str(condition or ""),
                "--week", str(week or ""),
            ]
            if ach_key:
                cmd += ["--ach-channel", str(ach_key)]
        elif suffix == ".mat":
            cmd = [sys.executable, str(PIPELINES_DIR/"01_import_mat_recording.py"), "--mat-file", str(data_path), "--project-root", str(project_root), "--recording-id", str(rec_id), "--eeg-key", str(eeg_key), "--emg-key", str(emg_key), "--fs", str(fs), "--epoch-sec", str(epoch_sec), "--code-map", str(code_map or "{}"), "--mouse-id", str(mouse_id or ""), "--group", str(group or ""), "--condition", str(condition or ""), "--week", str(week or "")]
            if ach_key:
                cmd += ["--ach-key", str(ach_key)]
            if eeg_fs_key:
                cmd += ["--eeg-fs-key", str(eeg_fs_key)]
            if ach_fs_key:
                cmd += ["--ach-fs-key", str(ach_fs_key)]
            if scoring_key: cmd += ["--scoring-key", str(scoring_key)]
        else:
            return f"Unsupported file extension '{suffix}'. Use .mat, .edf, or .bdf.", refresh
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
    Output("recording-id-store", "data"),
    Output("qc-graph", "figure"),
    Output("qc-graph", "style"),
    Output("empty-qc-message", "style"),
    Output("window-label", "children"),
    Output("load-status", "children"),
    Output("window-store", "data", allow_duplicate=True),
    Output("selected-interval-store", "data", allow_duplicate=True),
    Input("load-recording", "n_clicks"),
    State("project-root-store", "data"),
    State("recording-dropdown", "value"),
    prevent_initial_call=True,
)
def load_recording_cb(n, project_root, recording_id):
    """
    Load selected recording into QC/Review.

    This callback has exactly 8 outputs, so every return path returns exactly
    8 values in this order:
    recording_id, figure, graph_style, placeholder_style, window_label,
    load_status, window_store, selected_interval_store.
    """
    empty_fig = go.Figure()

    hidden_graph = {"display": "none"}
    visible_graph = {"display": "block"}

    visible_placeholder = {
        "border": "1px dashed #bbb",
        "borderRadius": "8px",
        "padding": "40px",
        "textAlign": "center",
        "color": "#555",
        "background": "#fafafa",
        "marginTop": "12px",
        "display": "block",
    }
    hidden_placeholder = {"display": "none"}

    default_window = {"start_min": 0.0, "window_min": 15.0}

    if not project_root:
        return (
            None,
            empty_fig,
            hidden_graph,
            visible_placeholder,
            "",
            "Load a project first.",
            default_window,
            None,
        )

    if not recording_id:
        return (
            None,
            empty_fig,
            hidden_graph,
            visible_placeholder,
            "",
            "Choose a recording first. If the list is empty, import a .mat recording first.",
            default_window,
            None,
        )

    try:
        rec = load_recording(project_root, recording_id)
        fig = make_review_figure(project_root, recording_id, 0.0, 15.0)

        duration_min = float(rec["duration_s"]) / 60.0
        end_min = min(duration_min, 15.0)

        return (
            recording_id,
            fig,
            visible_graph,
            hidden_placeholder,
            f"Window: 0.00–{end_min:.2f} min",
            f"Loaded {recording_id}",
            default_window,
            None,
        )

    except Exception as e:
        return (
            None,
            empty_fig,
            hidden_graph,
            visible_placeholder,
            "",
            f"Could not load recording: {type(e).__name__}: {e}",
            default_window,
            None,
        )



@app.callback(
    Output("qc-window-range-slider", "min"),
    Output("qc-window-range-slider", "max"),
    Output("qc-window-range-slider", "value"),
    Output("qc-window-range-slider", "marks"),
    Output("qc-window-range-slider", "disabled"),
    Output("qc-window-range-label", "children"),
    Input("window-store", "data"),
    State("project-root-store", "data"),
    State("recording-id-store", "data"),
)
def sync_qc_window_range_slider(window, project_root, recording_id):
    """
    Show where the current QC window sits inside the full recording.
    The highlighted slider interval is the visible window.
    """
    if not project_root or not recording_id:
        return 0, 1, [0, 1], {0: "0", 1: "1"}, True, "Load a recording to use the timeline."

    try:
        rec = load_recording(project_root, recording_id)
        duration_min = float(rec["duration_s"]) / 60.0

        start = float((window or {}).get("start_min", 0.0))
        window_min = float((window or {}).get("window_min", 15.0))
        end = min(duration_min, start + window_min)

        start = max(0.0, min(start, duration_min))
        end = max(start + 0.1, min(end, duration_min))

        # Keep marks sparse and readable.
        marks = {}
        for x in np.linspace(0, duration_min, 5):
            marks[round(float(x), 2)] = f"{x:.0f}"

        label = f"Visible window: {start:.2f}–{end:.2f} min / total {duration_min:.2f} min"

        return 0, max(duration_min, 1.0), [round(start, 3), round(end, 3)], marks, False, label

    except Exception as e:
        return 0, 1, [0, 1], {0: "0", 1: "1"}, True, f"Timeline unavailable: {type(e).__name__}: {e}"


@app.callback(
    Output("window-store", "data", allow_duplicate=True),
    Input("qc-window-range-slider", "value"),
    State("window-store", "data"),
    prevent_initial_call=True,
)
def move_window_from_qc_timeline(value, current_window):
    """
    Move the visible QC window by dragging the small recording timeline.
    """
    if not value or len(value) < 2:
        return no_update

    try:
        start = float(value[0])
        end = float(value[1])
    except Exception:
        return no_update

    if end <= start:
        return no_update

    window_min = end - start

    old_start = float((current_window or {}).get("start_min", -9999))
    old_window = float((current_window or {}).get("window_min", -9999))

    # Avoid feedback loops when the callback only reflects the current value.
    if abs(old_start - start) < 0.01 and abs(old_window - window_min) < 0.01:
        return no_update

    return {"start_min": start, "window_min": window_min}



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


@app.callback(
    Output("selected-interval-store", "data"),
    Output("selected-interval-label", "children"),
    Output("qc-graph", "figure", allow_duplicate=True),
    Input("qc-graph", "selectedData"),
    State("qc-graph", "figure"),
    prevent_initial_call=True,
)
def update_selection(selected, fig):
    if not selected:
        return no_update, no_update, no_update

    x0 = x1 = None

    if "range" in selected:
        r = selected["range"]

        if "x" in r:
            x0, x1 = r["x"]
        else:
            for k, v in r.items():
                if str(k).lower().startswith("x") and isinstance(v, list) and len(v) >= 2:
                    x0, x1 = v[0], v[1]
                    break

    if x0 is None and selected.get("points"):
        xs = [p.get("x") for p in selected["points"] if "x" in p]
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

    patch = Patch()

    # Preserve existing final-score background shapes.
    existing_shapes = []
    try:
        existing_shapes = [
            s for s in fig.get("layout", {}).get("shapes", [])
            if s.get("name") != "selected_interval"
        ]
    except Exception:
        existing_shapes = []

    selected_shape = {
        "type": "rect",
        "name": "selected_interval",
        "xref": "x",
        "yref": "paper",
        "x0": x0,
        "x1": x1,
        "y0": 0,
        "y1": 1,
        "fillcolor": "rgba(0,120,255,0.13)",
        "line": {"color": "rgba(0,90,220,0.95)", "width": 2},
        "layer": "above",
    }

    patch["layout"]["shapes"] = existing_shapes + [selected_shape]

    dur = (x1 - x0) * 60.0

    return (
        {"start_min": x0, "end_min": x1},
        f"Selected interval: {x0:.2f}–{x1:.2f} min ({dur:.1f} s)",
        patch,
    )


def parse_interval_from_selected_data(selected):
    """Read the current Plotly selectedData payload as a min-based interval.

    This is used by the scoring callback as a direct fallback/override for
    keyboard scoring. It prevents a fast keyboard press from using an older
    selected-interval store while Dash is still processing the newest lasso/box
    selection callback.
    """
    if not selected:
        return None

    x0 = x1 = None
    try:
        if "range" in selected:
            r = selected.get("range") or {}
            if "x" in r and isinstance(r["x"], (list, tuple)) and len(r["x"]) >= 2:
                x0, x1 = r["x"][0], r["x"][1]
            else:
                for k, v in r.items():
                    if str(k).lower().startswith("x") and isinstance(v, (list, tuple)) and len(v) >= 2:
                        x0, x1 = v[0], v[1]
                        break

        if x0 is None and selected.get("points"):
            xs = [pt.get("x") for pt in selected.get("points", []) if "x" in pt]
            if len(xs) >= 2:
                x0, x1 = min(xs), max(xs)

        if x0 is None or x1 is None:
            return None

        x0 = float(x0)
        x1 = float(x1)
        if x1 < x0:
            x0, x1 = x1, x0
        if x1 <= x0:
            return None
        return {"start_min": x0, "end_min": x1}
    except Exception:
        return None


@app.callback(Output("score-status", "children"), Output("qc-graph", "figure", allow_duplicate=True), Output("selected-interval-store", "data", allow_duplicate=True), Output("selected-interval-label", "children", allow_duplicate=True), Output("qc-graph", "selectedData", allow_duplicate=True), Input("score-wake", "n_clicks"), Input("score-nrem", "n_clicks"), Input("score-rem", "n_clicks"), Input("score-somnotate", "n_clicks"), Input("score-layer1", "n_clicks"), Input("score-manual", "n_clicks"), Input("score-window-somnotate", "n_clicks"), Input("score-window-layer1", "n_clicks"), Input("score-window-manual", "n_clicks"), Input("btn-reset-final-empty", "n_clicks"), Input("btn-undo", "n_clicks"), Input("btn-export", "n_clicks"), Input("btn-fill-empty-somnotate", "n_clicks"), Input("btn-fill-empty-somnotate-export", "n_clicks"), Input("btn-export-bottom", "n_clicks"), State("selected-interval-store", "data"), State("qc-graph", "selectedData"), State("project-root-store", "data"), State("recording-id-store", "data"), State("window-store", "data"), prevent_initial_call=True)
def score_or_export(*args):
    selected_store, graph_selected, project_root, recording_id, window = args[-5], args[-4], args[-3], args[-2], args[-1]
    if not project_root or not recording_id:
        return "No recording loaded.", no_update, no_update, no_update, no_update

    trig = callback_context.triggered_id

    if trig == "btn-undo":
        ok, msg = undo_last_action(project_root, recording_id)
        return msg, refresh_qc_figure_after_scoring(project_root, recording_id, window) if ok else no_update, no_update, no_update, no_update

    if trig in {"btn-export", "btn-export-bottom"}:
        ok, msg = export_final(project_root, recording_id)
        return msg, no_update, no_update, no_update, no_update

    if trig == "btn-reset-final-empty":
        ok, msg = reset_final_to_empty(project_root, recording_id)
        return msg, refresh_qc_figure_after_scoring(project_root, recording_id, window) if ok else no_update, None if ok else no_update, "Selection cleared." if ok else no_update, None if ok else no_update

    if trig in {"btn-fill-empty-somnotate", "btn-fill-empty-somnotate-export"}:
        ok, msg = fill_empty_final_with_somnotate(project_root, recording_id, export_after=(trig == "btn-fill-empty-somnotate-export"))
        return msg, refresh_qc_figure_after_scoring(project_root, recording_id, window) if ok else no_update, None if ok else no_update, "Selection cleared." if ok else no_update, None if ok else no_update

    # Determine whether to apply to selected interval or full visible window.
    window_buttons = {"score-window-somnotate", "score-window-layer1", "score-window-manual"}
    if trig in window_buttons:
        start = float((window or {}).get("start_min", 0.0))
        wmin = float((window or {}).get("window_min", 15.0))
        end = start + wmin
        scope_text = "visible window"
    else:
        selected = parse_interval_from_selected_data(graph_selected) or selected_store
        if not selected:
            return "Select an interval first, wait for the selected interval text to update, then score. Fast keyboard scoring is blocked until a confirmed/current selection is available.", no_update, no_update, no_update, no_update
        start = float(selected["start_min"])
        end = float(selected["end_min"])
        scope_text = "selected interval"

    if trig == "score-wake":
        ok, msg = apply_manual_label(project_root, recording_id, start, end, "Wake")
    elif trig == "score-nrem":
        ok, msg = apply_manual_label(project_root, recording_id, start, end, "NREM")
    elif trig == "score-rem":
        ok, msg = apply_manual_label(project_root, recording_id, start, end, "REM")
    elif trig in {"score-somnotate", "score-window-somnotate"}:
        ok, msg = apply_source_label(project_root, recording_id, start, end, "Somnotate")
    elif trig in {"score-layer1", "score-window-layer1"}:
        ok, msg = apply_source_label(project_root, recording_id, start, end, "Layer 1")
    elif trig in {"score-manual", "score-window-manual"}:
        ok, msg = apply_source_label(project_root, recording_id, start, end, "Manual")
    else:
        return "Unknown action.", no_update, no_update, no_update, no_update

    if ok:
        msg = f"{msg} Applied to {scope_text}: {start:.2f}–{end:.2f} min."

    clear_selection = ok and scope_text == "selected interval"
    selection_message = "Selection cleared after scoring. Select a new interval before using keyboard scoring again." if clear_selection else no_update
    return (
        msg,
        refresh_qc_figure_after_scoring(project_root, recording_id, window) if ok else no_update,
        None if clear_selection else no_update,
        selection_message,
        None if clear_selection else no_update,
    )


# -----------------------------------------------------------------------------
# Optional video QC callbacks
# -----------------------------------------------------------------------------

@app.callback(
    Output("video-file-input", "value"),
    Output("video-offset-input", "value"),
    Output("video-player-container", "children"),
    Output("video-status", "children"),
    Input("recording-id-store", "data"),
    Input("save-video-settings", "n_clicks"),
    Input("convert-video-mp4", "n_clicks"),
    State("project-root-store", "data"),
    State("video-file-input", "value"),
    State("video-offset-input", "value"),
    prevent_initial_call=True,
)
def update_video_panel(recording_id, save_clicks, convert_clicks, project_root, video_file_value, video_offset_value):
    if not project_root or not recording_id:
        return "", 0.0, html.Div("Load a recording to enable video QC.", className="app-subtitle"), "Load a recording first."

    trig = callback_context.triggered_id

    if trig == "save-video-settings":
        video_file = str(video_file_value or "").strip()
        offset_s = safe_float(video_offset_value, 0.0)
        ok, msg = save_video_metadata(project_root, recording_id, video_file, offset_s)
        return video_file, offset_s, video_panel_children(video_file, offset_s), msg

    if trig == "convert-video-mp4":
        original_video_file = str(video_file_value or "").strip()
        offset_s = safe_float(video_offset_value, 0.0)
        ok, msg, converted_path = convert_avi_to_browser_mp4(original_video_file)
        if not ok or not converted_path:
            return original_video_file, offset_s, video_panel_children(original_video_file, offset_s), msg

        ok_save, save_msg = save_video_metadata(project_root, recording_id, converted_path, offset_s)
        combined_msg = msg if ok_save else f"{msg}\nCould not save converted path: {save_msg}"
        if ok_save:
            combined_msg = f"{msg}\n{save_msg}"
        return converted_path, offset_s, video_panel_children(converted_path, offset_s), combined_msg

    video_file, offset_s = load_video_metadata(project_root, recording_id)
    return video_file, offset_s, video_panel_children(video_file, offset_s), video_format_message(video_file)


@app.callback(
    Output("video-seek-store", "data"),
    Input("jump-video-window", "n_clicks"),
    Input("jump-video-selected", "n_clicks"),
    State("window-store", "data"),
    State("selected-interval-store", "data"),
    State("video-offset-input", "value"),
    prevent_initial_call=True,
)
def request_video_seek(n_window, n_selected, window_data, selected_data, offset_s):
    trig = callback_context.triggered_id
    offset_s = safe_float(offset_s, 0.0)

    if trig == "jump-video-selected":
        if not selected_data:
            return {"error": "Select an interval first."}
        recording_start_s = float(selected_data.get("start_min", 0.0)) * 60.0
        recording_end_s = float(selected_data.get("end_min", selected_data.get("start_min", 0.0))) * 60.0
        auto_play = True
    else:
        window_data = window_data or {}
        start_min = float(window_data.get("start_min", 0.0))
        window_min = float(window_data.get("window_min", 15.0))
        recording_start_s = start_min * 60.0
        recording_end_s = (start_min + window_min) * 60.0
        auto_play = False

    video_start_s = max(0.0, recording_start_s - offset_s)
    video_end_s = max(video_start_s, recording_end_s - offset_s)

    return {
        "time_s": video_start_s,
        "end_time_s": video_end_s,
        "duration_s": max(0.0, video_end_s - video_start_s),
        "recording_time_s": recording_start_s,
        "recording_end_s": recording_end_s,
        "offset_s": offset_s,
        "source": trig,
        "auto_play": auto_play,
    }


app.clientside_callback(
    """
    function(data) {
        if (!data) {
            return "";
        }
        if (data.error) {
            return data.error;
        }
        const video = document.getElementById("qc-video-player");
        if (!video) {
            return "No video player loaded. Save a valid video path first.";
        }

        const start = Math.max(0, Number(data.time_s || 0));
        const end = Math.max(start, Number(data.end_time_s || start));
        const duration = Math.max(0, end - start);
        const autoPlay = Boolean(data.auto_play);

        try {
            if (video._qcStopHandler) {
                video.removeEventListener("timeupdate", video._qcStopHandler);
                video._qcStopHandler = null;
            }

            video.currentTime = start;

            if (duration > 0) {
                const stopHandler = function() {
                    if (video.currentTime >= end - 0.03) {
                        video.pause();
                        try { video.currentTime = end; } catch (e) {}
                        video.removeEventListener("timeupdate", stopHandler);
                        video._qcStopHandler = null;
                    }
                };
                video._qcStopHandler = stopHandler;
                video.addEventListener("timeupdate", stopHandler);
            }

            if (autoPlay && duration > 0) {
                const p = video.play();
                if (p && p.catch) {
                    p.catch(function() {});
                }
                return "Playing selected video interval: " + start.toFixed(2) + "–" + end.toFixed(2) + " s.";
            }

            video.pause();
            return "Video jumped to " + start.toFixed(2) + " s.";
        } catch (e) {
            return "Could not control video: " + e;
        }
    }
    """,
    Output("video-seek-feedback", "children"),
    Input("video-seek-store", "data"),
)


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
# QC mouse mode: pan vs select scoring window
# -----------------------------------------------------------------------------
@app.callback(
    Output("qc-graph", "figure", allow_duplicate=True),
    Output("qc-mode-status", "children"),
    Input("qc-mode-pan", "n_clicks"),
    Input("qc-mode-select-window", "n_clicks"),
    State("qc-graph", "figure"),
    prevent_initial_call=True,
)
def set_qc_mouse_mode_select_window(n_pan, n_select, fig):
    if not fig:
        return no_update, "Load a recording first."

    trig = callback_context.triggered_id

    patch = Patch()

    if trig == "qc-mode-select-window":
        patch["layout"]["dragmode"] = "select"
        patch["layout"]["selectdirection"] = "h"
        return patch, "Select mode active: drag horizontally over the signal to choose a scoring window."

    patch["layout"]["dragmode"] = "pan"
    patch["layout"]["selectdirection"] = "h"
    return patch, "Pan mode active: drag the plot to move through the recording."


# -----------------------------------------------------------------------------
# Dissociation review queue helpers
# -----------------------------------------------------------------------------
def load_dissociation_events_for_recording(project_root, recording_id):
    if not project_root or not recording_id:
        return None

    project_root = Path(project_root).expanduser().resolve()

    try:
        rec_dir = recording_dir_from_manifest(project_root, recording_id)
    except Exception:
        rec_dir = project_root / "recordings" / str(recording_id)

    events_file = rec_dir / "dissociation_analysis" / "dissociation_events.csv"

    if not events_file.exists():
        return None

    events = pd.read_csv(events_file)

    if len(events) == 0:
        return None

    if "rank" not in events.columns:
        events = events.copy()
        events["rank"] = np.arange(1, len(events) + 1)

    return events


def dissociation_event_options(events):
    if events is None or len(events) == 0:
        return []

    options = []

    for _, row in events.iterrows():
        rank = int(row.get("rank", len(options) + 1))
        start_min = float(row.get("start_min", row.get("start_s", 0) / 60.0))
        end_min = float(row.get("end_min", row.get("end_s", 0) / 60.0))
        score = float(row.get("max_dissociation_index", np.nan))
        reason = str(row.get("main_reason", ""))

        label = f"#{rank} | {start_min:.2f}–{end_min:.2f} min | score {score:.3f} | {reason}"

        event_id = str(row.get("event_id", f"event_{rank}"))

        options.append({"label": label, "value": event_id})

    return options


def get_event_row(events, event_id):
    if events is None or len(events) == 0 or not event_id:
        return None

    if "event_id" in events.columns:
        m = events["event_id"].astype(str) == str(event_id)
        if m.any():
            return events.loc[m].iloc[0]

    try:
        idx = int(event_id)
        if 0 <= idx < len(events):
            return events.iloc[idx]
    except Exception:
        pass

    return None


# -----------------------------------------------------------------------------
# Pan-to-move recording window
# -----------------------------------------------------------------------------
@app.callback(
    Output("window-store", "data", allow_duplicate=True),
    Input("qc-graph", "relayoutData"),
    State("window-store", "data"),
    State("recording-id-store", "data"),
    State("project-root-store", "data"),
    prevent_initial_call=True,
)
def pan_qc_graph_to_window(relayout, window_data, recording_id, project_root):
    """
    Allow the user to move through the recording by using Plotly pan.

    When the user pans any subplot horizontally, Dash receives the new x-axis
    range. We convert that range into the app's current window.
    """
    if not relayout or not recording_id or not project_root:
        return no_update

    # Ignore selection-only updates.
    if "selections" in relayout:
        return no_update

    x0 = x1 = None

    # Accept any xaxis range, since panels have independent x axes.
    for key in list(relayout.keys()):
        if key.endswith(".range[0]"):
            prefix = key.replace(".range[0]", "")
            k0 = f"{prefix}.range[0]"
            k1 = f"{prefix}.range[1]"
            if k0 in relayout and k1 in relayout:
                x0 = relayout[k0]
                x1 = relayout[k1]
                break

    if x0 is None or x1 is None:
        for key in list(relayout.keys()):
            if key.endswith(".range") and isinstance(relayout[key], (list, tuple)) and len(relayout[key]) >= 2:
                x0, x1 = relayout[key][0], relayout[key][1]
                break

    if x0 is None or x1 is None:
        return no_update

    try:
        x0 = float(x0)
        x1 = float(x1)
    except Exception:
        return no_update

    if x1 < x0:
        x0, x1 = x1, x0

    if x1 <= x0:
        return no_update

    try:
        rec = load_recording(project_root, recording_id)
        duration_min = float(rec["duration_s"]) / 60.0
    except Exception:
        return no_update

    window_min = max(0.5, x1 - x0)
    start_min = max(0.0, min(x0, max(0.0, duration_min - window_min)))

    old_start = float((window_data or {}).get("start_min", -9999))
    old_window = float((window_data or {}).get("window_min", -9999))

    # Avoid tiny feedback-loop updates.
    if abs(old_start - start_min) < 0.01 and abs(old_window - window_min) < 0.01:
        return no_update

    return {"start_min": start_min, "window_min": window_min}



# -----------------------------------------------------------------------------
# QC dissociation review queue callbacks
# -----------------------------------------------------------------------------
@app.callback(
    Output("qc-diss-event-dropdown", "options"),
    Output("qc-diss-event-dropdown", "value"),
    Output("qc-diss-event-status", "children"),
    Input("qc-refresh-diss-events", "n_clicks"),
    Input("recording-id-store", "data"),
    State("project-root-store", "data"),
    prevent_initial_call=True,
)
def refresh_qc_dissociation_events(n, recording_id, project_root):
    events = load_dissociation_events_for_recording(project_root, recording_id)

    if events is None or len(events) == 0:
        return [], None, "No dissociation events found yet. Run dissociation analysis in the Dissociation tab first."

    options = dissociation_event_options(events)
    value = options[0]["value"] if options else None

    return options, value, f"Loaded {len(options)} dissociation events."


@app.callback(
    Output("qc-diss-event-dropdown", "value", allow_duplicate=True),
    Input("qc-prev-diss-event", "n_clicks"),
    Input("qc-next-diss-event", "n_clicks"),
    State("qc-diss-event-dropdown", "options"),
    State("qc-diss-event-dropdown", "value"),
    prevent_initial_call=True,
)
def step_qc_dissociation_event(n_prev, n_next, options, value):
    if not options:
        return no_update

    values = [o["value"] for o in options]
    if value not in values:
        return values[0]

    idx = values.index(value)
    trig = callback_context.triggered_id

    if trig == "qc-prev-diss-event":
        idx = max(0, idx - 1)
    elif trig == "qc-next-diss-event":
        idx = min(len(values) - 1, idx + 1)

    return values[idx]


@app.callback(
    Output("window-store", "data", allow_duplicate=True),
    Output("selected-interval-store", "data", allow_duplicate=True),
    Output("qc-diss-event-status", "children", allow_duplicate=True),
    Input("qc-diss-event-dropdown", "value"),
    State("project-root-store", "data"),
    State("recording-id-store", "data"),
    State("window-store", "data"),
    prevent_initial_call=True,
)
def jump_to_qc_dissociation_event(event_id, project_root, recording_id, window_data):
    events = load_dissociation_events_for_recording(project_root, recording_id)

    row = get_event_row(events, event_id)

    if row is None:
        return no_update, no_update, "Could not find selected dissociation event."

    start_min = float(row.get("start_min", row.get("start_s", 0) / 60.0))
    end_min = float(row.get("end_min", row.get("end_s", 0) / 60.0))

    if end_min <= start_min:
        end_min = start_min + 0.5

    current_window = float((window_data or {}).get("window_min", 15.0))
    window_min = max(5.0, current_window)

    midpoint = (start_min + end_min) / 2.0
    new_start = max(0.0, midpoint - window_min / 2.0)

    selected = {
        "start_min": start_min,
        "end_min": end_min,
    }

    rank = row.get("rank", "")
    reason = row.get("main_reason", "")
    score = row.get("max_dissociation_index", np.nan)

    return (
        {"start_min": new_start, "window_min": window_min},
        selected,
        f"Jumped to dissociation event #{rank}: {start_min:.2f}–{end_min:.2f} min | score {score:.3f} | {reason}",
    )


@app.callback(
    Output("qc-graph", "figure", allow_duplicate=True),
    Input("selected-interval-store", "data"),
    State("qc-graph", "figure"),
    prevent_initial_call=True,
)
def shade_selected_interval_from_store(selected, fig):
    if not selected:
        return no_update

    try:
        x0 = float(selected["start_min"])
        x1 = float(selected["end_min"])
    except Exception:
        return no_update

    if x1 <= x0:
        return no_update

    patch = Patch()

    # Preserve existing shapes except the selected-event marker.
    existing = []
    try:
        existing = [
            s for s in fig.get("layout", {}).get("shapes", [])
            if s.get("name") != "selected_interval"
        ]
    except Exception:
        existing = []

    selected_shape = {
        "type": "rect",
        "name": "selected_interval",
        "xref": "x",
        "yref": "paper",
        "x0": x0,
        "x1": x1,
        "y0": 0,
        "y1": 1,
        "fillcolor": "rgba(0,120,255,0.12)",
        "line": {"color": "rgba(0,90,220,0.95)", "width": 2},
        "layer": "above",
    }

    patch["layout"]["shapes"] = existing + [selected_shape]

    return patch


# -----------------------------------------------------------------------------
# Dissociation callbacks
# -----------------------------------------------------------------------------


def rgba_from_hex(hex_color, alpha=0.07):
    """Convert #RRGGBB to rgba string."""
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return f"rgba(150,150,150,{alpha})"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def state_to_soft_fill(state, alpha=0.07):
    state = str(state)
    if state == "Sleep":
        state = "Layer 1 Sleep"
    color = STATE_COLORS.get(state, STATE_COLORS.get("Uncertain", "#9e9e9e"))
    return rgba_from_hex(color, alpha=alpha)


def state_bouts_from_epoch_table(df, state_col, start_min, end_min):
    """
    Convert epoch-level scoring into merged bouts inside the visible window.
    Requires t0_s, t1_s and a state column.
    """
    if df is None or len(df) == 0:
        return []

    if state_col not in df.columns:
        return []

    tmp = df.copy()
    tmp = tmp.dropna(subset=["t0_s", "t1_s"])
    tmp["t0_min"] = tmp["t0_s"].astype(float) / 60.0
    tmp["t1_min"] = tmp["t1_s"].astype(float) / 60.0

    tmp = tmp[(tmp["t0_min"] < end_min) & (tmp["t1_min"] > start_min)].copy()
    if len(tmp) == 0:
        return []

    tmp = tmp.sort_values("t0_min")

    bouts = []
    cur_state = None
    cur_start = None
    cur_end = None

    for _, row in tmp.iterrows():
        state = str(row[state_col])
        x0 = max(float(start_min), float(row["t0_min"]))
        x1 = min(float(end_min), float(row["t1_min"]))

        if x1 <= x0:
            continue

        if cur_state is None:
            cur_state = state
            cur_start = x0
            cur_end = x1
            continue

        if state == cur_state and x0 <= cur_end + 1e-6:
            cur_end = max(cur_end, x1)
        else:
            bouts.append((cur_start, cur_end, cur_state))
            cur_state = state
            cur_start = x0
            cur_end = x1

    if cur_state is not None:
        bouts.append((cur_start, cur_end, cur_state))

    return bouts


def add_scoring_background_to_raw_panels(fig, rec, start_min, end_min, raw_rows, source="Final"):
    source = "Final"  # force final-score shading
    """
    Add light scoring-colour shading over raw signal panels.

    This gives the raw signal context without hiding the black trace.
    """
    if source == "Final":
        df = rec.get("final")
        col = "final_state"
    elif source == "Manual":
        df = rec.get("manual")
        col = "manual_state"
    elif source == "Somnotate":
        df = rec.get("som")
        col = "somnotate_state"
    else:
        df = None
        col = ""

    if df is None or len(df) == 0:
        return fig

    bouts = state_bouts_from_epoch_table(df, col, start_min, end_min)

    for x0, x1, state in bouts:
        fill = state_to_soft_fill(state, alpha=0.07)

        for rr in raw_rows:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=fill,
                line_width=0,
                layer="above",
                row=rr,
                col=1,
            )

    return fig


def compact_metric_card(label, value):
    return html.Div(
        className="metric-card",
        children=[
            html.Div(str(value), className="metric-value"),
            html.Div(label, className="metric-label"),
        ],
    )


def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_first_numeric_col(df, candidates):
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def render_dissociation_dashboard(analysis_dir):
    """Clean, review-oriented dissociation dashboard."""
    analysis_dir = Path(analysis_dir)
    pairwise = safe_read_csv(analysis_dir / "dissociation_pairwise_summary.csv")
    state = safe_read_csv(analysis_dir / "dissociation_state_summary.csv")
    events = safe_read_csv(analysis_dir / "dissociation_events.csv")

    children = []
    n_events = 0 if events is None else len(events)
    max_score = "—"
    top_reason = "—"
    if events is not None and len(events):
        if "max_dissociation_index" in events.columns:
            max_score = f"{events['max_dissociation_index'].max():.3f}"
        if "main_reason" in events.columns:
            top_reason = str(events["main_reason"].astype(str).value_counts().index[0])

    key_pair_pct = "—"
    if pairwise is not None and len(pairwise) and "percent_disagree" in pairwise.columns:
        key_pair_pct = f"{pairwise['percent_disagree'].max():.1f}%"

    children.append(html.Div(className="dashboard-grid", children=[
        compact_metric_card("Dissociation events", n_events),
        compact_metric_card("Max score", max_score),
        compact_metric_card("Highest pairwise disagreement", key_pair_pct),
        compact_metric_card("Top reason", top_reason),
    ]))

    # Pairwise percent disagreement, horizontal and readable.
    if pairwise is not None and len(pairwise):
        label_col = "pair" if "pair" in pairwise.columns else pairwise.columns[0]
        value_col = "percent_disagree" if "percent_disagree" in pairwise.columns else find_first_numeric_col(pairwise, ["disagreement_fraction", "n_disagree_epochs"])
        if value_col:
            df = pairwise.copy().sort_values(value_col, ascending=True)
            fig = go.Figure(go.Bar(
                x=df[value_col], y=df[label_col].astype(str), orientation="h",
                marker_color="#6366F1", text=df[value_col].round(2), textposition="auto",
                hovertemplate="%{y}<br>%{x:.2f}<extra></extra>",
            ))
            fig.update_layout(
                title="Pairwise disagreement (% epochs)", template="plotly_white", height=300,
                margin=dict(l=180, r=25, t=55, b=45), xaxis_title="% disagreement", yaxis_title="",
            )
            children.append(html.Div(className="card", children=[dcc.Graph(figure=fig)]))

    # Event reasons.
    if events is not None and len(events) and "main_reason" in events.columns:
        counts = events["main_reason"].astype(str).value_counts().reset_index()
        counts.columns = ["reason", "count"]
        counts = counts.sort_values("count", ascending=True)
        fig = go.Figure(go.Bar(
            x=counts["count"], y=counts["reason"], orientation="h",
            marker_color="#14B8A6", text=counts["count"], textposition="auto",
        ))
        fig.update_layout(
            title="Why events were flagged", template="plotly_white", height=max(300, 50 * len(counts) + 110),
            margin=dict(l=220, r=25, t=55, b=45), xaxis_title="Number of events", yaxis_title="",
        )
        children.append(html.Div(className="card", children=[dcc.Graph(figure=fig)]))

    # Event timeline with reason split.
    if events is not None and len(events):
        df = events.copy()
        if "start_min" not in df.columns and "start_s" in df.columns:
            df["start_min"] = df["start_s"].astype(float) / 60.0
        score_col = "max_dissociation_index" if "max_dissociation_index" in df.columns else find_first_numeric_col(df, ["mean_dissociation_index"])
        if "start_min" in df.columns and score_col:
            fig = go.Figure()
            if "main_reason" in df.columns:
                reasons = list(df["main_reason"].astype(str).fillna("Unknown").unique())
            else:
                reasons = ["Dissociation"]
                df["main_reason"] = "Dissociation"
            palette = ["#F97316", "#6366F1", "#14B8A6", "#EF4444", "#A855F7", "#64748B"]
            for i, reason in enumerate(reasons):
                sub = df[df["main_reason"].astype(str) == reason]
                fig.add_trace(go.Scatter(
                    x=sub["start_min"], y=sub[score_col], mode="markers", name=reason,
                    marker=dict(size=9, color=palette[i % len(palette)], opacity=0.8),
                    customdata=np.stack([
                        sub.get("event_id", pd.Series([""] * len(sub))).astype(str),
                        sub.get("end_min", sub.get("end_s", pd.Series([np.nan]*len(sub)))).astype(str),
                    ], axis=-1),
                    hovertemplate="%{customdata[0]}<br>Start=%{x:.2f} min<br>Score=%{y:.3f}<extra></extra>",
                ))
            fig.update_layout(
                title="Dissociation event timeline", template="plotly_white", height=360,
                margin=dict(l=55, r=25, t=55, b=45), xaxis_title="Time (min)", yaxis_title="Dissociation score",
                legend=dict(orientation="h", y=1.08),
            )
            children.append(html.Div(className="card", children=[dcc.Graph(figure=fig)]))

    # Top ranked events table.
    if events is not None and len(events):
        cols = [c for c in ["rank", "event_id", "start_min", "end_min", "duration_s", "max_dissociation_index", "main_reason", "states_at_peak"] if c in events.columns]
        df = events[cols].head(25).copy()
        for c in ["start_min", "end_min", "max_dissociation_index"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(3)
        children.append(html.Div(className="card", children=[
            html.H4("Top events to review"),
            dash_table.DataTable(
                data=df.to_dict("records"), columns=[{"name": c.replace("_", " "), "id": c} for c in df.columns],
                page_size=10, sort_action="native", filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 12, "padding": "8px", "textAlign": "left", "whiteSpace": "normal", "height": "auto", "maxWidth": "320px"},
                style_header={"fontWeight": "bold", "background": "#F3F4F6", "border": "1px solid #E5E7EB"},
                style_data={"border": "1px solid #E5E7EB"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#FAFAFA"}],
            ),
        ]))

    if not children:
        children.append(html.Div("No dissociation outputs found yet.", className="card"))
    return children

def table_from_csv(path: Path, max_rows=50):
    if not path.exists(): return "Not found."
    df = pd.read_csv(path)
    if len(df) > max_rows: df = df.head(max_rows)
    return dcc.Graph(figure=go.Figure(data=[go.Table(header=dict(values=list(df.columns)), cells=dict(values=[df[c] for c in df.columns]))]).update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10)))



@app.callback(
    Output("diss-log", "children"),
    Output("diss-pairwise", "children"),
    Output("diss-state", "children"),
    Output("diss-events", "children"),
    Input("btn-run-diss", "n_clicks"),
    State("project-root-store", "data"),
    State("stats-recording", "value"),
    State("diss-threshold", "value"),
    prevent_initial_call=True,
)
def run_diss(n, project_root, recording_id, threshold):
    if not project_root or not recording_id:
        return "Load project and choose recording.", no_update, no_update, no_update

    try:
        project_root = Path(project_root).expanduser().resolve()
        rec_dir = recording_dir_from_manifest(project_root, recording_id)

        layer1_file = rec_dir / "layer1_wake_sleep.csv"

        if not layer1_file.exists():
            return (
                f"Layer 1 file not found:\n{layer1_file}",
                no_update,
                no_update,
                no_update,
            )

        # The existing pipeline expects:
        #   project_root / recordings / recording_id
        # If the dropdown value is "recordings/name", passing that directly causes:
        #   recordings/recordings/name
        # So use the actual folder basename.
        pipeline_recording_id = rec_dir.name

        threshold_text = str(threshold or "0.2").replace(",", ".")

        cmd = [
            sys.executable,
            str(PIPELINES_DIR / "30_dissociation_analysis.py"),
            "--project-root",
            str(project_root),
            "--recording-id",
            str(pipeline_recording_id),
            "--threshold",
            threshold_text,
        ]

        code, out = run_command(cmd)

        analysis = rec_dir / "dissociation_analysis"

        dashboard = render_dissociation_dashboard(analysis)
        short_status = html.Div(className="status-line", children=[
            html.B("Dissociation analysis complete. "),
            html.Span(f"Outputs saved in: {analysis}"),
            html.Details([
                html.Summary("Show command log"),
                html.Pre(out, className="log-box"),
            ], style={"marginTop": "8px"}),
        ])

        return (
            short_status,
            dashboard,
            "",
            "",
        )

    except Exception as e:
        return (
            f"Dissociation analysis failed: {type(e).__name__}: {e}",
            no_update,
            no_update,
            no_update,
        )



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




# -----------------------------------------------------------------------------
# Global QC mouse mode controls
# -----------------------------------------------------------------------------
@app.callback(
    Output("qc-graph", "figure", allow_duplicate=True),
    Output("global-qc-mode-status", "children"),
    Input("global-qc-mode-pan", "n_clicks"),
    Input("global-qc-mode-select-window", "n_clicks"),
    State("qc-graph", "figure"),
    prevent_initial_call=True,
)
def set_global_qc_mouse_mode_select_window(n_pan, n_select, fig):
    if not fig:
        return no_update, "Load a recording in QC / Review first."

    trig = callback_context.triggered_id
    patch = Patch()

    if trig == "global-qc-mode-select-window":
        patch["layout"]["dragmode"] = "select"
        patch["layout"]["selectdirection"] = "h"
        return patch, "Select mode active: drag horizontally on the QC plot to choose a scoring window."

    patch["layout"]["dragmode"] = "pan"
    patch["layout"]["selectdirection"] = "h"
    return patch, "Pan mode active: drag the QC plot to move through the recording."


if __name__ == "__main__":
    app.run(debug=True, port=8050)
