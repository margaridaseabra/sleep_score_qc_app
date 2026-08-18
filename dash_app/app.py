
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import platform
from datetime import datetime
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
LOGS_DIR = APP_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
VERSION_FILE = APP_DIR / "VERSION"
APP_VERSION = VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else "dev"
TESTED_SOMNOTATE_VERSION = "0.5.0"
TESTED_SOMNOTATE_COMMIT = "a20f33de62511d8c172e333896608b7fc166d0f0"


def _default_somnotate_root() -> str:
    configured = str(os.environ.get("SOMNOTATE_ROOT", "")).strip()
    if configured:
        return configured
    candidate = Path.home() / "somnotate"
    return str(candidate) if candidate.exists() else ""


DEFAULT_SOMNOTATE_ROOT = _default_somnotate_root()


def _safe_log_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "command"))
    return value.strip("._") or "command"


def _write_command_log(cmd: list[str], cwd: Path, returncode: int, stdout: str, stderr: str, started: datetime, finished: datetime) -> Path:
    command_name = _safe_log_name(Path(cmd[1]).stem if len(cmd) > 1 else Path(cmd[0]).stem)
    stamp = started.strftime("%Y%m%d_%H%M%S_%f")
    path = LOGS_DIR / f"{stamp}_{command_name}.log"
    content = [
        f"Started: {started.isoformat()}",
        f"Finished: {finished.isoformat()}",
        f"Elapsed seconds: {(finished-started).total_seconds():.3f}",
        f"Return code: {returncode}",
        f"Platform: {platform.platform()}",
        f"Python: {sys.executable}",
        f"Python version: {sys.version}",
        f"Working directory: {cwd}",
        "Command:",
        subprocess.list2cmdline([str(x) for x in cmd]),
        "",
        "--- STDOUT ---",
        stdout or "<empty>",
        "",
        "--- STDERR ---",
        stderr or "<empty>",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8", errors="replace")
    return path


DEFAULT_PROJECT_ROOT = os.environ.get("SLEEP_QC_PROJECT_ROOT", str(APP_DIR / "project_data"))


def PInput(*args, **kwargs):
    """Dash input with persistence for settings that are global to the browser.

    Video path/offset are intentionally *not* browser-persistent because they are
    recording-specific and are stored in that recording's ``metadata.json``.
    Persisting them in localStorage can show a stale path from another recording
    and can race with the callback that loads the real per-recording metadata.
    """
    component_id = kwargs.get("id")
    if component_id in {"video-file-input", "video-offset-input"}:
        kwargs.setdefault("persistence", False)
    else:
        kwargs.setdefault("persistence", True)
        kwargs.setdefault("persistence_type", "local")
    if component_id == "project-root-input":
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
# Opacity for scoring-colour backgrounds over EEG/EMG/ACh panels.
# Increase for stronger colours; decrease if the black trace becomes too obscured.
SCORING_BACKGROUND_ALPHA = 0.16
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
    run_cwd = Path(cwd or APP_DIR).resolve()
    started = datetime.now()
    stdout = ""
    stderr = ""
    returncode = 999
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(run_cwd),
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            shell=False,
        )
        returncode = int(p.returncode)
        stdout = p.stdout or ""
        stderr = p.stderr or ""
    except Exception as exc:
        stderr = repr(exc)
    finished = datetime.now()
    log_path = _write_command_log(cmd, run_cwd, returncode, stdout, stderr, started, finished)
    parts = [
        f"Return code: {returncode}",
        f"Diagnostic log: {log_path}",
    ]
    if stdout.strip():
        parts += ["", "[stdout]", stdout.strip()]
    if stderr.strip():
        parts += ["", "[stderr]", stderr.strip()]
    return returncode, "\n".join(parts)


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

    The path is fully URL-encoded so Windows drive letters, UNC paths, spaces and
    backslashes survive the browser round-trip.  A file modification token is
    included to prevent the browser from reusing a stale/black cached video after
    the user changes or converts the source.
    """
    raw = str(video_file or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        token = str(path.stat().st_mtime_ns) if path.exists() else "missing"
    except OSError:
        token = "unknown"
    return "/_local_video?path=" + quote(str(path), safe="") + "&v=" + quote(token, safe="")


def _ffprobe_video_info(video_file: str | Path | None) -> dict[str, Any]:
    """Return basic video stream metadata using ffprobe when available."""
    raw = str(video_file or "").strip()
    if not raw:
        return {}
    path = Path(raw).expanduser()
    if not path.exists() or not path.is_file():
        return {}
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,duration,bit_rate",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        if result.returncode != 0:
            return {}
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        return dict(streams[0]) if streams else {}
    except Exception:
        return {}


def browser_mp4_path_for_video(video_path: Path) -> Path:
    """Deterministic H.264/yuv420p browser copy stored beside the source video."""
    return video_path.with_name(f"{video_path.stem}_browser.mp4")


def _video_cache_root() -> Path:
    """Return a local per-user cache directory for short QC video clips."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        root = base / "SleepStageQC" / "video_cache"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "SleepStageQC" / "video_cache"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "SleepStageQC" / "video_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _source_video_signature(path: Path) -> str:
    """Stable signature that invalidates cached clips when the source changes."""
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:16]


def _safe_recording_cache_name(recording_id: str | None) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(recording_id or "recording")).strip("._")
    return value or "recording"


def _cached_clip_path(source: Path, recording_id: str | None, start_s: float, end_s: float) -> Path:
    cache_dir = _video_cache_root() / _safe_recording_cache_name(recording_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = _source_video_signature(source)
    start_ms = int(round(float(start_s) * 1000.0))
    end_ms = int(round(float(end_s) * 1000.0))
    return cache_dir / f"{sig}_{start_ms}_{end_ms}_qc.mp4"


def _prune_video_cache(max_bytes: int = 10 * 1024**3) -> None:
    """Keep the local QC cache bounded without touching original videos."""
    root = _video_cache_root()
    files = []
    total = 0
    for path in root.rglob("*.mp4"):
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append((stat.st_mtime, stat.st_size, path))
        total += stat.st_size
    if total <= max_bytes:
        return
    for _, size, path in sorted(files):
        try:
            path.unlink()
            total -= size
        except OSError:
            pass
        if total <= max_bytes:
            break


def clear_video_review_cache(recording_id: str | None = None) -> tuple[bool, str]:
    """Delete cached short review clips, optionally only for one recording."""
    root = _video_cache_root()
    target = root / _safe_recording_cache_name(recording_id) if recording_id else root
    if not target.exists():
        return True, "Video review cache is already empty."
    try:
        if target == root:
            for child in list(root.iterdir()):
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        else:
            shutil.rmtree(target, ignore_errors=True)
        return True, f"Cleared local video review cache: {target}"
    except Exception as e:
        return False, f"Could not clear video review cache: {type(e).__name__}: {e}"


def _clip_info_covers_selection(
    clip_info: dict[str, Any] | None,
    source: Path,
    selected_video_start_s: float,
    selected_video_end_s: float,
) -> bool:
    if not clip_info:
        return False
    try:
        clip_path = Path(str(clip_info.get("clip_path", ""))).expanduser()
        return (
            clip_path.exists()
            and str(clip_info.get("source_signature", "")) == _source_video_signature(source)
            and float(clip_info.get("source_start_s", 0.0)) <= float(selected_video_start_s) + 1e-6
            and float(clip_info.get("source_end_s", 0.0)) >= float(selected_video_end_s) - 1e-6
        )
    except Exception:
        return False


def prepare_local_qc_clip(
    video_file: str | Path,
    recording_id: str | None,
    selected_recording_start_s: float,
    selected_recording_end_s: float,
    video_offset_s: float = 0.0,
    context_s: float = 60.0,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Create/reuse a short, seek-friendly local H.264 clip for synchronized QC.

    The original video can live on a network drive. Only the selected interval
    plus a small amount of context is decoded and written to the local user cache.
    Clip timestamps start at zero, while ``source_start_s`` keeps the exact mapping
    back to the original video and therefore to recording time.
    """
    raw = str(video_file or "").strip()
    if not raw:
        return False, "No video linked to this recording.", None
    source = Path(raw).expanduser()
    if not source.exists() or not source.is_file():
        return False, f"Video file not found: {source}", None

    offset = float(video_offset_s or 0.0)
    selected_video_start = max(0.0, float(selected_recording_start_s) - offset)
    selected_video_end = max(selected_video_start, float(selected_recording_end_s) - offset)
    if selected_video_end <= selected_video_start:
        selected_video_end = selected_video_start + 1.0

    info = _ffprobe_video_info(source)
    duration = safe_float(info.get("duration"), 0.0)
    clip_start = max(0.0, selected_video_start - float(context_s))
    clip_end = selected_video_end + float(context_s)
    if duration > 0:
        clip_end = min(duration, clip_end)
    if clip_end <= clip_start:
        clip_end = clip_start + max(1.0, selected_video_end - selected_video_start)

    out_path = _cached_clip_path(source, recording_id, clip_start, clip_end)
    signature = _source_video_signature(source)
    result_info = {
        "clip_path": str(out_path),
        "source_path": str(source),
        "source_signature": signature,
        "source_start_s": float(clip_start),
        "source_end_s": float(clip_end),
        "selected_video_start_s": float(selected_video_start),
        "selected_video_end_s": float(selected_video_end),
        "video_offset_s": offset,
    }

    if out_path.exists() and out_path.is_file() and out_path.stat().st_size > 0:
        result_info["reused"] = True
        return True, (
            f"Local QC clip ready (reused): {clip_start/60:.2f}–{clip_end/60:.2f} video min · "
            f"{out_path.stat().st_size / (1024**2):.1f} MB"
        ), result_info

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "FFmpeg was not found in the active app environment.", None

    duration_to_encode = max(0.25, clip_end - clip_start)
    cmd = [
        ffmpeg,
        "-y",
        "-ss", f"{clip_start:.3f}",
        "-i", str(source),
        "-t", f"{duration_to_encode:.3f}",
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-force_key_frames", "expr:gte(t,n_forced*1)",
        "-sc_threshold", "0",
        "-movflags", "+faststart",
        str(out_path),
    ]
    started = datetime.now()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except Exception as e:
        out_path.unlink(missing_ok=True)
        return False, f"Could not create local QC clip: {type(e).__name__}: {e}", None
    elapsed = (datetime.now() - started).total_seconds()
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size <= 0:
        out_path.unlink(missing_ok=True)
        err = (proc.stderr or proc.stdout or "FFmpeg failed").strip()
        if len(err) > 2500:
            err = err[-2500:]
        return False, f"Could not create local QC clip.\n{err}", None

    _prune_video_cache()
    result_info["reused"] = False
    result_info["elapsed_s"] = float(elapsed)
    result_info["size_mb"] = float(out_path.stat().st_size / (1024**2))
    return True, (
        f"Local QC clip ready in {elapsed:.1f} s: {clip_start/60:.2f}–{clip_end/60:.2f} video min · "
        f"{result_info['size_mb']:.1f} MB"
    ), result_info


def epoch_review_cached_video_children(clip_info: dict[str, Any] | None):
    """Render the synchronized player for a short local QC clip."""
    if not clip_info:
        return html.Div(
            "Select an interval to prepare a short local QC clip.",
            className="app-subtitle",
            style={"padding": "14px"},
        )
    clip_path = Path(str(clip_info.get("clip_path", ""))).expanduser()
    if not clip_path.exists():
        return html.Div(f"Cached QC clip not found: {clip_path}", className="status-line")
    source_start = float(clip_info.get("source_start_s", 0.0) or 0.0)
    source_end = float(clip_info.get("source_end_s", source_start) or source_start)
    return html.Div([
        html.Video(
            id="epoch-review-video-player",
            src=video_url_for_path(clip_path),
            controls=True,
            preload="auto",
            muted=True,
            playsInline=True,
            **{
                "data-qc-local-clip": "1",
                "data-source-start-s": f"{source_start:.6f}",
                "data-source-end-s": f"{source_end:.6f}",
            },
            style={
                "width": "100%",
                "height": "360px",
                "objectFit": "contain",
                "background": "#000",
                "borderRadius": "10px",
            },
        ),
        html.Div(
            f"Local QC clip · source video {source_start/60:.2f}–{source_end/60:.2f} min · "
            f"cache: {clip_path.parent}",
            className="app-subtitle",
            style={"marginTop": "5px", "wordBreak": "break-all"},
        ),
    ])


def _browser_video_choice(video_file: str | Path | None) -> tuple[Path | None, str, bool]:
    """Choose the best local file for browser playback.

    Returns ``(path_to_play, message, conversion_recommended)``.  If a current
    ``*_browser.mp4`` copy already exists, it is preferred automatically.
    """
    raw = str(video_file or "").strip()
    if not raw:
        return None, "No video selected yet.", False

    source = Path(raw).expanduser()
    if not source.exists() or not source.is_file():
        return source, f"Video file not found: {source}", False

    browser_copy = browser_mp4_path_for_video(source)
    if browser_copy.exists() and browser_copy.is_file() and browser_copy.stat().st_size > 0:
        try:
            if browser_copy.stat().st_mtime >= source.stat().st_mtime:
                return browser_copy, f"Using browser-compatible copy: {browser_copy.name}", False
        except OSError:
            return browser_copy, f"Using browser-compatible copy: {browser_copy.name}", False

    info = _ffprobe_video_info(source)
    codec = str(info.get("codec_name", "") or "").lower()
    pix_fmt = str(info.get("pix_fmt", "") or "").lower()
    suffix = source.suffix.lower()

    # Chrome/Edge/Safari are most consistently happy with H.264 in an MP4/M4V
    # container and 4:2:0 pixel formats.  Other combinations may show controls
    # but render a black frame, which is especially confusing during QC.
    safe_codec = codec in {"h264", "avc1"}
    safe_container = suffix in {".mp4", ".m4v"}
    safe_pix = (not pix_fmt) or pix_fmt in {"yuv420p", "yuvj420p"}
    if safe_codec and safe_container and safe_pix:
        detail = f"H.264/{pix_fmt or 'compatible pixel format'}" if codec else "browser-compatible MP4"
        return source, f"Video codec looks browser-compatible ({detail}).", False

    if not info:
        if suffix in {".mp4", ".m4v"}:
            return source, "MP4 selected. If the image is black, make a browser-compatible MP4 below.", True
        return source, "This video format may not play reliably in the browser. Make a browser-compatible MP4 below.", True

    detail = ", ".join(x for x in [codec or "unknown codec", pix_fmt or "unknown pixel format"] if x)
    return source, f"Video may render black in the browser ({detail}). Make a browser-compatible H.264 MP4 below.", True


def video_format_message(video_file: str | Path | None) -> str:
    _, message, _ = _browser_video_choice(video_file)
    return message


def video_panel_children(video_file: str | Path | None, offset_s: float | int | str | None = 0.0):
    playback_path, message, conversion_recommended = _browser_video_choice(video_file)
    if playback_path is None:
        return html.Div("No video file saved for this recording yet.", className="app-subtitle")

    if not playback_path.exists():
        return html.Div(message, className="status-line")

    messages = [html.Div(message, className="status-line")]
    if conversion_recommended:
        messages.append(html.Div(
            "Tip: if you see a black player or playback does not start, click “Make browser-compatible MP4”.",
            className="app-subtitle",
            style={"marginTop": "4px"},
        ))

    return html.Div(children=messages + [
        html.Video(
            id="qc-video-player",
            src=video_url_for_path(playback_path),
            controls=True,
            preload="auto",
            muted=True,
            playsInline=True,
            style={"width": "100%", "maxHeight": "420px", "background": "#000", "borderRadius": "10px"},
        ),
        html.Div(
            f"Video offset: {float(offset_s or 0):.3f} s. Video time = recording time - offset.",
            className="app-subtitle",
            style={"marginTop": "6px"},
        ),
    ])


def epoch_review_video_children(video_file: str | Path | None, offset_s: float | int | str | None = 0.0):
    """Video player used by the synchronized epoch-review panel."""
    if not video_file:
        return html.Div(
            [
                html.Div("No video linked to this recording.", style={"fontWeight": "700"}),
                html.Div(
                    "Use Video QC below to save a video path. The EEG/EMG epoch reviewer still works without video.",
                    className="app-subtitle",
                    style={"marginTop": "4px"},
                ),
            ],
            style={"padding": "14px"},
        )

    playback_path, message, conversion_recommended = _browser_video_choice(video_file)
    if playback_path is None or not playback_path.exists():
        return html.Div(
            [
                html.Div("Saved video could not be found.", style={"fontWeight": "700"}),
                html.Div(str(playback_path or video_file), className="app-subtitle", style={"marginTop": "4px", "wordBreak": "break-all"}),
            ],
            style={"padding": "14px"},
        )

    notice_style = {"marginTop": "5px"}
    if conversion_recommended:
        notice_style["fontWeight"] = "600"

    return html.Div([
        # Legacy full-source helper. Keep a distinct ID so synchronized-review
        # JavaScript can never accidentally attach to the 10-hour source video.
        html.Video(
            id="epoch-review-full-source-video-player",
            src=video_url_for_path(playback_path),
            controls=True,
            preload="metadata",
            muted=True,
            playsInline=True,
            style={
                "width": "100%",
                "height": "360px",
                "objectFit": "contain",
                "background": "#000",
                "borderRadius": "10px",
            },
        ),
        html.Div(message, className="app-subtitle", style=notice_style),
        html.Div(
            f"Offset {float(offset_s or 0):.3f} s · video time = recording time − offset",
            className="app-subtitle",
            style={"marginTop": "3px"},
        ),
    ])


def _state_at_time(source: pd.DataFrame | None, label_col: str, time_s: float, default: str = "Undefined") -> str:
    if source is None or label_col not in source.columns or not {"t0_s", "t1_s"}.issubset(source.columns):
        return default
    try:
        t0 = pd.to_numeric(source["t0_s"], errors="coerce")
        t1 = pd.to_numeric(source["t1_s"], errors="coerce")
        m = (t0 <= float(time_s)) & (t1 > float(time_s))
        if not m.any():
            return default
        return normalize_state_label(source.loc[m, label_col].iloc[0])
    except Exception:
        return default


def _review_epoch_table(rec: dict[str, Any]) -> pd.DataFrame:
    """Return the canonical scoring epochs used by the main QC plot.

    The main viewer is based on Layer 1 epoch boundaries.  The synchronized
    video reviewer must use those same boundaries instead of trusting an older
    ``final_scoring.csv`` file, which may have been created with a different
    epoch length or may contain stale/invalid time columns after a project is
    moved or reprocessed.
    """
    layer1 = rec.get("layer1")
    if layer1 is None or not {"t0_s", "t1_s"}.issubset(layer1.columns):
        return pd.DataFrame()

    cols = ["t0_s", "t1_s"]
    if "epoch_id" in layer1.columns:
        cols.insert(0, "epoch_id")
    out = layer1.loc[:, cols].copy()
    if "epoch_id" not in out.columns:
        out.insert(0, "epoch_id", np.arange(len(out), dtype=int))

    out["t0_s"] = pd.to_numeric(out["t0_s"], errors="coerce")
    out["t1_s"] = pd.to_numeric(out["t1_s"], errors="coerce")
    out = out.dropna(subset=["t0_s", "t1_s"])
    out = out[out["t1_s"] > out["t0_s"]].copy()
    if len(out) == 0:
        return out

    # Final is a label layer, not the source of epoch geometry.  Sample it at
    # the canonical epoch midpoint so the summary still shows the reviewed
    # Final state even when Final and Layer 1 were generated at different times.
    final = rec.get("final")
    if final is not None:
        out["final_state"] = labels_at_epoch_midpoints(out, final, "final_state")
    else:
        out["final_state"] = "Undefined"

    return out.sort_values(["t0_s", "t1_s"]).reset_index(drop=True)


def _epoch_geometry_from_table(source: pd.DataFrame | None) -> pd.DataFrame:
    """Extract a clean epoch geometry table from any scoring layer."""
    if source is None or not {"t0_s", "t1_s"}.issubset(source.columns):
        return pd.DataFrame()
    cols = ["t0_s", "t1_s"]
    if "epoch_id" in source.columns:
        cols.insert(0, "epoch_id")
    out = source.loc[:, cols].copy()
    if "epoch_id" not in out.columns:
        out.insert(0, "epoch_id", np.arange(len(out), dtype=int))
    out["t0_s"] = pd.to_numeric(out["t0_s"], errors="coerce")
    out["t1_s"] = pd.to_numeric(out["t1_s"], errors="coerce")
    out = out.dropna(subset=["t0_s", "t1_s"])
    out = out[out["t1_s"] > out["t0_s"]].copy()
    return out.sort_values(["t0_s", "t1_s"]).reset_index(drop=True)


def _epoch_rows_for_selection(rec: dict[str, Any], selected: dict[str, Any] | None) -> pd.DataFrame:
    """Return scoring epochs overlapping the selected QC interval.

    Layer 1 is normally the canonical epoch grid, but older/moved projects can
    contain a stale or partial Layer 1 table while Final/manual scoring still
    has the correct full-recording geometry.  The synchronized reviewer should
    therefore fall back to another valid scoring layer instead of showing an
    empty panel for a visibly valid selection.
    """
    if not selected:
        return pd.DataFrame()
    try:
        start_s = float(selected.get("start_min", 0.0)) * 60.0
        end_s = float(selected.get("end_min", selected.get("start_min", 0.0))) * 60.0
    except Exception:
        return pd.DataFrame()
    if end_s <= start_s:
        return pd.DataFrame()

    # Prefer Layer 1 so review follows the same epoch grid as the main QC plot.
    # If it does not cover this part of the recording, use the first other
    # prepared scoring layer that does.  Final is preferred because it normally
    # uses the app's base scoring epoch length; Somnotate may use a coarser grid.
    candidates = [
        _review_epoch_table(rec),
        _epoch_geometry_from_table(rec.get("final")),
        _epoch_geometry_from_table(rec.get("manual")),
        _epoch_geometry_from_table(rec.get("som")),
    ]
    for epochs in candidates:
        if len(epochs) == 0:
            continue
        t0 = epochs["t0_s"].to_numpy(float)
        t1 = epochs["t1_s"].to_numpy(float)
        mask = (t0 < end_s) & (t1 > start_s)
        if not mask.any():
            continue
        out = epochs.loc[mask].reset_index(drop=True)
        # Attach Final labels when the fallback geometry came from a layer that
        # does not already carry them.
        if "final_state" not in out.columns:
            final = rec.get("final")
            out["final_state"] = (
                labels_at_epoch_midpoints(out, final, "final_state")
                if final is not None else "Undefined"
            )
        return out

    # Prepared scoring tables can occasionally have a short gap or end early
    # even though the underlying EEG/EMG arrays continue.  For synchronized
    # review we still want to show the signal and let the user move through the
    # selected interval. Reconstruct only the geometry; labels remain sampled
    # from the existing scoring sources and therefore stay Undefined where no
    # label exists.
    return _signal_epoch_rows_for_selection(rec, start_s, end_s)


def _recording_signal_duration_s(rec: dict[str, Any]) -> float:
    """Best available EEG/EMG duration without loading full arrays into RAM."""
    durations: list[float] = []
    fs = safe_float(rec.get("fs"), 0.0)
    if fs > 0:
        recording_dir = Path(rec.get("recording_dir", "."))
        for name in ("eeg.npy", "emg.npy"):
            path = recording_dir / name
            if not path.exists():
                continue
            try:
                arr = np.load(path, mmap_mode="r")
                durations.append(float(len(arr)) / fs)
            except Exception:
                pass
    meta_duration = safe_float(rec.get("duration_s"), 0.0)
    if meta_duration > 0:
        durations.append(meta_duration)
    return min(durations) if durations else 0.0


def _infer_review_epoch_sec(rec: dict[str, Any]) -> float:
    """Infer the scoring epoch length from prepared layers, then metadata."""
    for key in ("layer1", "final", "manual", "som"):
        source = rec.get(key)
        if source is None or not {"t0_s", "t1_s"}.issubset(source.columns):
            continue
        try:
            dt = (
                pd.to_numeric(source["t1_s"], errors="coerce")
                - pd.to_numeric(source["t0_s"], errors="coerce")
            ).to_numpy(float)
            dt = dt[np.isfinite(dt) & (dt > 1e-6) & (dt <= 120.0)]
            if len(dt):
                return float(np.median(dt))
        except Exception:
            pass
    meta = rec.get("metadata") or {}
    for key in ("epoch_sec", "time_resolution", "somnotate_epoch_sec"):
        value = safe_float(meta.get(key), 0.0)
        if value > 0:
            return float(value)
    return 1.0


def _signal_epoch_rows_for_selection(rec: dict[str, Any], start_s: float, end_s: float) -> pd.DataFrame:
    """Reconstruct a visual epoch grid when scoring tables have a gap.

    This is only a review/navigation fallback. Labels are sampled from whichever
    scoring layers exist; it does not create or alter scoring files.
    """
    duration_s = _recording_signal_duration_s(rec)
    if duration_s <= 0 or start_s >= duration_s:
        return pd.DataFrame()
    end_s = min(float(end_s), duration_s)
    if end_s <= start_s:
        return pd.DataFrame()

    epoch_s = max(0.001, _infer_review_epoch_sec(rec))
    first = max(0.0, np.floor(float(start_s) / epoch_s) * epoch_s)
    # Include every reconstructed epoch that overlaps the selected interval.
    starts = np.arange(first, end_s + epoch_s, epoch_s, dtype=float)
    rows = []
    for t0 in starts:
        t1 = min(float(t0 + epoch_s), duration_s)
        if t0 < end_s and t1 > start_s and t1 > t0:
            rows.append((t0, t1))
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows, columns=["t0_s", "t1_s"])
    out.insert(0, "epoch_id", np.floor(out["t0_s"].to_numpy(float) / epoch_s + 1e-9).astype(int))
    final = rec.get("final")
    out["final_state"] = (
        labels_at_epoch_midpoints(out, final, "final_state")
        if final is not None else "Undefined"
    )
    out["review_geometry_source"] = "signal_fallback"
    return out.reset_index(drop=True)


def _epoch_review_empty_message(rec: dict[str, Any], selected: dict[str, Any] | None) -> str:
    if not selected:
        return "Select an interval in the main QC plot."
    try:
        start_s = float(selected.get("start_min", 0.0)) * 60.0
        end_s = float(selected.get("end_min", selected.get("start_min", 0.0))) * 60.0
    except Exception:
        return "Could not interpret the selected interval."
    duration_s = _recording_signal_duration_s(rec)
    if duration_s > 0 and start_s >= duration_s:
        return (
            f"This selection starts at {_format_recording_clock(start_s)}, but the EEG/EMG arrays end at "
            f"{_format_recording_clock(duration_s)}. No signal exists here to display."
        )
    if duration_s > 0 and end_s > duration_s:
        return (
            f"Only part of this selection has EEG/EMG data (signal ends at "
            f"{_format_recording_clock(duration_s)}). Select an earlier interval."
        )
    return "No scoring epochs overlap this selection, and a signal-based review grid could not be reconstructed."


def _format_recording_clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def make_epoch_review_figure(
    rec: dict[str, Any], epoch_row: pd.Series, selected: dict[str, Any] | None = None
) -> go.Figure:
    """Create the EEG/EMG review view for the full selected interval."""
    t0_s = float(epoch_row["t0_s"])
    t1_s = float(epoch_row["t1_s"])
    epoch_s = max(0.001, t1_s - t0_s)

    if selected:
        selection_start_s = float(selected.get("start_min", t0_s / 60.0)) * 60.0
        selection_end_s = float(selected.get("end_min", t1_s / 60.0)) * 60.0
        if selection_end_s < selection_start_s:
            selection_start_s, selection_end_s = selection_end_s, selection_start_s
    else:
        selection_start_s, selection_end_s = t0_s, t1_s

    # A little context keeps the boundaries readable, while the full selected
    # interval stays on screen for the entire video playback.
    context_s = min(1.0, epoch_s)
    view_start_s = max(0.0, selection_start_s - context_s)
    view_end_s = min(float(rec.get("duration_s", selection_end_s + context_s)), selection_end_s + context_s)
    if view_end_s <= view_start_s:
        view_start_s, view_end_s = selection_start_s, selection_end_s

    eeg_t_min, eeg = downsample_npy_window(
        rec["recording_dir"] / "eeg.npy", rec["fs"], view_start_s, view_end_s, max_points=18000
    )
    emg_t_min, emg = downsample_npy_window(
        rec["recording_dir"] / "emg.npy", rec["fs"], view_start_s, view_end_s, max_points=18000
    )
    eeg_t_s = eeg_t_min * 60.0
    emg_t_s = emg_t_min * 60.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.11,
        subplot_titles=["EEG", "EMG"],
        row_heights=[0.56, 0.44],
    )
    fig.add_trace(
        go.Scattergl(x=eeg_t_s, y=eeg, mode="lines", line={"color": RAW_TRACE_COLOR, "width": 1}, name="EEG"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(x=emg_t_s, y=emg, mode="lines", line={"color": RAW_TRACE_COLOR, "width": 1}, name="EMG"),
        row=2,
        col=1,
    )

    final_state = normalize_state_label(epoch_row.get("final_state", "Undefined"))
    midpoint = (t0_s + t1_s) / 2.0
    som_state = _state_at_time(rec.get("som"), "somnotate_state", midpoint)
    manual_state = _state_at_time(rec.get("manual"), "manual_state", midpoint)
    layer1_state = _state_at_time(rec.get("layer1"), "layer1_label", midpoint)
    if layer1_state == "Sleep":
        layer1_state = "Layer 1 Sleep"

    highlight_state = next(
        (x for x in [final_state, som_state, manual_state, layer1_state] if x not in {"Undefined", "Uncertain", ""}),
        "Undefined",
    )
    highlight_color = STATE_COLORS.get(highlight_state, STATE_COLORS["Undefined"])

    for row in (1, 2):
        fig.add_vrect(
            x0=t0_s,
            x1=t1_s,
            fillcolor=highlight_color,
            opacity=0.14,
            line={"color": highlight_color, "width": 2},
            row=row,
            col=1,
        )

    # One browser-only tracer across both panels.  Dash/Python never updates
    # this line while video is playing; JavaScript moves only x0/x1.
    fig.add_shape(
        type="line",
        x0=selection_start_s,
        x1=selection_start_s,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "#d62728", "width": 2.5},
        name="video_playhead",
        layer="above",
    )

    # Keep the signal panel deliberately simple: the current scoring epoch is
    # indicated by the shaded rectangle above.  The video can play continuously,
    # but the EEG/EMG panel changes only when the user presses Prev/Next/Replay.

    eeg_range = robust_range(eeg)
    emg_range = robust_range(emg)
    if eeg_range:
        fig.update_yaxes(range=eeg_range, row=1, col=1)
    if emg_range:
        fig.update_yaxes(range=emg_range, row=2, col=1)

    fig.update_xaxes(range=[view_start_s, view_end_s], row=1, col=1)
    fig.update_xaxes(range=[view_start_s, view_end_s], title_text="Recording time (s)", row=2, col=1)
    fig.update_layout(
        height=330,
        margin={"l": 52, "r": 18, "t": 42, "b": 44},
        showlegend=False,
        hovermode="x unified",
        uirevision=f"epoch-review-{epoch_row.get('epoch_id', t0_s)}",
    )
    return fig


def epoch_review_summary(rec: dict[str, Any], epoch_row: pd.Series, position: int, count: int):
    t0_s = float(epoch_row["t0_s"])
    t1_s = float(epoch_row["t1_s"])
    midpoint = (t0_s + t1_s) / 2.0
    final_state = normalize_state_label(epoch_row.get("final_state", "Undefined"))
    som_state = _state_at_time(rec.get("som"), "somnotate_state", midpoint)
    manual_state = _state_at_time(rec.get("manual"), "manual_state", midpoint)
    layer1_state = _state_at_time(rec.get("layer1"), "layer1_label", midpoint)
    if layer1_state == "Sleep":
        layer1_state = "Layer 1 Sleep"

    def badge(label: str, value: str):
        color = STATE_COLORS.get(value, STATE_COLORS["Undefined"])
        return html.Span(
            [html.Span(f"{label}: ", style={"fontWeight": "600"}), html.Span(value)],
            style={
                "display": "inline-block",
                "padding": "4px 8px",
                "margin": "2px 4px 2px 0",
                "borderRadius": "999px",
                "border": f"1px solid {color}",
                "background": color + "18" if str(color).startswith("#") else "transparent",
            },
        )

    return html.Div([
        html.Div(
            f"Epoch {position + 1} of {count} · ID {epoch_row.get('epoch_id', position)} · "
            f"{_format_recording_clock(t0_s)}–{_format_recording_clock(t1_s)} "
            f"({t1_s - t0_s:.3f} s)",
            style={"fontWeight": "700", "marginBottom": "5px"},
        ),
        html.Div([
            badge("Final", final_state),
            badge("Somnotate", som_state),
            badge("Manual", manual_state),
            badge("L1", layer1_state),
        ]),
    ])


def load_video_metadata(project_root: str | Path | None, recording_id: str | None) -> tuple[str, float]:
    if not project_root or not recording_id:
        return "", 0.0
    try:
        rd = recording_dir_from_manifest(project_root, recording_id)
        meta = read_json(rd / "metadata.json")
        # ``video_file`` is the preferred browser-playback path.  Older metadata
        # or interrupted conversions may only contain ``video_source_file``.
        video_file = str(meta.get("video_file", "") or meta.get("video_source_file", "") or "").strip()
        return video_file, float(meta.get("video_offset_s", 0.0) or 0.0)
    except Exception:
        return "", 0.0


def save_video_metadata(project_root: str | Path, recording_id: str, video_file: str, video_offset_s: float) -> tuple[bool, str]:
    try:
        rd = recording_dir_from_manifest(project_root, recording_id)
        meta_path = rd / "metadata.json"
        meta = read_json(meta_path)
        video_file = str(video_file or "").strip()

        # Never silently erase a valid per-recording video because a dynamic Dash
        # input briefly reported an empty value while the tab was re-rendering.
        if not video_file:
            existing = str(meta.get("video_file", "") or meta.get("video_source_file", "") or "").strip()
            if existing:
                return False, f"No video path was received; kept the existing saved video:\n{existing}"
            return False, "Video path is empty. Paste the full local/network path before clicking Save video."

        meta["video_file"] = video_file
        meta["video_offset_s"] = float(video_offset_s or 0.0)
        write_json(meta_path, meta)

        source = Path(video_file).expanduser()
        if not source.exists():
            return True, f"Saved video settings, but Python cannot currently access this file:\n{video_file}"
        return True, f"Saved video settings for {recording_id}:\n{video_file}\n{video_format_message(video_file)}"
    except Exception as e:
        return False, f"Could not save video settings: {type(e).__name__}: {e}"


def browser_mp4_path_for_avi(avi_path: Path) -> Path:
    """Backward-compatible alias for the generic browser-copy naming helper."""
    return browser_mp4_path_for_video(avi_path)


def save_video_source_metadata(
    project_root: str | Path,
    recording_id: str,
    source_video_file: str | Path,
    browser_video_file: str | Path,
) -> tuple[bool, str]:
    """Record both local source and browser-compatible video paths."""
    try:
        rd = recording_dir_from_manifest(project_root, recording_id)
        meta_path = rd / "metadata.json"
        meta = read_json(meta_path)
        meta["video_source_file"] = str(Path(source_video_file).expanduser().resolve())
        meta["video_file"] = str(Path(browser_video_file).expanduser().resolve())
        meta["video_conversion"] = {
            "format": "mp4_h264_yuv420p",
            "storage": "local_beside_source",
        }
        write_json(meta_path, meta)
        return True, "Saved the source video and browser-compatible MP4 paths in metadata.json."
    except Exception as e:
        return False, f"Could not save video conversion metadata: {type(e).__name__}: {e}"


def convert_video_to_browser_mp4(video_file: str | Path) -> tuple[bool, str, str | None]:
    """Create an H.264/yuv420p MP4 that is reliable in Chrome/Edge/Safari.

    This works for AVI, MOV and MP4 sources, including MP4 files encoded with
    codecs such as MPEG-4 Part 2 (``mp4v``) that can produce a black HTML5 video
    element even though the browser can read the duration.  The source file is
    never modified.
    """
    raw = str(video_file or "").strip()
    if not raw:
        return False, "Choose a video file first.", None

    source = Path(raw).expanduser().resolve()
    if not source.exists() or not source.is_file():
        return False, f"Video file not found: {source}", None

    if source.name.lower().endswith("_browser.mp4"):
        info = _ffprobe_video_info(source)
        if str(info.get("codec_name", "")).lower() == "h264":
            return True, f"This is already a browser-compatible MP4:\n{source}", str(source)

    out_path = browser_mp4_path_for_video(source)

    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            if out_path.stat().st_mtime >= source.stat().st_mtime:
                info = _ffprobe_video_info(out_path)
                if not info or str(info.get("codec_name", "")).lower() == "h264":
                    return True, f"Reusing existing browser-compatible MP4:\n{out_path}", str(out_path)
        except OSError:
            pass

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return (
            False,
            "FFmpeg was not found in the active environment. Update the app environment from environment.yml and reactivate sleep_stage_qc_v2.",
            None,
        )

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(source),
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
        result = subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"Could not run FFmpeg: {type(e).__name__}: {e}", None

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        if len(err) > 3500:
            err = err[-3500:]
        return False, f"FFmpeg conversion failed. Terminal output:\n{err}", None

    if not out_path.exists() or out_path.stat().st_size == 0:
        return False, f"FFmpeg finished but no usable MP4 was created: {out_path}", None

    info = _ffprobe_video_info(out_path)
    codec = str(info.get("codec_name", "") or "").lower()
    pix_fmt = str(info.get("pix_fmt", "") or "").lower()
    return (
        True,
        "Created browser-compatible H.264 MP4. The original video was not changed.\n"
        f"Browser MP4: {out_path}\nCodec: {codec or 'h264'} / {pix_fmt or 'yuv420p'}",
        str(out_path),
    )


def convert_avi_to_browser_mp4(video_file: str | Path) -> tuple[bool, str, str | None]:
    """Compatibility wrapper retained for older callbacks/tests."""
    return convert_video_to_browser_mp4(video_file)


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
    """Resolve a recording folder without tying a project to one computer/OS.

    The canonical location is ``<project_root>/recordings/<recording_id>``.
    Older manifests may contain an absolute ``recording_dir`` written on a
    different machine (for example ``/Volumes/...`` on macOS).  Prefer the
    canonical folder when it exists, then use a valid manifest path as a
    backwards-compatible fallback.
    """
    project_root = Path(project_root).expanduser().resolve()
    canonical = project_root / "recordings" / str(recording_id)

    # This makes a copied/moved project portable between macOS, Windows,
    # external-drive letters, OneDrive locations, etc.
    if canonical.exists():
        return canonical

    manifest = load_manifest(project_root)
    if manifest is not None and len(manifest) and "recording_id" in manifest.columns:
        m = manifest[manifest["recording_id"].astype(str) == str(recording_id)]
        if len(m) and "recording_dir" in m.columns:
            raw = m.iloc[0]["recording_dir"]
            if pd.notna(raw) and str(raw).strip():
                candidate = Path(str(raw)).expanduser()
                if not candidate.is_absolute():
                    candidate = project_root / candidate
                if candidate.exists():
                    return candidate.resolve()

    # Return the expected canonical path so any later FileNotFoundError points
    # to the current project root rather than to a stale path from another OS.
    return canonical


def available_recordings(project_root: str | Path | None) -> list[dict[str, str]]:
    manifest = load_manifest(project_root)
    if manifest is None or len(manifest) == 0 or "recording_id" not in manifest.columns:
        return []
    return [{"label": str(x), "value": str(x)} for x in manifest["recording_id"].astype(str).tolist()]


def somnotate_recording_options(project_root: str | Path | None, *, require_manual: bool = False) -> list[dict[str, Any]]:
    """Build user-friendly Somnotate recording choices with readiness checks.

    Every manifest recording is shown, but entries that cannot currently be used
    are disabled with a reason in the label. This avoids launching a long
    Somnotate command only to fail later because a copied/moved recording is
    missing prepared files or manual scoring.
    """
    options: list[dict[str, Any]] = []
    for item in available_recordings(project_root):
        recording_id = str(item["value"])
        label = recording_id
        disabled = False
        try:
            rec_dir = recording_dir_from_manifest(project_root, recording_id)
            required = ["metadata.json", "eeg.npy", "emg.npy"]
            missing = [name for name in required if not (rec_dir / name).exists()]
            if missing:
                disabled = True
                label = f"{recording_id} — not prepared ({', '.join(missing)} missing)"
            elif require_manual and not (rec_dir / "manual_scoring_aligned.csv").exists():
                disabled = True
                label = f"{recording_id} — no manual scoring"
            elif require_manual:
                label = f"{recording_id} — ready for model QC"
            else:
                label = f"{recording_id} — ready"
        except Exception:
            disabled = True
            label = f"{recording_id} — recording path unavailable"
        options.append({"label": label, "value": recording_id, "disabled": disabled})
    return options


def _model_metadata(model_path: Path) -> dict[str, Any]:
    candidates = [model_path.with_suffix(".metadata.json"), model_path.with_name(model_path.name + ".metadata.json")]
    for meta_path in candidates:
        if meta_path.exists():
            try:
                return read_json(meta_path)
            except Exception:
                return {}
    return {}


def available_models(project_root: str | Path | None = None) -> list[dict[str, str]]:
    """Return Somnotate .pickle models from the app and the loaded project.

    Models trained from inside the app are saved under
    project_root/somnotate_models, while lab-shared models can still live in
    APP_DIR/somnotate_models.
    """
    folders = []
    if SOMNOTATE_MODELS_DIR.exists():
        folders.append(SOMNOTATE_MODELS_DIR)
    if project_root:
        p = Path(project_root).expanduser() / "somnotate_models"
        if p.exists():
            folders.append(p)

    seen = set()
    models = []
    for folder in folders:
        for p in sorted(folder.glob("*.pickle")):
            rp = str(p.expanduser().resolve())
            if rp in seen:
                continue
            seen.add(rp)
            # Include model compatibility metadata when available.
            meta = _model_metadata(p)
            epoch = meta.get("somnotate_epoch_sec", meta.get("epoch_sec"))
            legacy = bool(meta.get("legacy_model") or meta.get("legacy_unverified"))
            bits = [p.name]
            if epoch is not None:
                bits.append(f"{float(epoch):g}s")
            if legacy:
                bits.append("LEGACY")
            bits.append(p.parent.name)
            label = "  —  ".join(bits)
            models.append({"label": label, "value": str(p)})
    return models


def model_quality_path(model_path: str | Path) -> Path:
    return Path(model_path).expanduser().with_suffix(".quality.json")


def available_quality_reports(project_root: str | Path | None = None) -> list[dict[str, str]]:
    """Return saved model-QC reports from trained models and explicit evaluations."""
    candidates: list[Path] = []
    for item in available_models(project_root):
        quality = model_quality_path(item["value"])
        if quality.exists():
            candidates.append(quality)

    if project_root:
        qc_dir = Path(project_root).expanduser() / "somnotate_model_qc"
        if qc_dir.exists():
            candidates.extend(qc_dir.glob("*.quality.json"))

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for report_path in candidates:
        key = str(report_path.resolve()) if report_path.exists() else str(report_path)
        if key in seen or not report_path.exists():
            continue
        seen.add(key)
        try:
            report = read_json(report_path)
        except Exception:
            report = {}
        model_name = Path(str(report.get("model_file") or report_path.stem)).name
        report_type = str(report.get("report_type") or "training-qc")
        if report_type == "existing-model-evaluation":
            context = str(report.get("evaluation_context") or "evaluation")
            context_label = "independent validation" if context == "independent-validation" else "training/unknown data"
            label = f"{model_name} — evaluation ({context_label})"
        else:
            label = f"{model_name} — training QC"
        try:
            stamp = report_path.stat().st_mtime
        except Exception:
            stamp = 0.0
        out.append({"label": label, "value": str(report_path), "_mtime": stamp})

    out.sort(key=lambda x: x.get("_mtime", 0.0), reverse=True)
    return [{"label": x["label"], "value": x["value"]} for x in out]


def _pct_text(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):.1f}%"
    except Exception:
        return "n/a"


def _quality_metric_card(label: str, value: str, note: str = ""):
    return html.Div(
        style={
            "border": "1px solid #d7dce3",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "minWidth": "150px",
            "background": "rgba(255,255,255,0.6)",
        },
        children=[
            html.Div(label, style={"fontSize": "12px", "fontWeight": "700", "opacity": 0.75}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "marginTop": "2px"}),
            html.Div(note, style={"fontSize": "11px", "opacity": 0.7, "marginTop": "2px"}) if note else None,
        ],
    )


def render_quality_evaluation(block: dict[str, Any] | None, title: str):
    if not block:
        return html.Div([html.H5(title), html.Div("Not available for this report.", className="status-line")])

    cards = html.Div(
        style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"},
        children=[
            _quality_metric_card("Mean accuracy", _pct_text(block.get("mean_accuracy")), f"{block.get('n_recordings', 0)} recordings"),
            _quality_metric_card("Worst recording", _pct_text(block.get("worst_accuracy")), "check animal-to-animal robustness"),
            _quality_metric_card("Balanced accuracy", _pct_text(block.get("balanced_accuracy")), "mean Wake/NREM/REM recall"),
            _quality_metric_card("Macro F1", _pct_text(block.get("macro_f1")), "equal weight to Wake/NREM/REM"),
        ],
    )

    per_state = pd.DataFrame(block.get("per_state", []))
    state_table = html.Div()
    if len(per_state):
        rows = []
        for _, row in per_state.iterrows():
            rows.append(
                html.Tr([
                    html.Td(str(row.get("state", ""))),
                    html.Td(_pct_text(row.get("precision"))),
                    html.Td(_pct_text(row.get("recall"))),
                    html.Td(_pct_text(row.get("f1"))),
                    html.Td(str(int(row.get("support", 0)))),
                ])
            )
        state_table = html.Table(
            style={"borderCollapse": "collapse", "width": "100%", "fontSize": "13px"},
            children=[
                html.Thead(html.Tr([html.Th("State"), html.Th("Precision"), html.Th("Recall"), html.Th("F1"), html.Th("Epochs")])),
                html.Tbody(rows),
            ],
        )

    rec_df = pd.DataFrame(block.get("per_recording", []))
    rec_table = html.Div()
    if len(rec_df):
        rec_table = dash_table.DataTable(
            data=[{"recording_id": str(r["recording_id"]), "accuracy": _pct_text(r["accuracy"])} for _, r in rec_df.iterrows()],
            columns=[{"name": "Recording", "id": "recording_id"}, {"name": "Accuracy", "id": "accuracy"}],
            page_size=min(10, len(rec_df)),
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "12px", "padding": "5px", "textAlign": "left"},
        )

    labels = block.get("confusion_labels", [])
    matrix = block.get("confusion_matrix", [])
    confusion = html.Div()
    if labels and matrix:
        z = np.asarray(matrix, dtype=float)
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                text=z.astype(int),
                texttemplate="%{text}",
                showscale=False,
                hovertemplate="Manual %{y}<br>Predicted %{x}<br>Epochs %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Confusion matrix",
            xaxis_title="Predicted",
            yaxis_title="Manual",
            height=330,
            margin=dict(l=60, r=20, t=55, b=55),
            template="plotly_white",
        )
        confusion = dcc.Graph(figure=fig, config={"displayModeBar": False})

    return html.Div(
        style={"marginTop": "12px"},
        children=[
            html.H5(title),
            cards,
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "minmax(300px, 1fr) minmax(300px, 1fr)", "gap": "14px"},
                children=[
                    html.Div([html.B("Per-state performance"), state_table]),
                    html.Div([html.B("Per-recording performance"), rec_table]),
                ],
            ),
            confusion,
        ],
    )


def render_model_quality_report(report_file: str | Path | None):
    if not report_file:
        return html.Div("No saved model-quality report selected yet.", className="status-line")
    report_path = Path(str(report_file)).expanduser()
    if not report_path.exists():
        return html.Div(f"Quality report not found: {report_path}", className="status-line")
    try:
        report = read_json(report_path)
    except Exception as exc:
        return html.Div(f"Could not read quality report: {type(exc).__name__}: {exc}", className="status-line")

    model = Path(str(report.get("model_file") or "unknown_model"))
    report_type = str(report.get("report_type") or "training-qc")
    context = str(report.get("evaluation_context") or "")
    if report_type == "existing-model-evaluation":
        if context == "independent-validation":
            evidence = html.Div(
                "Independent validation: these recordings were declared not to have been used for training. This is the most informative simple evaluation of a fixed model.",
                className="status-line",
                style={"marginTop": "8px", "fontWeight": "600"},
            )
        else:
            evidence = html.Div(
                "Training/unknown-data evaluation: these scores are descriptive and may be optimistic. They should not be presented as independent validation unless the recordings were genuinely held out from training.",
                className="status-line",
                style={"marginTop": "8px", "fontWeight": "600"},
            )
        evaluations = [render_quality_evaluation(report.get("evaluation"), "Existing-model evaluation")]
    else:
        evidence = html.Div(
            "Training QC: recording-level cross-validation estimates generalization within the supplied training cohort. Independent held-out recordings provide stronger confirmation when available.",
            className="status-line",
            style={"marginTop": "8px"},
        )
        evaluations = [
            render_quality_evaluation(report.get("cross_validation"), "Leave-one-recording-out cross-validation"),
            render_quality_evaluation(report.get("heldout_test"), "Independent held-out test recordings"),
        ]

    guidance = report.get("guidance", [])
    return html.Div(
        children=[
            html.Div([html.B("Model: "), html.Span(model.name)]),
            html.Div([html.B("QC report: "), html.Span(str(report_path))], style={"fontSize": "12px", "opacity": 0.8}),
            evidence,
            *evaluations,
            html.Div(
                className="status-line",
                style={"marginTop": "12px", "whiteSpace": "normal"},
                children=[
                    html.B("How to judge the model"),
                    html.Ul([html.Li(str(x)) for x in guidance]) if guidance else html.Div(
                        "Inspect recording-level performance and Wake/NREM/REM metrics; do not rely on overall accuracy alone."
                    ),
                ],
            ),
        ]
    )


def state_display_codes(labels: list[str] | np.ndarray, row_name: str = "") -> np.ndarray:
    out = []
    for x in labels:
        sx = normalize_state_label(x)
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


def adaptive_spectrogram_params(window_s: float) -> tuple[float, float, int]:
    """Choose STFT parameters based on the visible QC window.

    A 4 s STFT window is useful for broad sleep review, but it becomes very
    blocky when the user zooms into a short 20–60 s interval. These settings
    trade frequency resolution for time resolution only when the visible window
    is short.
    """
    window_s = float(window_s or 0.0)

    if window_s <= 60.0:
        # Fine temporal view: ~0.25 s hop at 75% overlap. Frequency bins are
        # coarser, but the spectrogram remains informative when zoomed in.
        return 1.0, 0.75, 1200

    if window_s <= 180.0:
        # Intermediate view: good balance for transition inspection.
        return 2.0, 0.75, 1200

    # Broad sleep-review view: stable spectral estimate and manageable size.
    return 4.0, 0.75, 900


def compute_eeg_spectrogram_window(
    npy_path: Path,
    fs: float,
    start_s: float,
    end_s: float,
    max_freq_hz: float = 30.0,
    nperseg_s: float | None = None,
    overlap_fraction: float | None = None,
    max_time_bins: int | None = None,
):
    """Compute an EEG spectrogram for the visible review window.

    Returns x in minutes, frequency in Hz, and log-power in dB. The function is
    intentionally windowed so it stays responsive in the Dash QC viewer. For
    short zoomed windows, STFT settings are automatically made more temporal.
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

        if nperseg_s is None or overlap_fraction is None or max_time_bins is None:
            auto_nperseg_s, auto_overlap, auto_max_bins = adaptive_spectrogram_params(float(end_s) - float(start_s))
            if nperseg_s is None:
                nperseg_s = auto_nperseg_s
            if overlap_fraction is None:
                overlap_fraction = auto_overlap
            if max_time_bins is None:
                max_time_bins = auto_max_bins

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
FINAL_TEXT_COLUMNS = {
    "recording_id": "",
    "final_state": "Undefined",
    "final_source": "",
    "review_status": "",
    "review_notes": "",
}


def normalize_final_scoring_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Final scoring writable and normalize legacy/non-canonical labels."""
    out = df.copy()
    for col, default in FINAL_TEXT_COLUMNS.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].astype(object)

    # Older/imported Somnotate values can be stored as e.g. ``awake`` and
    # ``non-REM``. Canonicalize these so display and exported numeric codes
    # remain correct even for a Final file written before this fix.
    if "final_state" in out.columns:
        canonical = out["final_state"].map(normalize_state_label)
        canonical = canonical.replace({"Sleep": "NREM"})
        out["final_state"] = canonical.astype(object)
        out["final_code"] = [FINAL_EXPORT_CODE.get(str(x), -1) for x in canonical]
    return out


def read_final_scoring(path: str | Path) -> pd.DataFrame:
    return normalize_final_scoring_dtypes(pd.read_csv(path))


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
    if som is not None:
        # Accept both current canonical app labels and raw labels emitted by
        # Somnotate versions such as "awake" and "non-REM". This keeps old
        # imported result files usable after moving/upgrading the app.
        if "somnotate_state" in som.columns:
            som["somnotate_state"] = som["somnotate_state"].map(normalize_state_label)
        rename_prob = {}
        for col in som.columns:
            if str(col).startswith("somnotate_P_"):
                suffix = str(col)[len("somnotate_P_"):]
                canon = normalize_state_label(suffix)
                if canon in {"Wake", "NREM", "REM"}:
                    rename_prob[col] = f"somnotate_P_{canon}"
        if rename_prob:
            som = som.rename(columns=rename_prob)
    features_file = recording_dir / "epoch_features.csv"
    features = pd.read_csv(features_file) if features_file.exists() else None
    final_file = ensure_final_scoring(recording_dir, recording_id)
    final = read_final_scoring(final_file)
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
    # 5 Photometry if available
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
    visible_window_s = max(0.0, (end_min - float(start_min)) * 60.0)
    spec_nperseg_s, spec_overlap, spec_max_bins = adaptive_spectrogram_params(visible_window_s)

    spec = compute_eeg_spectrogram_window(
        rec["recording_dir"] / "eeg.npy",
        fs,
        start_min * 60,
        end_min * 60,
        max_freq_hz=30.0,
        nperseg_s=spec_nperseg_s,
        overlap_fraction=spec_overlap,
        max_time_bins=spec_max_bins,
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
        try:
            fig.layout.annotations[spec_row - 1].text = (
                f"EEG spectrogram (0.5–20 Hz; STFT {spec_nperseg_s:g} s)"
            )
        except Exception:
            pass
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
    # Photometry — black, before probabilities
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
    final = read_final_scoring(final_file)
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
    final = read_final_scoring(final_file)
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
    previous = normalize_final_scoring_dtypes(pd.read_csv(snap))
    final = read_final_scoring(final_file)
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
    final = read_final_scoring(final_file)
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
    final = read_final_scoring(final_file)
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


def _edf_safe_label(label: str, fallback: str) -> str:
    """EDF signal labels are short; keep them readable and browser/Sirenia-safe."""
    s = str(label or fallback).strip() or fallback
    # EDF labels are conventionally limited to 16 ASCII characters.
    s = "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in s)
    return s[:16] or fallback[:16]


def _edf_physical_range(values: np.ndarray) -> tuple[float, float]:
    """Choose a safe physical min/max for EDF export."""
    y = np.asarray(values, dtype=float)
    finite = y[np.isfinite(y)]

    if finite.size == 0:
        return -1.0, 1.0

    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0

    if hi <= lo:
        pad = max(abs(lo) * 0.01, 1.0)
        return lo - pad, hi + pad

    pad = max((hi - lo) * 0.001, 1e-9)
    return lo - pad, hi + pad


def _load_signal_for_edf(path: Path) -> np.ndarray:
    """Load a saved .npy signal and replace non-finite values before EDF writing."""
    x = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float64).ravel()

    if np.any(~np.isfinite(x)):
        finite = x[np.isfinite(x)]
        fill = float(np.nanmedian(finite)) if finite.size else 0.0
        x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)

    return x


def export_recording_edf(rec: dict[str, Any], final: pd.DataFrame, edf_out: Path) -> tuple[bool, str]:
    """Export EEG/EMG/optional photometry as EDF+ with Final scoring annotations.

    The CSV and MAT exports contain the exact epoch table. The EDF export is mainly
    for opening the recording in EDF-compatible viewers: it writes the signal
    channels plus one EDF+ annotation per reviewed Final epoch. Undefined/Uncertain
    epochs are skipped so the annotation track stays clean.
    """
    if pyedflib is None:
        return False, "EDF export skipped because pyedflib is not installed/importable."

    rd = Path(rec["recording_dir"])
    fs = float(rec["fs"])

    signal_specs: list[tuple[str, Path, float, str]] = [
        ("EEG", rd / "eeg.npy", fs, "uV"),
        ("EMG", rd / "emg.npy", fs, "uV"),
    ]

    phot = find_photometry(rec)
    if phot is not None:
        phot_path, phot_fs, phot_label = phot
        # Avoid adding EEG/EMG twice if metadata points there accidentally.
        if Path(phot_path).resolve() not in {(rd / "eeg.npy").resolve(), (rd / "emg.npy").resolve()}:
            signal_specs.append((str(phot_label or "Photometry"), Path(phot_path), float(phot_fs), "a.u."))

    signals: list[np.ndarray] = []
    headers: list[dict[str, Any]] = []

    for i, (label, path, sig_fs, dimension) in enumerate(signal_specs):
        if not path.exists():
            continue

        y = _load_signal_for_edf(path)
        if y.size == 0:
            continue

        physical_min, physical_max = _edf_physical_range(y)
        signals.append(y)
        headers.append({
            "label": _edf_safe_label(label, f"ch{i+1}"),
            "dimension": dimension,
            "sample_frequency": float(sig_fs),
            "physical_min": float(physical_min),
            "physical_max": float(physical_max),
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "",
            "prefilter": "",
        })

    if not signals:
        return False, "EDF export skipped because no eeg.npy/emg.npy signal files were found."

    edf_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = pyedflib.EdfWriter(
            str(edf_out),
            n_channels=len(signals),
            file_type=pyedflib.FILETYPE_EDFPLUS,
        )
        try:
            writer.setSignalHeaders(headers)
            writer.writeSamples(signals)

            exported_annotations = 0
            for _, row in final.iterrows():
                state = str(row.get("final_state", "Undefined") or "Undefined")
                code = int(pd.to_numeric(row.get("final_code", -1), errors="coerce")) if pd.notna(row.get("final_code", -1)) else -1

                if state in {"", "Undefined", "Uncertain", "nan", "None"} or code < 0:
                    continue

                t0 = float(row.get("t0_s", 0.0))
                t1 = float(row.get("t1_s", t0))
                duration = max(0.0, t1 - t0)
                writer.writeAnnotation(t0, duration, state)
                exported_annotations += 1
        finally:
            writer.close()

        return True, f"EDF+ exported with {len(signals)} signal channel(s) and {exported_annotations} Final scoring annotation(s):\n{edf_out}"

    except Exception as e:
        return False, f"EDF export failed: {type(e).__name__}: {e}"


def export_final(project_root: str, recording_id: str):
    rec = load_recording(project_root, recording_id)
    final_file = ensure_final_scoring(rec["recording_dir"], rec["recording_id"])
    final = read_final_scoring(final_file)
    out_dir = rec["recording_dir"] / "exports"
    out_dir.mkdir(exist_ok=True)

    csv_out = out_dir / f"{recording_id}_final_scoring_dash.csv"
    mat_out = out_dir / f"{recording_id}_final_scoring_dash.mat"
    edf_out = out_dir / f"{recording_id}_signals_with_final_scoring.edf"

    messages = []

    final.to_csv(csv_out, index=False)
    messages.append(f"CSV exported:\n{csv_out}")

    if savemat is not None:
        savemat(
            mat_out,
            {
                "scoring": final["final_code"].to_numpy(dtype=np.int16),
                "final_code": final["final_code"].to_numpy(dtype=np.int16),
                "final_state": final["final_state"].fillna("Undefined").astype(str).to_numpy(dtype=object),
                "t0_s": final["t0_s"].to_numpy(float),
                "t1_s": final["t1_s"].to_numpy(float),
            },
        )
        messages.append(f"MAT exported:\n{mat_out}")
    else:
        messages.append("MAT export skipped because scipy.io.savemat is unavailable.")

    ok_edf, edf_msg = export_recording_edf(rec, final, edf_out)
    messages.append(edf_msg)

    return True, "\n\n".join(messages)



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


app = Dash(__name__, suppress_callback_exceptions=True, title=f"Sleep Stage QC {APP_VERSION}")


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

    response = send_file(path, mimetype=mimetype, conditional=True, as_attachment=False)
    # Explicitly advertise byte ranges because HTML5 video seeking depends on
    # them, especially for large recordings on mapped/network drives.
    response.headers.setdefault("Accept-Ranges", "bytes")
    # The URL contains the file mtime, so it is safe (and much faster for large
    # network videos) to let the browser cache byte ranges. A changed/reconverted
    # file receives a different URL automatically.
    response.headers["Cache-Control"] = "private, max-age=86400"
    return response


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
                        f"Version {APP_VERSION} · Interactive review, model comparison, dissociation QC and export for EEG/EMG sleep scoring.",
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
    PInput(id="mat-file"), PInput(id="import-recording-id"),
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
    html.Div(id="epoch-review-panel"), dcc.Graph(id="epoch-review-graph"),
    html.Button(id="epoch-review-prev"), html.Button(id="epoch-review-play-selection"),
    html.Button(id="epoch-review-pause"), html.Button(id="epoch-review-replay"), html.Button(id="epoch-review-next"),
    html.Div(id="epoch-review-summary"), html.Div(id="epoch-review-live-position"),
    html.Div(id="epoch-review-video-container"), html.Div(id="epoch-review-cache-status"),
    html.Div(id="epoch-review-video-feedback"), html.Div(id="epoch-review-video-diagnostics"),
    dcc.Store(id="epoch-review-index-store"), dcc.Store(id="epoch-review-clip-store"),
    dcc.Store(id="epoch-review-seek-store"), dcc.Store(id="epoch-review-playback-command-store"),
    dcc.Store(id="epoch-review-playback-time-store"), dcc.Store(id="epoch-review-rendered-position-store"),
    dcc.Interval(id="epoch-review-clock"), dcc.Interval(id="epoch-review-playhead-clock"),
    html.Button(id="clear-video-review-cache"),

    # Somnotate tab
    PInput(id="som-recording-ids"), PInput(id="som-target-fs"), PDropdown(id="som-epoch-sec"), PInput(id="som-root"),
    PInput(id="som-conda-env"), PInput(id="som-python"), PDropdown(id="som-model-file"), PInput(id="som-model-file-custom"),
    html.Div(id="som-epoch-warning"), html.Div(id="som-existing-epoch-summary"), html.Div(id="som-train-epoch-summary"),
    dcc.Checklist(id="som-existing-steps"), html.Button(id="btn-som-existing"),
    PInput(id="som-eval-ids"), PDropdown(id="som-eval-context"), dcc.Checklist(id="som-eval-steps"), html.Button(id="btn-som-evaluate"),
    PInput(id="som-train-ids"), PInput(id="som-test-ids"), PInput(id="som-model-name"),
    dcc.Checklist(id="som-train-steps"), html.Button(id="btn-som-train"), html.Button(id="btn-som-import-results"),
    PDropdown(id="som-qc-report-file"), html.Div(id="som-model-qc"),
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
                html.Div([html.Label("Photometry variable or EDF channel, optional"), PInput(id="ach-key", type="text", value="ne", style={"width":"100%"})]),
                html.Div([html.Label("EEG sampling frequency variable, optional"), PInput(id="eeg-fs-key", type="text", value="eeg_frequency", style={"width":"100%"})]),
                html.Div([html.Label("Photometry sampling frequency variable, optional"), PInput(id="ach-fs-key", type="text", value="ne_frequency", style={"width":"100%"})]),
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

                html.Div(className="qc-plot-with-actions", children=[
                    html.Div(className="qc-plot-main", children=[
                        html.Div(id="empty-qc-message", children=[html.H4("No recording loaded yet"), html.P("Load a project and choose a recording first.")], className="empty-panel"),
                        dcc.Graph(id="qc-graph", style={"display":"none"}, config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False, "modeBarButtonsToAdd": ["select2d", "pan2d", "zoom2d", "resetScale2d"]}),
                        html.Div("Tip: use mouse wheel / trackpad scroll over the QC plot to zoom; press P to pan and S to select scoring windows.", className="app-subtitle", style={"marginTop": "4px"}),
                        html.Div(id="selected-interval-label", className="status-line"),
                    ]),
                    html.Div(className="qc-window-actions", children=[
                        html.Div("Apply", className="qc-window-actions-title"),
                        html.Div("visible window", className="qc-window-actions-subtitle"),
                        html.Button(
                            "Somnotate",
                            id="score-window-somnotate",
                            title="Apply Somnotate scoring to the whole visible window",
                        ),
                        html.Button(
                            "L1",
                            id="score-window-layer1",
                            title="Apply Layer 1 scoring to the whole visible window",
                        ),
                        html.Button(
                            "Manual",
                            id="score-window-manual",
                            title="Apply manual scoring to the whole visible window",
                        ),
                    ]),
                ]),
                html.Div(id="score-status", className="status-line", style={"whiteSpace":"pre-wrap", "marginBottom":"12px"}),

                html.Div(
                    id="epoch-review-panel",
                    style={"display": "none"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "alignItems": "flex-start", "flexWrap": "wrap"},
                            children=[
                                html.Div([
                                    html.H4("Synchronized epoch review", style={"margin": "0 0 3px 0"}),
                                    html.Div(
                                        "Select an interval in the main QC plot. The video plays the selection; use Prev/Next to inspect EEG/EMG one scoring epoch at a time.",
                                        className="app-subtitle",
                                    ),
                                ], style={"flex": "1 1 520px"}),
                                html.Div(
                                    style={"display": "flex", "gap": "6px", "alignItems": "center", "flexWrap": "wrap"},
                                    children=[
                                        html.Button("◀ Epoch", id="epoch-review-prev", n_clicks=0, title="Previous scoring epoch"),
                                        html.Button("▶ Play selection", id="epoch-review-play-selection", n_clicks=0, title="Play the selected video interval"),
                                        html.Button("⏸ Pause", id="epoch-review-pause", n_clicks=0, title="Pause synchronized playback"),
                                        html.Button("↻ Replay epoch", id="epoch-review-replay", n_clicks=0, title="Replay only the current scoring epoch"),
                                        html.Button("Epoch ▶", id="epoch-review-next", n_clicks=0, title="Next scoring epoch"),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(id="epoch-review-summary", style={"margin": "9px 0 4px 0"}),
                        html.Div(
                            id="epoch-review-live-position",
                            children="Video: waiting for short local QC clip…",
                            className="app-subtitle",
                            style={"margin": "0 0 8px 0", "fontVariantNumeric": "tabular-nums"},
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "12px", "alignItems": "stretch", "flexWrap": "wrap"},
                            children=[
                                html.Div(
                                    dcc.Graph(
                                        id="epoch-review-graph",
                                        config={"displayModeBar": False, "scrollZoom": False, "displaylogo": False},
                                        style={"height": "360px"},
                                    ),
                                    style={"flex": "1 1 48%", "minWidth": "420px"},
                                ),
                                html.Div(
                                    [
                                        dcc.Loading(type="circle", children=html.Div(id="epoch-review-video-container")),
                                        html.Div(id="epoch-review-cache-status", className="status-line", style={"marginTop": "5px", "whiteSpace": "pre-wrap"}),
                                        html.Div(id="epoch-review-video-feedback", className="status-line", style={"marginTop": "5px"}),
                                        html.Div(id="epoch-review-video-diagnostics", className="app-subtitle", style={"marginTop": "3px"}),
                                    ],
                                    style={"flex": "1 1 48%", "minWidth": "360px"},
                                ),
                            ],
                        ),
                        dcc.Store(id="epoch-review-index-store"),
                        dcc.Store(id="epoch-review-clip-store"),
                        dcc.Store(id="epoch-review-seek-store"),
                        dcc.Store(id="epoch-review-playback-command-store"),
                        dcc.Store(id="epoch-review-playback-time-store"),
                        dcc.Store(id="epoch-review-rendered-position-store"),
                        html.Div(id="epoch-review-playhead-dummy", style={"display": "none"}),
                        dcc.Interval(id="epoch-review-clock", interval=200, n_intervals=0),
                        # This faster clock is client-side only: it moves the
                        # Plotly playhead without re-rendering signal data.
                        dcc.Interval(id="epoch-review-playhead-clock", interval=100, n_intervals=0),
                    ],
                ),

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
                            html.Button("Make full browser MP4 (optional)", id="convert-video-mp4", n_clicks=0, title="Optional full-video conversion. Synchronized review normally uses short local QC clips instead."),
                            html.Button("Clear local review cache", id="clear-video-review-cache", n_clicks=0, title="Delete short local QC clips for this recording"),
                            dcc.Loading(type="circle", children=html.Div(id="video-status", className="status-line")),
                        ],
                    ),
                    html.Div(
                        "Synchronized epoch review automatically creates short, seek-friendly local QC clips from the selected interval, so a 10-hour source video does not need a full 30-minute transcode. The original video can stay on a network drive or local disk. Full-video conversion below is optional and mainly useful for the standalone player.",
                        className="app-subtitle",
                        style={"marginTop": "6px"},
                    ),
                    html.Div(id="video-player-container", style={"marginTop": "10px"}),
                    dcc.Store(id="video-seek-store"),
                    html.Div(id="video-seek-feedback", className="status-line"),
                    html.Details(children=[
                        html.Summary("How browser-compatible conversion works"),
                        html.Div(
                            "For synchronized review, the app re-encodes only a short selected region into the local cache with H.264 and one-second keyframes. If you need the standalone full-video player to be browser-compatible, use the optional full-video conversion above.",
                            className="app-subtitle",
                            style={"marginTop": "6px", "marginBottom": "6px"},
                        ),
                        html.Pre(
                            'ffmpeg -i "videoname.avi" -map 0:v:0 -an -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 -movflags +faststart "videoname.mp4"',
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
                    html.Button("Export final scoring CSV + MAT + EDF", id="btn-export"),
                    html.Button("Reset Final to empty", id="btn-reset-final-empty"),
                    html.Div("Shortcuts: P Pan, S Select window, 1 Wake, 2 NREM, 3 REM, A Somnotate/automatic, L Layer 1, M Manual"),
                ]),
                html.Div(className="card", style={"marginTop": "16px"}, children=[
                    html.H4("Final scoring utilities"),
                    html.Div(
                        "Use these at the end of review, or to initialise an empty Final row from Somnotate. Existing reviewed labels are not overwritten. The export button writes CSV, MAT, and EDF+ when pyedflib is available.",
                        className="app-subtitle",
                    ),
                    html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr 1fr", "gap":"8px", "marginTop":"8px"}, children=[
                        html.Button("Fill empty Final with Somnotate", id="btn-fill-empty-somnotate", n_clicks=0),
                        html.Button("Fill empty Final with Somnotate + export", id="btn-fill-empty-somnotate-export", n_clicks=0),
                        html.Button("Export final scoring CSV + MAT + EDF", id="btn-export-bottom", n_clicks=0),
                    ]),
                ]),
            ]),
        ])

    if tab == "tab-somnotate":
        models = available_models(project_root)
        quality_reports = available_quality_reports(project_root)
        som_recording_options = somnotate_recording_options(project_root, require_manual=False)
        som_manual_options = somnotate_recording_options(project_root, require_manual=True)
        ready_score_ids = [str(o["value"]) for o in som_recording_options if not o.get("disabled")]
        section_style = {
            "border": "1px solid #d7dce3",
            "borderRadius": "10px",
            "padding": "14px",
            "marginTop": "14px",
        }
        helper_style = {"fontSize": "12px", "opacity": 0.78, "marginTop": "4px", "lineHeight": "1.4"}
        return html.Div(className="card", children=[
            html.H3("Somnotate"),
            html.Div(
                "Score recordings with an existing model, evaluate an already-trained model against manual scoring, or train a new model with recording-level quality control.",
                className="app-subtitle",
            ),
            html.Div(
                f"Tested upstream: Somnotate {TESTED_SOMNOTATE_VERSION} at commit {TESTED_SOMNOTATE_COMMIT[:7]}. The app modifies only a temporary pipeline copy.",
                style={"fontSize":"12px", "opacity":0.72, "marginTop":"4px"},
            ),

            html.Div(style=section_style, children=[
                html.H4("Somnotate setup", style={"marginTop":"0"}),
                html.Div("These settings are shared by scoring, evaluation and training.", className="app-subtitle"),
                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"10px", "marginTop":"10px"}, children=[
                    html.Div([html.Label("Target sampling rate (Hz)"), PInput(id="som-target-fs", type="number", value=512.0, style={"width":"100%"})]),
                    html.Div([html.Label("Somnotate epoch length"), PDropdown(
                        id="som-epoch-sec",
                        options=[
                            {"label": "1 s epochs", "value": "1.0"},
                            {"label": "2 s epochs", "value": "2.0"},
                            {"label": "5 s epochs (legacy models)", "value": "5.0"},
                        ],
                        value="5.0",
                        clearable=False,
                        style={"width":"100%"},
                    )]),
                    html.Div([html.Label("Somnotate repository"), PInput(id="som-root", type="text", value=DEFAULT_SOMNOTATE_ROOT, placeholder=str(Path.home() / "somnotate"), style={"width":"100%"})]),
                    html.Div([html.Label("Somnotate Conda environment"), PInput(id="som-conda-env", type="text", value="somnotate_env", style={"width":"100%"})]),
                    html.Div(style={"gridColumn":"1 / -1"}, children=[
                        html.Label("Optional Somnotate Python executable"),
                        PInput(id="som-python", type="text", style={"width":"100%"}),
                        html.Div("Usually leave this empty; the app resolves the Conda environment automatically.", style=helper_style),
                    ]),
                ]),
            ]),

            html.Div(style=section_style, children=[
                html.H4("Existing model", style={"marginTop":"0"}),
                html.Div("Select one model here. The same model is used by both Score and Evaluate below.", className="app-subtitle"),
                html.Div(style={"display":"grid", "gridTemplateColumns":"1fr 1fr", "gap":"10px", "marginTop":"10px"}, children=[
                    html.Div([
                        html.Label("Model from app/project"),
                        PDropdown(id="som-model-file", options=models, value=models[0]["value"] if models else None, placeholder="Choose a .pickle model"),
                    ]),
                    html.Div([
                        html.Label("Or external model path"),
                        PInput(id="som-model-file-custom", type="text", placeholder=r"C:\path\to\model.pickle or /path/to/model.pickle", style={"width":"100%"}),
                        html.Div("If filled, this path takes priority over the dropdown.", style=helper_style),
                    ]),
                ]),
                html.Div(
                    id="som-epoch-warning",
                    className="status-line",
                    style={"whiteSpace": "pre-wrap", "marginTop": "8px"},
                    children=(
                        "Somnotate models are epoch-length specific. The selected epoch length must match the model used for scoring/evaluation. "
                        "New models trained here save this metadata automatically."
                    ),
                ),
            ]),

            html.Div(style=section_style, children=[
                html.H4("1. Score recordings with the selected model", style={"marginTop":"0"}),
                html.Div(
                    "Use this when you already have a trained model and want Somnotate predictions/probabilities for new recordings.",
                    className="app-subtitle",
                ),
                html.Div(style={"marginTop":"10px"}, children=[
                    html.Label("Recordings to score"),
                    PDropdown(
                        id="som-recording-ids",
                        options=som_recording_options,
                        value=ready_score_ids[:1],
                        multi=True,
                        placeholder="Choose one or more prepared recordings",
                        style={"width":"100%"},
                    ),
                    html.Div("Unavailable recordings are shown but disabled, with the reason in the label.", style=helper_style),
                ]),
                html.Div(
                    id="som-existing-epoch-summary",
                    className="status-line",
                    style={"whiteSpace": "pre-wrap", "margin": "6px 0"},
                    children="Existing-model scoring uses the Somnotate epoch length selected above.",
                ),
                dcc.Checklist(
                    id="som-existing-steps",
                    options=[
                        {"label":" Prepare", "value":"prepare"},
                        {"label":" Preprocess", "value":"preprocess"},
                        {"label":" Score", "value":"score"},
                        {"label":" Probabilities", "value":"probabilities"},
                        {"label":" Import results", "value":"import-results"},
                    ],
                    value=["prepare","preprocess","score","probabilities","import-results"],
                    inline=True,
                ),
                html.Button("Run scoring", id="btn-som-existing", n_clicks=0, style={"marginTop":"8px"}),
            ]),

            html.Div(style=section_style, children=[
                html.H4("2. Evaluate the selected model", style={"marginTop":"0"}),
                html.Div(
                    "No model retraining is done here. The fixed .pickle model is compared directly with manual Wake/NREM/REM scoring.",
                    className="app-subtitle",
                ),
                html.Div(style={"display":"grid", "gridTemplateColumns":"2fr 1fr", "gap":"10px", "marginTop":"10px"}, children=[
                    html.Div([
                        html.Label("Manually scored recordings"),
                        PDropdown(
                            id="som-eval-ids",
                            options=som_manual_options,
                            value=[],
                            multi=True,
                            placeholder="Choose recordings with manual scoring",
                            style={"width":"100%"},
                        ),
                        html.Div("Only prepared recordings with manual scoring can be selected; unavailable entries are disabled with a reason.", style=helper_style),
                    ]),
                    html.Div([
                        html.Label("Relationship to model training"),
                        PDropdown(
                            id="som-eval-context",
                            options=[
                                {"label":"Independent validation — not used to train this model", "value":"independent-validation"},
                                {"label":"Training recordings / not sure", "value":"training-or-unknown"},
                            ],
                            value="independent-validation",
                            clearable=False,
                        ),
                    ]),
                ]),
                html.Div(
                    className="status-line",
                    style={"marginTop":"8px", "whiteSpace":"normal"},
                    children=[
                        html.B("What this tells you: "),
                        html.Span("Independent held-out recordings provide the clearest test of a fixed model. If these recordings were used for training (or you are unsure), the app labels the result as descriptive rather than independent validation."),
                    ],
                ),
                dcc.Checklist(
                    id="som-eval-steps",
                    options=[
                        {"label":" Prepare recordings", "value":"prepare"},
                        {"label":" Preprocess signals", "value":"preprocess"},
                    ],
                    value=["prepare","preprocess"],
                    inline=True,
                    style={"marginTop":"8px"},
                ),
                html.Button("Evaluate model", id="btn-som-evaluate", n_clicks=0, style={"marginTop":"8px"}),
            ]),

            html.Div(style=section_style, children=[
                html.H4("3. Train your own model", style={"marginTop":"0"}),
                html.Div(
                    "Training recordings fit the final model. Optional held-out recordings are excluded from fitting and can be used for independent QC when manual scoring is available.",
                    className="app-subtitle",
                ),
                html.Div(
                    id="som-train-epoch-summary",
                    className="status-line",
                    style={"whiteSpace": "pre-wrap", "margin": "6px 0 8px 0"},
                    children="Training uses the Somnotate epoch length selected above. Training recordings must contain manual scoring.",
                ),
                html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"10px"}, children=[
                    html.Div([
                        html.Label("Training recordings"),
                        PDropdown(id="som-train-ids", options=som_manual_options, value=[], multi=True, placeholder="Choose manually scored training recordings", style={"width":"100%"}),
                        html.Div("Manual scoring required. Use several independent recordings/animals when possible.", style=helper_style),
                    ]),
                    html.Div([
                        html.Label("Held-out validation recordings (optional)"),
                        PDropdown(id="som-test-ids", options=som_manual_options, value=[], multi=True, placeholder="Choose recordings excluded from training", style={"width":"100%"}),
                        html.Div("Keep these out of the training list for an independent final-model check.", style=helper_style),
                    ]),
                    html.Div([html.Label("New model name"), PInput(id="som-model-name", type="text", value="my_somnotate_model", style={"width":"100%"})]),
                ]),
                dcc.Checklist(
                    id="som-train-steps",
                    options=[
                        {"label":" Prepare recordings", "value":"prepare"},
                        {"label":" Preprocess signals", "value":"preprocess"},
                        {"label":" Run recording-level model QC (recommended)", "value":"quality-control"},
                    ],
                    value=["prepare","preprocess","quality-control"],
                    inline=True,
                    style={"marginTop":"8px"},
                ),
                html.Div(
                    "QC uses leave-one-recording-out cross-validation across the training recordings. It does not overwrite the final trained model. If held-out manually scored recordings are supplied, the final model is also tested on them.",
                    style=helper_style,
                ),
                html.Button("Train model + QC", id="btn-som-train", n_clicks=0, style={"marginTop":"8px"}),
            ]),

            html.Div(style=section_style, children=[
                html.H4("Model quality reports", style={"marginTop":"0"}),
                html.Div(
                    "Evaluation and training QC reports are saved so they can be reopened later. The newest report is selected automatically after a successful run.",
                    className="app-subtitle",
                ),
                PDropdown(
                    id="som-qc-report-file",
                    options=quality_reports,
                    value=quality_reports[0]["value"] if quality_reports else None,
                    placeholder="No saved model-quality reports yet",
                    style={"marginTop":"8px"},
                ),
                dcc.Loading(
                    type="circle",
                    children=html.Div(
                        id="som-model-qc",
                        children=render_model_quality_report(quality_reports[0]["value"] if quality_reports else None),
                        style={"marginTop":"8px"},
                    ),
                ),
            ]),

            html.Details(style={"marginTop":"14px"}, children=[
                html.Summary("Advanced: import Somnotate outputs that already exist locally"),
                html.Div("Use this only when scoring/probability files already exist for the recording and only the app import step is needed.", style={"margin":"8px 0", "fontSize":"12px", "opacity":0.8}),
                html.Button("Import existing results", id="btn-som-import-results", n_clicks=0),
            ]),

            html.Div(id="som-action-status", className="status-line", style={"marginTop":"12px"}),
            dcc.Loading(type="circle", children=html.Pre(id="som-log", className="log-box")),
        ])

    if tab == "tab-stats":
        return html.Div(className="card", children=[
            html.H3("Dissociation review dashboard"),
            html.Div("Find where Layer 1, Somnotate, Manual and Final scoring disagree, then jump to those periods from the QC viewer.", className="app-subtitle"),
            html.Div(style={"display":"flex","gap":"8px","alignItems":"center", "flexWrap":"wrap"}, children=[
                html.Label("Recording:"), PDropdown(id="stats-recording", options=rec_options, value=rec_options[0]["value"] if rec_options else None, style={"width":"360px"}),
                html.Label("Event threshold:"), PInput(id="diss-threshold", type="number", value=0.20, step=0.05, style={"width":"100px"}),
                html.Button("Run dissociation analysis", id="btn-run-diss", n_clicks=0),
            ]),
            html.Div(
                className="status-line",
                style={"whiteSpace": "pre-wrap", "marginTop": "8px"},
                children=[
                    html.B("What does the threshold do? "),
                    html.Span(
                        "The pipeline gives each epoch a dissociation score from 0 to 1. "
                        "Epochs with score ≥ threshold are grouped into review events. "
                        "Lower values catch more possible problems; higher values show fewer, stronger disagreements. "
                        "Start around 0.20 for broad review and increase to 0.30 if the event list is too noisy."
                    ),
                ],
            ),
            html.Div(
                className="status-line",
                style={"whiteSpace": "pre-wrap", "marginTop": "4px"},
                children=[
                    html.B("Label note: "),
                    html.Span(
                        "Somnotate = Wake/NREM/REM. Somnotate Wake/Sleep = the same Somnotate output collapsed to Wake vs Sleep, "
                        "so it can be compared with Layer 1. Somnotate Wake/Sleep is not a separate model."
                    ),
                ],
            ),
            html.Div(id="diss-action-status", className="status-line"),
            dcc.Loading(type="circle", children=html.Div(id="diss-log")),
            html.Div(id="diss-pairwise"),
            html.Div(id="diss-state", style={"display":"none"}),
            html.Div(id="diss-events", style={"display":"none"}),
        ])

    return html.Div(className="card", children=[
        html.H3("About"),
        html.P(f"Sleep Stage QC {APP_VERSION}. This app supports semi-automated sleep scoring QC with manual review, Layer 1 Wake/Sleep, Somnotate comparison, dissociation event ranking, and export."),
        html.P(f"Somnotate integration is tested against Somnotate {TESTED_SOMNOTATE_VERSION}, commit {TESTED_SOMNOTATE_COMMIT[:7]}. The app modifies only a temporary copy of the upstream example pipeline."),
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
    Input("btn-som-existing", "n_clicks"), Input("btn-som-evaluate", "n_clicks"),
    Input("btn-som-train", "n_clicks"), Input("btn-som-import-results", "n_clicks"),
    prevent_initial_call=True,
)
def som_button_feedback(n_existing, n_evaluate, n_train, n_import):
    trig = callback_context.triggered_id
    messages = {
        "btn-som-existing": "Running Somnotate scoring with the selected model...",
        "btn-som-evaluate": "Evaluating the selected model against manual scoring...",
        "btn-som-train": "Training a new Somnotate model and running requested QC...",
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
    State("project-root-store", "data"), State("mat-file", "value"), State("import-recording-id", "value"), State("eeg-key", "value"), State("emg-key", "value"), State("ach-key", "value"), State("eeg-fs-key", "value"), State("ach-fs-key", "value"), State("scoring-key", "value"), State("import-epoch-sec", "value"), State("import-mouse-id", "value"), State("import-group", "value"), State("import-condition", "value"), State("import-week", "value"), State("code-map", "value"), State("manifest-refresh", "data"),
    prevent_initial_call=True,
)
def run_import_pipeline(n1,n2,n3,project_root,mat_file,rec_id,eeg_key,emg_key,ach_key,eeg_fs_key,ach_fs_key,scoring_key,epoch_sec,mouse_id,group,condition,week,code_map,refresh):
    if not project_root or not rec_id:
        return "Load project and enter recording ID first.", refresh
    trigger = callback_context.triggered_id
    if trigger == "btn-import-mat":
        if mat_file is None or str(mat_file).strip() == "" or str(mat_file).strip().lower() == "none":
            return (
                "Please paste the full path to a .mat, .edf, or .bdf file before pressing Import recording.\n\n"
                "Examples:\n"
                "Windows: E:\\sleep_data\\mouse01\\recording.mat\n"
                "macOS: /Volumes/T7/sleep_data/mouse01/recording.mat\n"
                "The same field also accepts .edf and .bdf files.",
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
            cmd = [sys.executable, str(PIPELINES_DIR/"01_import_mat_recording.py"), "--mat-file", str(data_path), "--project-root", str(project_root), "--recording-id", str(rec_id), "--eeg-key", str(eeg_key), "--emg-key", str(emg_key), "--epoch-sec", str(epoch_sec), "--code-map", str(code_map or "{}"), "--mouse-id", str(mouse_id or ""), "--group", str(group or ""), "--condition", str(condition or ""), "--week", str(week or "")]
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
    project_path = Path(str(project_root)).expanduser().resolve()
    recording_dir = project_path / "recordings" / str(rec_id)

    # Fail early with a useful message rather than launching a pipeline that cannot succeed.
    if trigger == "btn-compute-features":
        missing = [name for name in ("metadata.json", "eeg.npy", "emg.npy") if not (recording_dir / name).exists()]
        if missing:
            return (
                "Cannot compute features because the imported recording is incomplete.\n"
                f"Recording folder: {recording_dir}\n"
                f"Missing: {', '.join(missing)}",
                refresh,
            )
    if trigger == "btn-run-layer1" and not (recording_dir / "epoch_features.csv").exists():
        return (
            "Cannot run Layer 1 because epoch_features.csv does not exist.\n"
            "Run Compute epoch features first.\n"
            f"Expected file: {recording_dir / 'epoch_features.csv'}",
            refresh,
        )

    code, out = run_command(cmd)
    expected = None
    if trigger == "btn-import-mat":
        expected = recording_dir / "metadata.json"
    elif trigger == "btn-compute-features":
        expected = recording_dir / "epoch_features.csv"
    elif trigger == "btn-run-layer1":
        expected = recording_dir / "layer1_wake_sleep.csv"

    command_text = subprocess.list2cmdline([str(x) for x in cmd])
    if code != 0:
        return (
            f"FAILED (return code {code})\n\n$ {command_text}\n\n{out}",
            refresh,
        )
    if expected is not None and not expected.exists():
        return (
            "FAILED: the command returned successfully, but the expected output was not created.\n"
            f"Expected: {expected}\n\n$ {command_text}\n\n{out}",
            refresh,
        )
    return (
        f"SUCCESS\nCreated/verified: {expected}\n\n$ {command_text}\n\n{out}",
        (refresh or 0) + 1,
    )


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
    # Preserve the current scoring/review selection when the visible QC window
    # refreshes.  Clearing this store here caused the synchronized video to keep
    # playing from its already prepared local clip while the EEG/EMG reviewer
    # lost its interval and reset to "Select an interval to review."
    return make_review_figure(project_root, recording_id, start, wmin), f"Window: {start:.2f}–{end:.2f} min", no_update


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

    try:
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
    except Exception as e:
        return (
            f"Scoring action failed: {type(e).__name__}: {e}",
            no_update,
            no_update,
            no_update,
            no_update,
        )

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
# Synchronized epoch + video review
# -----------------------------------------------------------------------------


def _epoch_position_at_recording_time(epochs: pd.DataFrame, recording_time_s: float) -> int:
    """Map a recording timestamp to the selected epoch index."""
    if epochs is None or len(epochs) == 0:
        return 0
    t = float(recording_time_s)
    t0 = pd.to_numeric(epochs["t0_s"], errors="coerce").to_numpy(float)
    t1 = pd.to_numeric(epochs["t1_s"], errors="coerce").to_numpy(float)
    hit = np.where((t0 <= t) & (t < t1))[0]
    if len(hit):
        return int(hit[0])
    if t < t0[0]:
        return 0
    return int(len(epochs) - 1)


@app.callback(
    Output("epoch-review-index-store", "data"),
    Input("selected-interval-store", "data"),
    Input("epoch-review-prev", "n_clicks"),
    Input("epoch-review-next", "n_clicks"),
    State("epoch-review-index-store", "data"),
    State("project-root-store", "data"),
    State("recording-id-store", "data"),
    prevent_initial_call=True,
)
def navigate_epoch_review(selected, n_prev, n_next, current, project_root, recording_id):
    if not selected or not project_root or not recording_id:
        return None

    rec = load_recording(project_root, recording_id)
    epochs = _epoch_rows_for_selection(rec, selected)
    if len(epochs) == 0:
        return {"position": 0, "count": 0}

    selection_key = f"{float(selected['start_min']):.9f}:{float(selected['end_min']):.9f}:{recording_id}"
    trig = callback_context.triggered_id
    old_key = (current or {}).get("selection_key")
    old_pos = min(max(0, int((current or {}).get("position", 0) or 0)), len(epochs) - 1)

    # Manual review stays manual: Prev/Next always move from the epoch shown in
    # the signal panel. Continuous video playback does not drive this store.
    reference_pos = old_pos

    if trig == "selected-interval-store" or old_key != selection_key:
        pos = 0
    elif trig == "epoch-review-prev":
        pos = max(0, reference_pos - 1)
    elif trig == "epoch-review-next":
        pos = min(len(epochs) - 1, reference_pos + 1)
    else:
        pos = old_pos

    row = epochs.iloc[pos]
    return {
        "selection_key": selection_key,
        "position": int(pos),
        "count": int(len(epochs)),
        "epoch_id": int(row["epoch_id"]) if "epoch_id" in row and pd.notna(row["epoch_id"]) else int(pos),
        "t0_s": float(row["t0_s"]),
        "t1_s": float(row["t1_s"]),
    }


@app.callback(
    Output("epoch-review-panel", "style"),
    Input("selected-interval-store", "data"),
    State("project-root-store", "data"),
    State("recording-id-store", "data"),
)
def show_epoch_review_panel(selected, project_root, recording_id):
    if not project_root or not recording_id or not selected:
        return {"display": "none"}
    return {
        "display": "block",
        "margin": "12px 0 16px 0",
        "padding": "12px",
        "border": "1px solid rgba(127,127,127,0.35)",
        "borderRadius": "12px",
        "background": "rgba(127,127,127,0.045)",
    }


@app.callback(
    Output("epoch-review-clip-store", "data"),
    Output("epoch-review-video-container", "children"),
    Output("epoch-review-cache-status", "children"),
    Input("selected-interval-store", "data"),
    Input("video-file-input", "value"),
    Input("video-offset-input", "value"),
    Input("epoch-review-play-selection", "n_clicks"),
    State("epoch-review-clip-store", "data"),
    State("project-root-store", "data"),
    State("recording-id-store", "data"),
)
def prepare_epoch_review_video_clip(selected, video_file, offset_s, play_clicks, current_clip, project_root, recording_id):
    """Prepare the short local clip used by synchronized review.

    Selection changes normally prepare the clip proactively.  The Play Selection
    button is also an Input on purpose: if a dynamic Dash tab suppressed an
    earlier initial callback, clicking Play always forces this code path and
    cannot silently fall back to the full 10-hour source video.
    """
    trig = callback_context.triggered_id
    if not project_root or not recording_id:
        return None, html.Div("Load a recording first.", className="app-subtitle"), "Load a recording first."
    if not selected:
        return current_clip, no_update, "Select an interval to prepare a local QC clip."

    try:
        # metadata.json is the source of truth if the dynamic input is briefly
        # empty while the QC tab is being rebuilt.
        if not str(video_file or "").strip():
            video_file, saved_offset = load_video_metadata(project_root, recording_id)
            if offset_s in (None, ""):
                offset_s = saved_offset
        if not str(video_file or "").strip():
            return None, epoch_review_cached_video_children(None), "No video linked to this recording yet."

        source = Path(str(video_file)).expanduser()
        if not source.exists() or not source.is_file():
            msg = f"Video file not found: {source}"
            return None, html.Div(msg, className="status-line"), msg

        offset = safe_float(offset_s, 0.0)
        recording_start = float(selected.get("start_min", 0.0)) * 60.0
        recording_end = float(selected.get("end_min", selected.get("start_min", 0.0))) * 60.0
        selected_video_start = max(0.0, recording_start - offset)
        selected_video_end = max(selected_video_start, recording_end - offset)

        # Calling this here also guarantees the local cache directory is created
        # before FFmpeg starts, making failures visible and easy to diagnose.
        cache_root = _video_cache_root()
        print(
            f"[video-qc] trigger={trig} recording={recording_id} "
            f"selection={recording_start:.3f}-{recording_end:.3f}s source={source} "
            f"cache={cache_root}",
            flush=True,
        )

        if _clip_info_covers_selection(current_clip, source, selected_video_start, selected_video_end):
            clip_path = Path(str(current_clip.get("clip_path", "")))
            msg = (
                f"Local QC clip already covers this selection · "
                f"video {float(current_clip['source_start_s'])/60:.2f}–"
                f"{float(current_clip['source_end_s'])/60:.2f} min · {clip_path}"
            )
            # IMPORTANT: do not recreate the <video> element when the existing
            # local clip already covers the requested interval.  Re-mounting it
            # resets currentTime to 0 and can race the Play Selection callback:
            # the old element starts playing, then React replaces it with a new
            # paused element at 0 s.  Keeping the mounted player preserves the
            # seek/play state and makes nearby selections instant.
            return current_clip, no_update, msg

        ok, msg, clip_info = prepare_local_qc_clip(
            source,
            recording_id,
            recording_start,
            recording_end,
            offset,
            context_s=30.0,
        )
        if not ok or not clip_info:
            print(f"[video-qc] clip preparation failed: {msg}", flush=True)
            return None, html.Div(msg, className="status-line", style={"whiteSpace": "pre-wrap"}), msg

        clip_path = Path(str(clip_info.get("clip_path", "")))
        print(
            f"[video-qc] clip ready path={clip_path} exists={clip_path.exists()} "
            f"size={clip_path.stat().st_size if clip_path.exists() else 0}",
            flush=True,
        )
        return clip_info, epoch_review_cached_video_children(clip_info), msg
    except Exception as e:
        msg = f"Local QC clip preparation failed: {type(e).__name__}: {e}"
        print(f"[video-qc] {msg}", flush=True)
        return None, html.Div(msg, className="status-line", style={"whiteSpace": "pre-wrap"}), msg


@app.callback(
    Output("epoch-review-graph", "figure"),
    Output("epoch-review-summary", "children"),
    Output("epoch-review-seek-store", "data"),
    Output("epoch-review-rendered-position-store", "data"),
    Input("epoch-review-index-store", "data"),
    Input("epoch-review-replay", "n_clicks"),
    # Make the selection itself an Input.  In the dynamic QC layout the
    # selection can already exist before epoch-review-index-store is mounted;
    # previously that left the graph stuck on its initial empty figure even
    # though video clip preparation saw the selection correctly.
    Input("selected-interval-store", "data"),
    State("epoch-review-rendered-position-store", "data"),
    State("video-offset-input", "value"),
    State("project-root-store", "data"),
    State("recording-id-store", "data"),
)
def render_epoch_review(index_data, replay_clicks, selected, rendered, offset_s, project_root, recording_id):
    if not selected or not project_root or not recording_id:
        return go.Figure(), "Select an interval to review.", None, None

    rec = load_recording(project_root, recording_id)
    epochs = _epoch_rows_for_selection(rec, selected)
    if len(epochs) == 0:
        return go.Figure(), _epoch_review_empty_message(rec, selected), None, None

    trig = callback_context.triggered_id
    selection_key = f"{float(selected['start_min']):.9f}:{float(selected['end_min']):.9f}:{recording_id}"

    # Do not depend on the navigation store being initialized before the graph.
    # If it is missing/stale, render the first overlapping epoch immediately.
    # The navigation callback can populate/update the store afterwards.
    index_matches_selection = bool(index_data) and index_data.get("selection_key") == selection_key
    if index_matches_selection and int(index_data.get("count", 0) or 0) > 0:
        pos = min(max(0, int(index_data.get("position", 0) or 0)), len(epochs) - 1)
    else:
        pos = 0
        index_data = {
            "selection_key": selection_key,
            "position": 0,
            "count": int(len(epochs)),
            "epoch_id": int(epochs.iloc[0]["epoch_id"]) if "epoch_id" in epochs.columns and pd.notna(epochs.iloc[0]["epoch_id"]) else 0,
            "t0_s": float(epochs.iloc[0]["t0_s"]),
            "t1_s": float(epochs.iloc[0]["t1_s"]),
        }

    # Video playback intentionally does not drive the signal graph.  The graph
    # is redrawn only for a new selection, Prev/Next, or Replay epoch.

    row = epochs.iloc[pos]
    fig = make_epoch_review_figure(rec, row, selected)
    summary = epoch_review_summary(rec, row, pos, len(epochs))

    offset = safe_float(offset_s, 0.0)
    t0_s = float(row["t0_s"])
    t1_s = float(row["t1_s"])
    seek: dict[str, Any] = {
        "time_s": max(0.0, t0_s - offset),
        "end_time_s": max(0.0, t1_s - offset),
        "recording_time_s": t0_s,
        "recording_end_s": t1_s,
        "offset_s": offset,
        "auto_play": trig == "epoch-review-replay",
        "epoch_position": pos,
        "epoch_count": len(epochs),
    }

    return fig, summary, seek, {
        "selection_key": selection_key,
        "position": int(pos),
        "t0_s": float(row["t0_s"]),
        "t1_s": float(row["t1_s"]),
    }


@app.callback(
    Output("epoch-review-playback-command-store", "data"),
    Input("epoch-review-play-selection", "n_clicks"),
    Input("epoch-review-pause", "n_clicks"),
    State("selected-interval-store", "data"),
    State("video-offset-input", "value"),
    prevent_initial_call=True,
)
def request_epoch_review_playback(n_play, n_pause, selected, offset_s):
    trig = callback_context.triggered_id
    if trig == "epoch-review-pause":
        return {"action": "pause"}
    if trig != "epoch-review-play-selection" or not selected:
        return no_update

    offset = safe_float(offset_s, 0.0)
    recording_start_s = float(selected.get("start_min", 0.0)) * 60.0
    recording_end_s = float(selected.get("end_min", selected.get("start_min", 0.0))) * 60.0
    return {
        "action": "play",
        "time_s": max(0.0, recording_start_s - offset),
        "end_time_s": max(0.0, recording_end_s - offset),
        "offset_s": offset,
    }


# Manual epoch navigation / replay. Seeking is asynchronous in HTML5 video, so
# wait for metadata and the seeked event before calling play(). This avoids the
# common black/stuck player caused by play() racing currentTime assignment.
app.clientside_callback(
    r"""
    function(data) {
        if (!data) return "";

        const requestedStart = Math.max(0, Number(data.time_s || 0));
        const requestedEnd = Math.max(requestedStart, Number(data.end_time_s || requestedStart));
        const autoPlay = Boolean(data.auto_play);
        const deadline = Date.now() + 30000;

        function feedback(text) {
            const el = document.getElementById("epoch-review-video-feedback");
            if (el) el.textContent = text;
        }
        function getLocalVideo() {
            const video = document.getElementById("epoch-review-video-player");
            if (!video || video.dataset.qcLocalClip !== "1") return null;
            const sourceStart = Number(video.dataset.sourceStartS);
            const sourceEnd = Number(video.dataset.sourceEndS);
            if (!Number.isFinite(sourceStart) || !Number.isFinite(sourceEnd)) return null;
            if (requestedStart < sourceStart - 0.10 || requestedStart > sourceEnd + 0.10) return null;
            return video;
        }
        function waitForLocalVideo() {
            const video = getLocalVideo();
            if (!video) {
                if (Date.now() < deadline) {
                    feedback("Preparing local QC clip…");
                    window.setTimeout(waitForLocalVideo, 100);
                } else {
                    feedback("Timed out waiting for the local QC clip. Check the cache status and terminal output.");
                }
                return;
            }

            const sourceStart = Number(video.dataset.sourceStartS || 0);
            const start = Math.max(0, requestedStart - sourceStart);
            const end = Math.max(start, requestedEnd - sourceStart);
            video.muted = true;

            function clearStop() {
                if (video._epochReviewStopHandler) {
                    video.removeEventListener("timeupdate", video._epochReviewStopHandler);
                    video._epochReviewStopHandler = null;
                }
            }
            function installStop() {
                clearStop();
                const stopHandler = function() {
                    if (video.currentTime >= end - 0.025) {
                        video.pause();
                        clearStop();
                    }
                };
                video._epochReviewStopHandler = stopHandler;
                video.addEventListener("timeupdate", stopHandler);
            }
            function doSeek() {
                video.pause();
                clearStop();
                const afterSeek = function() {
                    feedback("Video aligned to current epoch.");
                    if (autoPlay && end > start) {
                        installStop();
                        const promise = video.play();
                        if (promise && promise.catch) {
                            promise.catch(function(err) {
                                feedback("Video play was blocked/failed: " + err + ".");
                            });
                        }
                    }
                };
                if (Math.abs(Number(video.currentTime || 0) - start) < 0.02) {
                    afterSeek();
                    return;
                }
                video.addEventListener("seeked", afterSeek, {once: true});
                try { video.currentTime = start; }
                catch (e) { feedback("Could not seek local QC clip: " + e); }
            }
            if (video.readyState < 1) video.addEventListener("loadedmetadata", doSeek, {once: true});
            else doSeek();
        }

        waitForLocalVideo();
        return "Preparing/aligning local QC clip…";
    }
    """,
    Output("epoch-review-video-feedback", "children"),
    Input("epoch-review-seek-store", "data"),
)


# Continuous playback of the whole selected interval. EEG/EMG following is
# driven by the lightweight clock callback below rather than by re-seeking the
# video every epoch.
app.clientside_callback(
    r"""
    function(command) {
        if (!command) return window.dash_clientside.no_update;

        function feedback(text) {
            const el = document.getElementById("epoch-review-video-feedback");
            if (el) el.textContent = text;
        }

        if (command.action === "pause") {
            const current = document.getElementById("epoch-review-video-player");
            if (current && current.dataset.qcLocalClip === "1") current.pause();
            return "Paused synchronized playback.";
        }
        if (command.action !== "play") return window.dash_clientside.no_update;

        const requestedStart = Math.max(0, Number(command.time_s || 0));
        const requestedEnd = Math.max(requestedStart, Number(command.end_time_s || requestedStart));
        const deadline = Date.now() + 30000;

        function getLocalVideo() {
            const video = document.getElementById("epoch-review-video-player");
            if (!video || video.dataset.qcLocalClip !== "1") return null;
            const sourceStart = Number(video.dataset.sourceStartS);
            const sourceEnd = Number(video.dataset.sourceEndS);
            if (!Number.isFinite(sourceStart) || !Number.isFinite(sourceEnd)) return null;
            if (requestedStart < sourceStart - 0.10 || requestedEnd > sourceEnd + 0.10) return null;
            return video;
        }

        function waitAndPlay() {
            const video = getLocalVideo();
            if (!video) {
                if (Date.now() < deadline) {
                    feedback("Preparing local QC clip…");
                    window.setTimeout(waitAndPlay, 100);
                } else {
                    feedback("Timed out waiting for a local QC clip covering this selection. Check the cache status and terminal output.");
                }
                return;
            }

            video.muted = true;
            if (video._epochReviewStopHandler) {
                video.removeEventListener("timeupdate", video._epochReviewStopHandler);
                video._epochReviewStopHandler = null;
            }

            const sourceStart = Number(video.dataset.sourceStartS || 0);
            const start = Math.max(0, requestedStart - sourceStart);
            const end = Math.max(start, requestedEnd - sourceStart);
            const stopHandler = function() {
                if (video.currentTime >= end - 0.025) {
                    video.pause();
                    video.removeEventListener("timeupdate", stopHandler);
                    video._epochReviewStopHandler = null;
                    feedback("Selected interval finished.");
                }
            };
            video._epochReviewStopHandler = stopHandler;
            video.addEventListener("timeupdate", stopHandler);

            function begin() {
                feedback("Starting synchronized playback…");
                const promise = video.play();
                if (promise && promise.then) {
                    promise.then(function() {
                        feedback("Playing selected interval.");
                    }).catch(function(err) {
                        feedback("Video could not play: " + err + ".");
                    });
                }
            }
            function afterSeek() {
                if (video.readyState >= 3) begin();
                else video.addEventListener("canplay", begin, {once: true});
            }
            function seekThenPlay() {
                if (Math.abs(Number(video.currentTime || 0) - start) < 0.02) afterSeek();
                else {
                    video.addEventListener("seeked", afterSeek, {once: true});
                    try { video.currentTime = start; }
                    catch (e) { feedback("Could not seek local QC clip: " + e); }
                }
            }
            if (video.readyState < 1) video.addEventListener("loadedmetadata", seekThenPlay, {once: true});
            else seekThenPlay();
        }

        waitAndPlay();
        return "Preparing local QC clip for synchronized playback…";
    }
    """,
    Output("epoch-review-video-feedback", "children", allow_duplicate=True),
    Input("epoch-review-playback-command-store", "data"),
    prevent_initial_call=True,
)


# Poll only lightweight DOM state. The server callback above updates the signal
# plot only when currentTime crosses into a different scoring epoch.
app.clientside_callback(
    r"""
    function(n, offsetValue, renderedEpoch) {
        const video = document.getElementById("epoch-review-video-player");
        if (!video || video.dataset.qcLocalClip !== "1") {
            return [window.dash_clientside.no_update, "Waiting for short local QC clip…"];
        }
        const offset = Number(offsetValue || 0);
        const sourceStart = Number(video.dataset.sourceStartS || 0);
        const err = video.error;
        const clipTime = Number(video.currentTime || 0);
        const sourceVideoTime = clipTime + sourceStart;
        const recordingTime = sourceVideoTime + offset;
        if (err) {
            const meanings = {1:"aborted", 2:"network error", 3:"decode error", 4:"source/codec not supported"};
            const msg = "Video error " + err.code + " (" + (meanings[err.code] || "unknown") + ").";
            return [{video_time_s:sourceVideoTime, clip_time_s:clipTime, recording_time_s:recordingTime, paused:true, error_code:err.code}, msg];
        }

        let msg = "";
        if (video.readyState < 1) msg = "Loading local QC clip metadata…";
        else if (video.readyState < 3 && !video.paused) msg = "Buffering local QC clip…";
        else if (video.paused) msg = "Video paused · recording " + recordingTime.toFixed(3) + " s · local clip " + clipTime.toFixed(3) + " s.";
        else msg = "Video playing · recording " + recordingTime.toFixed(3) + " s · local clip " + clipTime.toFixed(3) + " s.";

        // The moving playhead is updated by a separate fully client-side callback.
        // Only notify Python when playback leaves the epoch currently rendered in
        // the EEG/EMG panel.  Previously this store changed every 200 ms, causing
        // Dash to reread several large CSV files five times per second while the
        // same Flask server was also serving video byte ranges. That can make a
        // local short clip feel as if it is buffering or not playing at all.
        const t0 = renderedEpoch ? Number(renderedEpoch.t0_s) : NaN;
        const t1 = renderedEpoch ? Number(renderedEpoch.t1_s) : NaN;
        const insideRenderedEpoch = Number.isFinite(t0) && Number.isFinite(t1) &&
                                    recordingTime >= t0 && recordingTime < t1;

        if (insideRenderedEpoch) {
            video._epochReviewLastServerRequestMs = 0;
            return [window.dash_clientside.no_update, msg];
        }

        // If Python is still rendering the next epoch, do not queue duplicate
        // requests on every clock tick. One request roughly every 750 ms is enough
        // to keep 1-s scoring epochs visually in step while playback itself stays
        // smooth and entirely browser-driven.
        const now = Date.now();
        const last = Number(video._epochReviewLastServerRequestMs || 0);
        if (now - last < 750) {
            return [window.dash_clientside.no_update, msg];
        }
        video._epochReviewLastServerRequestMs = now;

        const data = {
            video_time_s: sourceVideoTime,
            clip_time_s: clipTime,
            recording_time_s: recordingTime,
            paused: Boolean(video.paused),
            ended: Boolean(video.ended),
            ready_state: Number(video.readyState || 0),
            network_state: Number(video.networkState || 0)
        };
        return [data, msg];
    }
    """,
    Output("epoch-review-playback-time-store", "data"),
    Output("epoch-review-video-diagnostics", "children"),
    Input("epoch-review-clock", "n_intervals"),
    State("video-offset-input", "value"),
    State("epoch-review-rendered-position-store", "data"),
)


# Pure browser-side playhead.  This does exactly one thing: read video.currentTime
# and move the named red Plotly line.  It never writes a Dash Store used by Python,
# never redraws the EEG/EMG traces, and never changes video playback.
app.clientside_callback(
    r"""
    function(n, selected, offsetValue) {
        if (!selected) return "";

        const start = Number(selected.start_min) * 60.0;
        const end = Number(selected.end_min) * 60.0;
        if (!Number.isFinite(start) || !Number.isFinite(end)) return "";

        const graphWrap = document.getElementById("epoch-review-graph");
        const gd = graphWrap ? graphWrap.querySelector(".js-plotly-plot") : null;
        if (!gd || !window.Plotly || !gd.layout) return "";

        let recordingTime = start;
        const video = document.getElementById("epoch-review-video-player");
        if (video && video.dataset.qcLocalClip === "1" && video.readyState >= 1) {
            const sourceStart = Number(video.dataset.sourceStartS || 0);
            const offset = Number(offsetValue || 0);
            const clipTime = Number(video.currentTime || 0);
            const candidate = sourceStart + clipTime + offset;
            if (Number.isFinite(candidate)) recordingTime = candidate;
        }

        // Before Play, the cached clip may be sitting at clip time 0, which is
        // intentionally earlier than the selected interval. Keep the tracer at
        // the selection start until the player has actually sought there.
        recordingTime = Math.max(start, Math.min(end, recordingTime));

        const shapes = (gd.layout.shapes || []);
        let idx = -1;
        for (let i = 0; i < shapes.length; i++) {
            if (shapes[i] && shapes[i].name === "video_playhead") {
                idx = i;
                break;
            }
        }
        if (idx < 0) return "";

        const update = {};
        update["shapes[" + idx + "].x0"] = recordingTime;
        update["shapes[" + idx + "].x1"] = recordingTime;
        window.Plotly.relayout(gd, update);

        return recordingTime.toFixed(3);
    }
    """,
    Output("epoch-review-playhead-dummy", "children"),
    Input("epoch-review-playhead-clock", "n_intervals"),
    State("selected-interval-store", "data"),
    State("video-offset-input", "value"),
)

# Lightweight text status only; it does not touch the graph.
app.clientside_callback(
    r"""
    function(n, offsetValue) {
        const video = document.getElementById("epoch-review-video-player");
        if (!video || video.dataset.qcLocalClip !== "1") {
            return "Video: waiting for short local QC clip…";
        }
        const offset = Number(offsetValue || 0);
        const clipTime = Number(video.currentTime || 0);
        const sourceStart = Number(video.dataset.sourceStartS || 0);
        const recordingTime = clipTime + sourceStart + offset;
        const state = video.paused ? "paused" : "playing";
        return "Video: recording " + recordingTime.toFixed(3) +
               " s · local clip " + clipTime.toFixed(3) + " s · " + state;
    }
    """,
    Output("epoch-review-live-position", "children"),
    Input("epoch-review-playhead-clock", "n_intervals"),
    State("video-offset-input", "value"),
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
    Input("clear-video-review-cache", "n_clicks"),
    State("project-root-store", "data"),
    State("video-file-input", "value"),
    State("video-offset-input", "value"),
    prevent_initial_call=True,
)
def update_video_panel(recording_id, save_clicks, convert_clicks, clear_cache_clicks, project_root, video_file_value, video_offset_value):
    if not project_root or not recording_id:
        return "", 0.0, html.Div("Load a recording to enable video QC.", className="app-subtitle"), "Load a recording first."

    trig = callback_context.triggered_id

    if trig == "save-video-settings":
        video_file = str(video_file_value or "").strip()
        offset_s = safe_float(video_offset_value, 0.0)
        ok, msg = save_video_metadata(project_root, recording_id, video_file, offset_s)

        # Always read back from metadata after saving.  This makes the UI reflect
        # exactly what was persisted for this recording and catches stale browser
        # persistence/race conditions immediately.
        saved_video, saved_offset = load_video_metadata(project_root, recording_id)
        if ok:
            if not saved_video:
                msg = (msg + "\nERROR: metadata.json did not contain a video path after save.").strip()
            elif saved_video != video_file:
                msg = (msg + f"\nSaved metadata resolves to:\n{saved_video}").strip()
            return saved_video, saved_offset, video_panel_children(saved_video, saved_offset), msg

        # If saving failed because the input was transiently empty, preserve and
        # reload any previously stored recording-specific video rather than blanking
        # the player.
        return saved_video, saved_offset, video_panel_children(saved_video, saved_offset), msg

    if trig == "clear-video-review-cache":
        ok, msg = clear_video_review_cache(recording_id)
        video_file, offset_s = load_video_metadata(project_root, recording_id)
        return video_file, offset_s, video_panel_children(video_file, offset_s), msg

    if trig == "convert-video-mp4":
        original_video_file = str(video_file_value or "").strip()
        offset_s = safe_float(video_offset_value, 0.0)
        ok, msg, converted_path = convert_video_to_browser_mp4(original_video_file)
        if not ok or not converted_path:
            return original_video_file, offset_s, video_panel_children(original_video_file, offset_s), msg

        ok_save, save_msg = save_video_metadata(project_root, recording_id, converted_path, offset_s)
        source_ok, source_msg = save_video_source_metadata(
            project_root,
            recording_id,
            original_video_file,
            converted_path,
        )
        messages = [msg]
        messages.append(save_msg if ok_save else f"Could not save converted path: {save_msg}")
        messages.append(source_msg if source_ok else f"Could not save source path: {source_msg}")
        return converted_path, offset_s, video_panel_children(converted_path, offset_s), "\n".join(messages)

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


def read_somnotate_model_epoch_metadata(model_file: str | Path | None) -> tuple[float | None, str]:
    if not model_file:
        return None, ""
    p = Path(str(model_file)).expanduser()
    candidates = [p.with_suffix(".metadata.json"), p.with_name(p.name + ".metadata.json")]
    for meta_path in candidates:
        if meta_path.exists():
            try:
                meta = read_json(meta_path)
                val = meta.get("somnotate_epoch_sec", meta.get("epoch_sec", meta.get("time_resolution")))
                if val is not None:
                    return float(val), str(meta_path)
                return None, str(meta_path)
            except Exception:
                return None, str(meta_path)
    return None, ""


@app.callback(
    Output("som-existing-epoch-summary", "children"),
    Output("som-train-epoch-summary", "children"),
    Input("som-epoch-sec", "value"),
)
def update_somnotate_epoch_summaries(som_epoch_sec):
    selected_epoch = safe_float(som_epoch_sec, 1.0)
    existing_msg = (
        f"Existing-model scoring will run Somnotate with {selected_epoch:g} s epochs. "
        f"Select only a model trained with {selected_epoch:g} s epochs, or use a legacy model only if you know it matches."
    )
    train_msg = (
        f"New model training will create a {selected_epoch:g} s Somnotate model. "
        "This epoch length is saved next to the trained model in a .metadata.json file. "
        "Change the dropdown above before pressing Train new model."
    )
    return existing_msg, train_msg


@app.callback(
    Output("som-epoch-warning", "children"),
    Input("som-model-file", "value"),
    Input("som-model-file-custom", "value"),
    Input("som-epoch-sec", "value"),
)
def update_somnotate_epoch_warning(model_file, custom_model_file, som_epoch_sec):
    model_file = str(custom_model_file or "").strip() or model_file
    selected_epoch = safe_float(som_epoch_sec, 1.0)
    base = (
        "Somnotate epoch warning: models are epoch-length specific. "
        "Use models with the same epoch length used for preprocessing/training. For example: 1 s models with 1 s epochs, 2 s models with 2 s epochs, and legacy 5 s Somnotate models with 5 s epochs. "
    )
    if not model_file:
        return base + f"You selected {selected_epoch:g} s epochs. Select a model, or train a new matching model."

    model_epoch, meta_path = read_somnotate_model_epoch_metadata(model_file)
    meta = _model_metadata(Path(str(model_file)).expanduser()) if model_file else {}
    legacy_note = ""
    if meta.get("legacy_model") or meta.get("legacy_unverified"):
        serialized_sklearn = meta.get("serialized_scikit_learn_version", meta.get("scikit_learn_version"))
        legacy_note = " This is a LEGACY model."
        if serialized_sklearn:
            legacy_note += f" It was serialized with scikit-learn {serialized_sklearn}; use a version-matched/retrained release model for final scientific analysis."

    if model_epoch is None:
        return (
            base
            + f"You selected {selected_epoch:g} s epochs. This model has no readable epoch metadata, "
            + "so only use it if you know it was trained with the same epoch length."
            + legacy_note
        )

    if abs(model_epoch - selected_epoch) > 1e-6:
        return (
            "⚠️ Somnotate epoch mismatch. "
            f"Selected model metadata says {model_epoch:g} s epochs, "
            f"but the app is set to {selected_epoch:g} s epochs. "
            "Choose a matching model or change Somnotate epoch sec before running. "
            "Known mismatches are blocked by the pipeline. "
            f"Metadata: {meta_path}"
        )

    return (
        f"Somnotate epoch OK: selected epoch = {selected_epoch:g} s and model metadata = {model_epoch:g} s. "
        "New models trained from this tab will also save epoch and runtime metadata."
        + legacy_note
    )


def _split_recording_ids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _recording_ids_cli(value) -> str:
    return ",".join(_split_recording_ids(value))


def validate_somnotate_ui_inputs(*, som_root, model_file=None, recording_ids=None, training=False) -> str | None:
    root_text = str(som_root or "").strip()
    if not root_text:
        return (
            "Somnotate repository path is empty. Clone the tested Somnotate checkout and set this field, "
            f"for example {Path.home() / 'somnotate'}."
        )
    root = Path(root_text).expanduser()
    if not root.exists():
        return f"Somnotate repository path does not exist: {root}"
    expected = root / "example_pipeline" / "01_preprocess_signals.py"
    if not expected.exists():
        return f"Somnotate example pipeline was not found under: {root}"

    ids = _split_recording_ids(recording_ids)
    if not ids:
        return "Enter at least one recording ID." if not training else "Enter at least one training recording ID."

    if not training:
        model_text = str(model_file or "").strip()
        if not model_text:
            return "Select a Somnotate model before running the existing-model workflow."
        if not Path(model_text).expanduser().exists():
            return f"Somnotate model file does not exist: {model_text}"
    return None


@app.callback(
    Output("som-log", "children"),
    Output("som-qc-report-file", "options"),
    Output("som-qc-report-file", "value"),
    Input("btn-som-existing", "n_clicks"),
    Input("btn-som-evaluate", "n_clicks"),
    Input("btn-som-train", "n_clicks"),
    Input("btn-som-import-results", "n_clicks"),
    State("project-root-store", "data"),
    State("som-recording-ids", "value"),
    State("som-target-fs", "value"),
    State("som-epoch-sec", "value"),
    State("som-root", "value"),
    State("som-conda-env", "value"),
    State("som-python", "value"),
    State("som-model-file", "value"),
    State("som-model-file-custom", "value"),
    State("som-existing-steps", "value"),
    State("som-eval-ids", "value"),
    State("som-eval-context", "value"),
    State("som-eval-steps", "value"),
    State("som-train-ids", "value"),
    State("som-test-ids", "value"),
    State("som-model-name", "value"),
    State("som-train-steps", "value"),
    prevent_initial_call=True,
)
def run_somnotate(
    n_exist, n_evaluate, n_train, n_import,
    project_root, rec_ids, target_fs, som_epoch_sec, som_root, som_env, som_py,
    model_file, custom_model_file, steps,
    eval_ids, eval_context, eval_steps,
    train_ids, test_ids, model_name, train_steps,
):
    if not project_root:
        return "Load project first.", no_update, no_update

    trig = callback_context.triggered_id
    base = [sys.executable, str(PIPELINES_DIR / "10_somnotate_layer.py")]
    epoch_arg = str(som_epoch_sec or "1.0")
    effective_model_file = str(custom_model_file or "").strip() or model_file

    if trig == "btn-som-existing":
        problem = validate_somnotate_ui_inputs(
            som_root=som_root,
            model_file=effective_model_file,
            recording_ids=rec_ids,
            training=False,
        )
        if problem:
            return "Somnotate setup problem: " + problem, no_update, no_update
        cmd = base + [
            "use-existing-model",
            "--project-root", str(project_root),
            "--recording-ids", _recording_ids_cli(rec_ids),
            "--somnotate-root", str(som_root or ""),
            "--somnotate-conda-env", str(som_env or "somnotate_env"),
            "--model-file", str(effective_model_file or ""),
            "--target-fs", str(target_fs or 512),
            "--epoch-sec", epoch_arg,
        ]
        if som_py:
            cmd += ["--somnotate-python", str(som_py)]
        for step in steps or []:
            cmd += [f"--{step}"]

    elif trig == "btn-som-evaluate":
        problem = validate_somnotate_ui_inputs(
            som_root=som_root,
            model_file=effective_model_file,
            recording_ids=eval_ids,
            training=False,
        )
        if problem:
            return "Somnotate evaluation setup problem: " + problem, no_update, no_update
        cmd = base + [
            "evaluate-model",
            "--project-root", str(project_root),
            "--recording-ids", _recording_ids_cli(eval_ids),
            "--somnotate-root", str(som_root or ""),
            "--somnotate-conda-env", str(som_env or "somnotate_env"),
            "--model-file", str(effective_model_file or ""),
            "--target-fs", str(target_fs or 512),
            "--epoch-sec", epoch_arg,
            "--evaluation-context", str(eval_context or "training-or-unknown"),
        ]
        if som_py:
            cmd += ["--somnotate-python", str(som_py)]
        for step in eval_steps or []:
            cmd += [f"--{step}"]

    elif trig == "btn-som-train":
        problem = validate_somnotate_ui_inputs(
            som_root=som_root,
            recording_ids=train_ids,
            training=True,
        )
        if problem:
            return "Somnotate setup problem: " + problem, no_update, no_update
        cmd = base + [
            "train-model",
            "--project-root", str(project_root),
            "--train-recording-ids", _recording_ids_cli(train_ids),
            "--test-recording-ids", _recording_ids_cli(test_ids),
            "--somnotate-root", str(som_root or ""),
            "--somnotate-conda-env", str(som_env or "somnotate_env"),
            "--model-name", str(model_name or "model"),
            "--target-fs", str(target_fs or 512),
            "--epoch-sec", epoch_arg,
        ]
        if som_py:
            cmd += ["--somnotate-python", str(som_py)]
        for step in train_steps or []:
            cmd += [f"--{step}"]

    elif trig == "btn-som-import-results":
        cmd = base + [
            "import-results",
            "--project-root", str(project_root),
            "--recording-ids", _recording_ids_cli(rec_ids),
            "--epoch-sec", epoch_arg,
        ]
    else:
        return no_update, no_update, no_update

    code, out = run_command(cmd)
    log_text = f"$ {' '.join(cmd)}\n\n{out}"
    if trig in {"btn-som-evaluate", "btn-som-train"} and code == 0:
        reports = available_quality_reports(project_root)
        selected = reports[0]["value"] if reports else no_update
        return log_text, reports, selected
    return log_text, no_update, no_update


@app.callback(
    Output("som-model-qc", "children"),
    Input("som-qc-report-file", "value"),
)
def update_somnotate_model_quality(report_file):
    return render_model_quality_report(report_file)



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


def rgba_from_hex(hex_color, alpha=SCORING_BACKGROUND_ALPHA):
    """Convert #RRGGBB to rgba string."""
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return f"rgba(150,150,150,{alpha})"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def state_to_soft_fill(state, alpha=SCORING_BACKGROUND_ALPHA):
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
        fill = state_to_soft_fill(state, alpha=SCORING_BACKGROUND_ALPHA)

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



def normalize_state_label(x, collapse_sleep=False):
    """Return a canonical app label for sleep-state values from any source."""
    if x is None:
        return "Undefined"
    try:
        if pd.isna(x):
            return "Undefined"
    except Exception:
        pass

    s = str(x).strip()
    if not s:
        return "Undefined"

    low = re.sub(r"[\s_-]+", " ", s.lower()).strip()
    compact = re.sub(r"[^a-z0-9]+", "", low)

    if compact in {"nan", "none", "null", "undefined", "uncertain", "unknown", "nd", "tr"}:
        out = "Undefined"
    elif compact in {"awake", "wake", "wk", "w"} or "wake" in low:
        out = "Wake"
    elif compact in {"nrem", "nonrem", "nr", "sws", "slowwavesleep"} or "non rem" in low:
        out = "NREM"
    elif compact in {"rem", "ps"}:
        out = "REM"
    elif "artifact" in low or compact == "artf":
        out = "Artifact"
    elif "sleep" in low:
        out = "Sleep"
    else:
        out = s

    if collapse_sleep and out in {"NREM", "REM", "Sleep"}:
        return "Sleep"
    return out


def is_unreviewed_state(x):
    return normalize_state_label(x) in {"Undefined", "Uncertain", ""}


def simplify_source_key(k):
    return re.sub(r"[^a-z0-9]+", "", str(k).lower())


SOURCE_KEY_ALIASES = {
    "Layer 1": {
        "layer1", "layer1state", "layer1label", "layer1ws", "layer1wakesleep", "l1", "layerone"
    },
    "Somnotate": {
        "somnotate", "somnotatestate", "somnotatefull", "somnotatewnr", "som", "somstate"
    },
    "Somnotate Wake/Sleep": {
        "somnotatews", "somnotatewakesleep", "somnotatebinary", "somws", "somnotatecollapsed"
    },
    "Final": {
        "final", "finalstate", "appfinal", "reviewed", "reviewedstate"
    },
    "Manual": {
        "manual", "manualstate", "manualscore", "manualscoring"
    },
}


def parse_states_at_peak(value):
    """Parse the event states_at_peak field into a source->state dictionary.

    The exact string format has changed across app versions, so this accepts
    dictionaries, JSON-like strings, and key=value / key: value summaries.
    """
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except Exception:
        pass
    if isinstance(value, dict):
        raw = value
    else:
        text = str(value).strip()
        if not text:
            return {}
        raw = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, dict):
                    raw = parsed
                    break
            except Exception:
                pass
        if raw is None:
            raw = {}
            # Accept formats such as "Layer 1=Wake | Somnotate=REM" or
            # "layer1: Wake, somnotate: REM".
            parts = re.split(r"\s*[|;\n]+\s*", text)
            if len(parts) == 1:
                parts = re.split(r"\s*,\s*(?=[A-Za-z0-9 _/-]+\s*[:=])", text)
            for part in parts:
                if not part:
                    continue
                if "=" in part:
                    k, v = part.split("=", 1)
                elif ":" in part:
                    k, v = part.split(":", 1)
                else:
                    continue
                k = k.strip().strip("'\"")
                v = v.strip().strip("'\"{}[]")
                if k:
                    raw[k] = v
    out = {}
    for k, v in raw.items():
        sk = simplify_source_key(k)
        matched = None
        for canonical, aliases in SOURCE_KEY_ALIASES.items():
            if sk in aliases:
                matched = canonical
                break
        if matched is None:
            # Keep unknown keys, but make them readable.
            matched = str(k).replace("_", " ").strip()
        out[matched] = normalize_state_label(v)
    return out


def event_source_series(events, source_name):
    """Return event-level state values for a source, using columns or states_at_peak."""
    if events is None or len(events) == 0:
        return pd.Series(dtype=object)

    candidates = {
        "Layer 1": ["layer1_state", "layer1_label", "layer_1_state", "Layer 1", "layer1"],
        "Somnotate": ["somnotate_state", "som_state", "Somnotate", "somnotate"],
        "Somnotate Wake/Sleep": ["somnotate_ws", "somnotate_wake_sleep", "somnotate_ws_state", "Somnotate WS"],
        "Final": ["final_state", "app_final_state", "Final", "final"],
        "Manual": ["manual_state", "Manual", "manual"],
    }.get(source_name, [])

    for col in candidates:
        if col in events.columns:
            return events[col].map(normalize_state_label)

    if "states_at_peak" in events.columns:
        vals = []
        for x in events["states_at_peak"]:
            d = parse_states_at_peak(x)
            vals.append(normalize_state_label(d.get(source_name, "Undefined")))
        return pd.Series(vals, index=events.index, dtype=object)

    return pd.Series(["Undefined"] * len(events), index=events.index, dtype=object)


def biological_interpretation(som_state, layer_state, final_state=None):
    som = normalize_state_label(som_state)
    layer = normalize_state_label(layer_state, collapse_sleep=True)
    final = normalize_state_label(final_state) if final_state is not None else "Undefined"
    if som == "REM" and layer == "Wake":
        return "REM/Wake conflict: check EMG, movement, and whether REM was overcalled or Layer 1 was too wake-biased."
    if som == "NREM" and layer == "Wake":
        return "NREM/Wake conflict: often quiet wake vs NREM ambiguity."
    if som == "Wake" and layer == "Sleep":
        return "Wake/Sleep conflict: possible low-EMG wake, drowsiness, or Layer 1 sleep overcall."
    if som == "REM" and final in {"Wake", "NREM"}:
        return "Reviewer corrected Somnotate REM; inspect as possible false REM or transition."
    if som == "REM" and final == "REM":
        return "Reviewer kept Somnotate REM at this peak."
    return "General scoring disagreement or low-confidence period."


def render_flagged_state_patterns(events):
    """Summarize what biological state combinations dominate flagged events."""
    if events is None or len(events) == 0:
        return html.Div(className="card", children=[html.H4("Somnotate vs Layer 1 biological patterns"), html.Div("No flagged events to summarize.", className="app-subtitle")])

    df = events.copy()
    df["somnotate_peak"] = event_source_series(df, "Somnotate")
    df["layer1_peak"] = event_source_series(df, "Layer 1").map(lambda x: normalize_state_label(x, collapse_sleep=True))
    df["final_peak"] = event_source_series(df, "Final")

    # Keep only biologically interpretable Layer 1 comparisons.
    # Layer 1 = Undefined usually means that the state could not be parsed or is missing,
    # not a meaningful Wake/Sleep conflict. Excluding it keeps this summary focused on
    # interpretable patterns such as Somnotate REM / Layer 1 Wake.
    before_filter_n = len(df)
    usable = df[df["layer1_peak"] != "Undefined"].copy()
    excluded_layer1_undefined_n = before_filter_n - len(usable)

    # Also drop rows where Somnotate is undefined, because the biological pattern
    # needs a named Somnotate state to be interpretable.
    usable = usable[usable["somnotate_peak"] != "Undefined"].copy()

    if len(usable) == 0:
        # Do not show a large warning card when Layer 1 is mostly Undefined.
        # In that case the biologically useful comparison is Somnotate vs Final,
        # which is rendered separately below/above this section.
        return html.Div(style={"display": "none"})

    usable["pattern"] = "Somnotate " + usable["somnotate_peak"].astype(str) + " / Layer 1 " + usable["layer1_peak"].astype(str)
    counts = usable.groupby(["somnotate_peak", "layer1_peak", "pattern"], dropna=False).size().reset_index(name="events")
    counts = counts.sort_values("events", ascending=False).head(12)
    total = len(usable)
    counts["% interpretable events"] = (100.0 * counts["events"] / max(total, 1)).round(1)
    counts["interpretation"] = [biological_interpretation(s, l) for s, l in zip(counts["somnotate_peak"], counts["layer1_peak"])]
    table = counts[["pattern", "events", "% interpretable events", "interpretation"]].copy()

    most = table.iloc[0]
    summary = (
        f"Most common interpretable flagged pattern: {most['pattern']} "
        f"({int(most['events'])} events, {most['% interpretable events']:.1f}% of events with defined Layer 1 and Somnotate states)."
    )
    if excluded_layer1_undefined_n:
        summary += f" Excluded {excluded_layer1_undefined_n} events where Layer 1 was Undefined."

    return html.Div(className="card", children=[
        html.H4("Somnotate vs Layer 1 biological patterns"),
        html.Div(summary, className="app-subtitle", style={"marginBottom": "8px"}),
        dash_table.DataTable(
            data=table.to_dict("records"),
            columns=[{"name": c, "id": c} for c in table.columns],
            page_size=8,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": 12, "padding": "8px", "textAlign": "left", "whiteSpace": "normal", "height": "auto", "maxWidth": "440px"},
            style_header={"fontWeight": "bold", "background": "#F3F4F6", "border": "1px solid #E5E7EB"},
            style_data={"border": "1px solid #E5E7EB"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#FAFAFA"}],
        ),
    ])


def somnotate_final_interpretation(som_state, final_state):
    """Short biological/review interpretation for Somnotate vs Final differences."""
    som = normalize_state_label(som_state)
    final = normalize_state_label(final_state)

    if som == "REM" and final == "Wake":
        return "Somnotate REM was corrected to Wake: inspect EMG/movement and possible false REM."
    if som == "REM" and final == "NREM":
        return "Somnotate REM was corrected to NREM: inspect theta/delta balance and REM transition boundaries."
    if som == "NREM" and final == "Wake":
        return "Somnotate NREM was corrected to Wake: quiet wake vs NREM ambiguity or sleep overcall."
    if som == "NREM" and final == "REM":
        return "Reviewer upgraded NREM to REM: possible missed REM or transition into REM."
    if som == "Wake" and final == "NREM":
        return "Somnotate Wake was corrected to NREM: low-EMG quiet sleep may have been missed."
    if som == "Wake" and final == "REM":
        return "Somnotate Wake was corrected to REM: check for REM with movement or model miss."
    return "Somnotate and Final differ at the event peak; inspect raw EEG/EMG and neighboring epochs."


def render_somnotate_final_patterns(events):
    """Summarize biologically meaningful Somnotate vs reviewed Final differences."""
    if events is None or len(events) == 0:
        return html.Div(className="card", children=[
            html.H4("Somnotate vs Final biological patterns"),
            html.Div("No flagged events to compare yet.", className="app-subtitle"),
        ])

    df = events.copy()
    df["somnotate_peak"] = event_source_series(df, "Somnotate")
    df["final_peak"] = event_source_series(df, "Final")
    df["layer1_peak"] = event_source_series(df, "Layer 1").map(lambda x: normalize_state_label(x, collapse_sleep=True))

    df["somnotate_peak"] = df["somnotate_peak"].map(normalize_state_label)
    df["final_peak"] = df["final_peak"].map(normalize_state_label)

    total_events = len(df)
    has_som = df["somnotate_peak"] != "Undefined"
    final_reviewed = ~df["final_peak"].map(is_unreviewed_state)
    usable = df[has_som & final_reviewed].copy()
    unreviewed_n = int((has_som & ~final_reviewed).sum())
    no_som_n = int((~has_som).sum())

    if len(usable) == 0:
        pieces = [
            "No flagged events have both a defined Somnotate state and a reviewed Final state yet."
        ]
        if unreviewed_n:
            pieces.append(f"{unreviewed_n} flagged events have Somnotate defined but Final is still Undefined/unreviewed.")
        if no_som_n:
            pieces.append(f"{no_som_n} flagged events have no defined Somnotate state at the peak.")
        return html.Div(className="card", children=[
            html.H4("Somnotate vs Final biological patterns"),
            html.Div(" ".join(pieces), className="app-subtitle"),
        ])

    agreement_n = int((usable["somnotate_peak"] == usable["final_peak"]).sum())
    disagree = usable[usable["somnotate_peak"] != usable["final_peak"]].copy()
    disagreement_n = len(disagree)

    cards = html.Div(className="dashboard-grid", children=[
        compact_metric_card("Flagged events with reviewed Final", len(usable)),
        compact_metric_card("Somnotate = Final", f"{agreement_n} ({100*agreement_n/max(len(usable),1):.1f}%)"),
        compact_metric_card("Somnotate ≠ Final", f"{disagreement_n} ({100*disagreement_n/max(len(usable),1):.1f}%)"),
        compact_metric_card("Final still unreviewed", unreviewed_n),
    ])

    if disagreement_n == 0:
        return html.Div(className="card", children=[
            html.H4("Somnotate vs Final biological patterns"),
            html.Div(
                "Among flagged events with reviewed Final scoring, Somnotate and Final agree at the event peak. "
                "This suggests the current flagged events are not mainly reviewer corrections of Somnotate state labels.",
                className="app-subtitle",
                style={"marginBottom": "8px"},
            ),
            cards,
        ])

    disagree["pattern"] = "Somnotate " + disagree["somnotate_peak"].astype(str) + " / Final " + disagree["final_peak"].astype(str)
    rows = []
    for (som, final, pattern), sub in disagree.groupby(["somnotate_peak", "final_peak", "pattern"], dropna=False):
        layer_mode = sub["layer1_peak"].mode()
        rows.append({
            "pattern": pattern,
            "events": int(len(sub)),
            "% Somnotate-Final disagreements": round(100.0 * len(sub) / max(disagreement_n, 1), 1),
            "main Layer 1 context": str(layer_mode.iloc[0]) if len(layer_mode) else "Undefined",
            "interpretation": somnotate_final_interpretation(som, final),
        })

    table = pd.DataFrame(rows).sort_values("events", ascending=False).head(12)
    most = table.iloc[0]
    summary = (
        f"Most common Somnotate/Final correction pattern: {most['pattern']} "
        f"({int(most['events'])} events, {most['% Somnotate-Final disagreements']:.1f}% of Somnotate-Final disagreements). "
        "This section ignores unreviewed Final epochs so it reflects reviewer decisions, not missing review."
    )

    return html.Div(className="card", children=[
        html.H4("Somnotate vs Final biological patterns"),
        html.Div(
            "This is the main biological/reviewer summary: in the flagged events, did the reviewer keep Somnotate's Wake/NREM/REM state, "
            "or systematically correct it to another biological state?",
            className="app-subtitle",
            style={"marginBottom": "8px"},
        ),
        cards,
        html.Div(summary, className="app-subtitle", style={"margin": "8px 0"}),
        dash_table.DataTable(
            data=table.to_dict("records"),
            columns=[{"name": c, "id": c} for c in table.columns],
            page_size=8,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": 12, "padding": "8px", "textAlign": "left", "whiteSpace": "normal", "height": "auto", "maxWidth": "440px"},
            style_header={"fontWeight": "bold", "background": "#F3F4F6", "border": "1px solid #E5E7EB"},
            style_data={"border": "1px solid #E5E7EB"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#FAFAFA"}],
        ),
    ])


def find_state_column(df, candidates):
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def render_somnotate_rem_outcome(recording_dir):
    """Compare all Somnotate REM epochs with reviewed Final labels."""
    recording_dir = Path(recording_dir)
    som = safe_read_csv(recording_dir / "somnotate" / "somnotate_results_timeseries.csv")
    final = safe_read_csv(recording_dir / "final_scoring.csv")
    layer1 = safe_read_csv(recording_dir / "layer1_wake_sleep.csv")

    if som is None or len(som) == 0:
        return html.Div(className="card", children=[
            html.H4("Somnotate REM review outcome"),
            html.Div("No Somnotate timeseries found yet. Run or import Somnotate results first.", className="app-subtitle"),
        ])

    som_state_col = find_state_column(som, ["somnotate_state", "state", "predicted_state", "label"])
    if som_state_col is None or not {"t0_s", "t1_s"}.issubset(som.columns):
        return html.Div(className="card", children=[
            html.H4("Somnotate REM review outcome"),
            html.Div("Somnotate results were found, but the app could not identify t0_s/t1_s and state columns.", className="app-subtitle"),
        ])

    som = som.copy()
    som["som_state_clean"] = som[som_state_col].map(normalize_state_label)
    rem_mask = som["som_state_clean"] == "REM"
    n_rem = int(rem_mask.sum())

    if n_rem == 0:
        return html.Div(className="card", children=[
            html.H4("Somnotate REM review outcome"),
            html.Div("Somnotate did not label any REM epochs in this recording.", className="app-subtitle"),
        ])

    rem = som.loc[rem_mask, ["t0_s", "t1_s", "som_state_clean"]].copy()
    if final is not None and len(final) and {"t0_s", "t1_s", "final_state"}.issubset(final.columns):
        rem["final_state_at_midpoint"] = labels_at_epoch_midpoints(rem, final, "final_state")
    else:
        rem["final_state_at_midpoint"] = "Undefined"

    if layer1 is not None and len(layer1) and {"t0_s", "t1_s", "layer1_label"}.issubset(layer1.columns):
        rem["layer1_at_midpoint"] = labels_at_epoch_midpoints(rem, layer1, "layer1_label")
        rem["layer1_at_midpoint"] = rem["layer1_at_midpoint"].map(lambda x: normalize_state_label(x, collapse_sleep=True))
    else:
        rem["layer1_at_midpoint"] = "Undefined"

    rem["final_clean"] = rem["final_state_at_midpoint"].map(normalize_state_label)
    kept = int((rem["final_clean"] == "REM").sum())
    corrected_nrem = int((rem["final_clean"] == "NREM").sum())
    corrected_wake = int((rem["final_clean"] == "Wake").sum())
    unreviewed = int(rem["final_clean"].map(is_unreviewed_state).sum())
    reviewed = max(n_rem - unreviewed, 0)
    corrected = corrected_nrem + corrected_wake + int((~rem["final_clean"].isin(["REM", "NREM", "Wake", "Undefined"])).sum())

    def pct(n, denom=n_rem):
        return f"{100*n/max(denom,1):.1f}%"

    cards = html.Div(className="dashboard-grid", children=[
        compact_metric_card("Somnotate REM epochs", n_rem),
        compact_metric_card("Kept as Final REM", f"{kept} ({pct(kept)})"),
        compact_metric_card("Corrected to NREM/Wake", f"{corrected_nrem + corrected_wake} ({pct(corrected_nrem + corrected_wake)})"),
        compact_metric_card("Still unreviewed", f"{unreviewed} ({pct(unreviewed)})"),
    ])

    outcome_counts = rem["final_clean"].value_counts().rename_axis("Final state").reset_index(name="Somnotate REM epochs")
    outcome_counts["% Somnotate REM"] = (100.0 * outcome_counts["Somnotate REM epochs"] / max(n_rem, 1)).round(1)

    # Build REM episodes from consecutive Somnotate REM epochs.
    rem_sorted = rem.sort_values("t0_s").reset_index(drop=True)
    median_step = np.nanmedian((rem_sorted["t1_s"].astype(float) - rem_sorted["t0_s"].astype(float)).to_numpy())
    if not np.isfinite(median_step) or median_step <= 0:
        median_step = 1.0
    episode_ids = []
    ep = 0
    prev_end = None
    for _, row in rem_sorted.iterrows():
        t0 = float(row["t0_s"])
        if prev_end is None or t0 - prev_end > max(2.0 * median_step, median_step + 1e-6):
            ep += 1
        episode_ids.append(ep)
        prev_end = float(row["t1_s"])
    rem_sorted["episode"] = episode_ids

    episode_rows = []
    for ep_id, sub in rem_sorted.groupby("episode"):
        duration = float(sub["t1_s"].astype(float).max() - sub["t0_s"].astype(float).min())
        pct_kept = 100.0 * (sub["final_clean"] == "REM").mean()
        pct_reviewed = 100.0 * (~sub["final_clean"].map(is_unreviewed_state)).mean()
        final_mode = sub["final_clean"].mode()
        layer_mode = sub["layer1_at_midpoint"].mode()
        episode_rows.append({
            "REM episode": int(ep_id),
            "start min": round(float(sub["t0_s"].astype(float).min()) / 60.0, 2),
            "end min": round(float(sub["t1_s"].astype(float).max()) / 60.0, 2),
            "duration s": round(duration, 1),
            "% kept Final REM": round(pct_kept, 1),
            "% reviewed": round(pct_reviewed, 1),
            "main Final state": str(final_mode.iloc[0]) if len(final_mode) else "Undefined",
            "main Layer 1 state": str(layer_mode.iloc[0]) if len(layer_mode) else "Undefined",
        })
    episodes = pd.DataFrame(episode_rows)
    if len(episodes):
        # Put most corrected/reviewed and longer REM episodes first.
        episodes = episodes.sort_values(["% kept Final REM", "% reviewed", "duration s"], ascending=[True, False, False]).head(12)

    note = (
        "This section asks: when Somnotate called REM, did the reviewer keep REM in Final, "
        "correct it to NREM/Wake, or leave it unreviewed? This is useful for spotting REM overcalls, "
        "REM with movement/high EMG, or systematic reviewer corrections."
    )

    children = [
        html.H4("Somnotate REM review outcome"),
        html.Div(note, className="app-subtitle", style={"marginBottom": "8px"}),
        cards,
        html.Div(style={"height": "8px"}),
        html.H5("All Somnotate REM epochs: Final outcome"),
        dash_table.DataTable(
            data=outcome_counts.to_dict("records"),
            columns=[{"name": c, "id": c} for c in outcome_counts.columns],
            page_size=6,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": 12, "padding": "8px", "textAlign": "left"},
            style_header={"fontWeight": "bold", "background": "#F3F4F6", "border": "1px solid #E5E7EB"},
            style_data={"border": "1px solid #E5E7EB"},
        ),
    ]
    if len(episodes):
        children += [
            html.H5("Somnotate REM episodes most likely corrected or needing review"),
            dash_table.DataTable(
                data=episodes.to_dict("records"),
                columns=[{"name": c, "id": c} for c in episodes.columns],
                page_size=8,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 12, "padding": "8px", "textAlign": "left", "whiteSpace": "normal", "height": "auto"},
                style_header={"fontWeight": "bold", "background": "#F3F4F6", "border": "1px solid #E5E7EB"},
                style_data={"border": "1px solid #E5E7EB"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#FAFAFA"}],
            ),
        ]
    return html.Div(className="card", children=children)

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

    children.append(html.Div(className="card", children=[
        html.H4("How events are ranked"),
        html.Div(
            "Epochs above the threshold are merged into review events. "
            "Events are ranked by strongest peak dissociation score, then average score, then duration. "
            "Rank #1 is therefore the strongest suspicious event, not necessarily the longest one.",
            className="app-subtitle",
        ),
    ]))

    # Prioritize biologically/reviewer-oriented summaries first.
    # Somnotate vs Final is usually more useful than Somnotate vs Layer 1 when
    # Layer 1 is missing/Undefined for many events.
    children.append(render_somnotate_final_patterns(events))
    children.append(render_somnotate_rem_outcome(analysis_dir.parent))
    children.append(render_flagged_state_patterns(events))

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
    host = os.environ.get("SLEEP_QC_HOST", "127.0.0.1")
    port = int(os.environ.get("SLEEP_QC_PORT", "8050"))
    app.run(debug=False, use_reloader=False, host=host, port=port)
