from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

APP_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = APP_ROOT / "VERSION"
APP_VERSION = VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else "dev"
TESTED_SOMNOTATE_VERSION = "0.5.0"
TESTED_SOMNOTATE_COMMIT = "a20f33de62511d8c172e333896608b7fc166d0f0"

# =============================================================================
# GENERAL HELPERS
# =============================================================================


def run_step(cmd: list[str], title: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(" ".join(str(x) for x in cmd))
    print()
    env = os.environ.copy()
    # Somnotate's validation script calls plt.show() even without --show.
    # Force a non-interactive backend so model QC works on Windows/macOS/Linux
    # without opening or blocking on a GUI window.
    env.setdefault("MPLBACKEND", "Agg")
    result = subprocess.run(
        [str(x) for x in cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)

        combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
        if ("features" in combined_output and "expecting" in combined_output and "LinearDiscriminantAnalysis" in combined_output):
            print()
            print("Somnotate feature-count mismatch detected.")
            print("This usually means the selected model was trained with a different")
            print("Somnotate epoch length / time_resolution or preprocessing configuration")
            print("than the one selected in the app.")
            print()
            print("For example, at 512 Hz with EEG+EMG:")
            print("  1 s Somnotate epochs often produce 156 features.")
            print("  5 s legacy Somnotate epochs often produce 788 features.")
            print()
            print("Fix: choose the epoch length that matches the model, or train a new")
            print("model at the epoch length you want to use.")

        raise RuntimeError(f"Failed: {title}")
    if result.stderr:
        # Some Somnotate scripts emit warnings to stderr even on success.
        print("[stderr]")
        print(result.stderr)
    return result


def resolve_python(python_executable: str = "", conda_env: str = "") -> str:
    """Resolve the Somnotate Python interpreter across Windows, macOS, and Linux."""

    if python_executable:
        candidate = Path(python_executable).expanduser()

        if candidate.is_file():
            print(f"Using explicit Somnotate Python: {candidate}")
            return str(candidate)

        raise FileNotFoundError(
            f"Somnotate Python executable was not found: {candidate}"
        )

    if not conda_env:
        print(f"No Somnotate environment selected; using: {sys.executable}")
        return sys.executable

    candidates: list[Path] = []

    # Ask Conda for registered environment locations.
    conda_executable = shutil.which("conda")

    if conda_executable:
        try:
            result = subprocess.run(
                [conda_executable, "env", "list", "--json"],
                capture_output=True,
                text=True,
                check=True,
            )

            conda_data = json.loads(result.stdout)

            for prefix_text in conda_data.get("envs", []):
                prefix = Path(prefix_text)

                if prefix.name.lower() != conda_env.lower():
                    continue

                if sys.platform.startswith("win"):
                    candidates.append(prefix / "python.exe")
                else:
                    candidates.append(prefix / "bin" / "python")

        except Exception as exc:
            print(f"Warning: could not query Conda environments: {exc}")

    home = Path.home()

    if sys.platform.startswith("win"):
        candidates.extend(
            [
                home / "AppData" / "Local" / "miniconda3" / "envs" / conda_env / "python.exe",
                home / "AppData" / "Local" / "anaconda3" / "envs" / conda_env / "python.exe",
                home / "miniconda3" / "envs" / conda_env / "python.exe",
                home / "anaconda3" / "envs" / conda_env / "python.exe",
                home / "mambaforge" / "envs" / conda_env / "python.exe",
            ]
        )
    else:
        candidates.extend(
            [
                home / "anaconda3" / "envs" / conda_env / "bin" / "python",
                home / "miniconda3" / "envs" / conda_env / "bin" / "python",
                home / "mambaforge" / "envs" / conda_env / "bin" / "python",
                home / "micromamba" / "envs" / conda_env / "bin" / "python",
            ]
        )

    checked: list[Path] = []
    seen: set[str] = set()

    for candidate in candidates:
        key = str(candidate).lower()

        if key in seen:
            continue

        seen.add(key)
        checked.append(candidate)

        if candidate.is_file():
            print(f"Using Somnotate environment '{conda_env}':")
            print(candidate)
            return str(candidate)

    checked_text = "\n".join(f"  - {candidate}" for candidate in checked)

    raise FileNotFoundError(
        f"Could not locate Python for Conda environment '{conda_env}'.\n"
        f"Checked:\n{checked_text}\n\n"
        "Run `conda env list` to verify that the environment exists, "
        "or provide the complete Python path in the Somnotate Python field."
    )


def split_ids(x: str | None) -> list[str]:
    return [s.strip() for s in str(x or "").split(",") if s.strip()]


def resolve_recording_dir(
    project_root: Path,
    recording_id: str,
    required_files: tuple[str, ...] = (),
) -> Path:
    """Resolve a recording folder portably across moved/copied projects.

    Prefer ``<project_root>/recordings/<recording_id>``, but fall back to the
    ``recording_dir`` stored in ``recordings_manifest.csv`` when that points to
    a valid folder.  ``required_files`` prevents an empty/stale canonical
    folder from hiding a usable manifest location.
    """
    project_root = Path(project_root).expanduser().resolve()
    canonical = project_root / "recordings" / str(recording_id)
    candidates: list[Path] = [canonical]

    manifest_path = project_root / "recordings_manifest.csv"
    if manifest_path.exists():
        try:
            manifest = pd.read_csv(manifest_path)
            if "recording_id" in manifest.columns:
                match = manifest[manifest["recording_id"].astype(str) == str(recording_id)]
                if len(match) and "recording_dir" in match.columns:
                    raw = match.iloc[0]["recording_dir"]
                    if pd.notna(raw) and str(raw).strip():
                        candidate = Path(str(raw)).expanduser()
                        if not candidate.is_absolute():
                            candidate = project_root / candidate
                        if candidate not in candidates:
                            candidates.append(candidate)
        except Exception:
            # A malformed manifest should not prevent the canonical layout from
            # being tried; the final error below will show what was checked.
            pass

    for candidate in candidates:
        if not candidate.exists():
            continue
        if all((candidate / rel).exists() for rel in required_files):
            return candidate.resolve()

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    missing = ""
    if required_files:
        missing = "\nRequired files: " + ", ".join(required_files)
    raise FileNotFoundError(
        f"Recording '{recording_id}' is not available as a prepared Sleep Stage QC recording.\n"
        f"Checked recording folders:\n{checked}{missing}\n\n"
        "If the recording was moved between computers, update/reload the project so "
        "recordings_manifest.csv points to the current location. If it has not yet "
        "been imported into the app, import/prepare it before running Somnotate QC."
    )


def read_metadata(rec_dir: Path) -> dict[str, Any]:
    path = rec_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def normalize_state(x: Any) -> str:
    """Normalize Somnotate/upstream state names to the app's canonical labels."""
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

    if compact in {"awake", "wake", "wk", "w"} or "wake" in low:
        return "Wake"
    if compact in {"nrem", "nonrem", "nr", "sws", "slowwavesleep"} or "non rem" in low:
        return "NREM"
    if compact in {"rem", "ps"}:
        return "REM"
    if compact in {"sleep"}:
        return "NREM"
    if compact in {"undefined", "uncertain", "unknown", "nd", "tr", "artifact", "artf", "nan", "none", "null"}:
        return "Undefined"
    return s


def epoch_tag(epoch_sec: float) -> str:
    x = float(epoch_sec)
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}s"
    return (f"{x:g}".replace(".", "p")) + "s"


def same_epoch(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


# =============================================================================
# PREPARE RECORDINGS FOR SOMNOTATE
# =============================================================================


def resample_if_needed(x: np.ndarray, fs_in: float, fs_out: float) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    if fs_out <= 0 or abs(float(fs_in) - float(fs_out)) < 1e-6:
        return x, float(fs_in)
    frac = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
    y = resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)
    return y, float(fs_out)


def write_edf(edf_path: Path, eeg: np.ndarray, emg: np.ndarray, fs: float) -> None:
    try:
        import pyedflib
    except ImportError as e:
        raise ImportError(
            "pyedflib is required to write EDF files. Install it in the app environment with:\n"
            "python -m pip install pyedflib"
        ) from e

    edf_path = Path(edf_path)
    edf_path.parent.mkdir(parents=True, exist_ok=True)
    signals = [np.asarray(eeg, dtype=np.float64), np.asarray(emg, dtype=np.float64)]
    labels = ["EEG", "EMG"]
    headers = []
    for label, sig in zip(labels, signals):
        physical_min = float(np.nanmin(sig))
        physical_max = float(np.nanmax(sig))
        if not np.isfinite(physical_min) or not np.isfinite(physical_max):
            raise ValueError(f"{label} contains non-finite values.")
        if physical_min == physical_max:
            physical_min -= 1.0
            physical_max += 1.0
        headers.append(
            {
                "label": label,
                "dimension": "uV",
                "sample_frequency": float(fs),
                "physical_min": physical_min,
                "physical_max": physical_max,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "",
                "prefilter": "",
            }
        )

    with pyedflib.EdfWriter(
        str(edf_path),
        n_channels=2,
        file_type=pyedflib.FILETYPE_EDFPLUS,
    ) as f:
        f.setSignalHeaders(headers)
        f.writeSamples(signals)


def somnotate_state_label(x: Any) -> str:
    """Convert app labels to labels expected by Somnotate's example pipeline."""
    state = normalize_state(x)
    return {
        "Wake": "awake",
        "NREM": "non-REM",
        "REM": "REM",
        "Undefined": "undefined",
    }.get(state, "undefined")


def export_manual_for_somnotate(manual_csv: Path, out_path: Path) -> str:
    """Export app manual scoring as a Visbrain/Somnotate hypnogram.

    Somnotate 0.5.0 reads this format with two header lines followed by
    ``state<TAB>end_time_seconds`` rows.  Using the upstream labels here is
    essential for model training (awake / non-REM / REM / undefined).
    """
    if not manual_csv.exists():
        return ""
    manual = pd.read_csv(manual_csv)
    if "manual_state" not in manual.columns or not {"t0_s", "t1_s"}.issubset(manual.columns):
        return ""

    manual = manual.sort_values("t0_s").reset_index(drop=True)
    if len(manual) == 0:
        return ""

    rows: list[tuple[str, float]] = []
    current_state: str | None = None
    current_end: float | None = None

    for _, r in manual.iterrows():
        state = somnotate_state_label(r["manual_state"])
        end_s = float(r["t1_s"])
        if current_state is None:
            current_state = state
            current_end = end_s
            continue
        if state == current_state:
            current_end = end_s
        else:
            rows.append((current_state, float(current_end)))
            current_state = state
            current_end = end_s

    if current_state is not None and current_end is not None:
        rows.append((current_state, float(current_end)))

    total_duration = float(max(end_s for _, end_s in rows))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"*Duration_sec\t{total_duration:.6f}\n")
        f.write("*Datafile\tUnspecified\n")
        for state, end_s in rows:
            f.write(f"{state}\t{end_s:.6f}\n")
    return str(out_path)


def prepare_one_recording(project_root: Path, recording_id: str, target_fs: float, epoch_sec: float) -> Path:
    rec_dir = resolve_recording_dir(
        project_root, recording_id, required_files=("metadata.json", "eeg.npy", "emg.npy")
    )
    som_dir = rec_dir / "somnotate"
    som_dir.mkdir(parents=True, exist_ok=True)

    tag = epoch_tag(epoch_sec)
    metadata = read_metadata(rec_dir)
    fs = float(metadata["sampling_rate_hz"])
    eeg_path = rec_dir / "eeg.npy"
    emg_path = rec_dir / "emg.npy"
    if not eeg_path.exists():
        raise FileNotFoundError(eeg_path)
    if not emg_path.exists():
        raise FileNotFoundError(emg_path)

    eeg = np.load(eeg_path, mmap_mode="r")
    emg = np.load(emg_path, mmap_mode="r")
    n = min(len(eeg), len(emg))
    eeg = np.asarray(eeg[:n], dtype=np.float32)
    emg = np.asarray(emg[:n], dtype=np.float32)

    eeg_rs, fs_out = resample_if_needed(eeg, fs, target_fs)
    emg_rs, fs_out = resample_if_needed(emg, fs, target_fs)
    n2 = min(len(eeg_rs), len(emg_rs))
    eeg_rs = eeg_rs[:n2]
    emg_rs = emg_rs[:n2]

    edf_path = som_dir / f"somnotate_input_{tag}.edf"
    preprocessed_path = som_dir / f"somnotate_preprocessed_{tag}.npy"
    automated_path = som_dir / f"somnotate_automated_{tag}.tsv"
    probabilities_path = som_dir / f"somnotate_state_probabilities_{tag}.npz"
    review_intervals_path = som_dir / f"somnotate_review_intervals_{tag}.tsv"
    manual_somnotate_path = som_dir / "somnotate_manual.tsv"

    write_edf(edf_path, eeg_rs, emg_rs, fs_out)

    manual_for_somnotate = export_manual_for_somnotate(
        manual_csv=rec_dir / "manual_scoring_aligned.csv",
        out_path=manual_somnotate_path,
    )

    manifest = pd.DataFrame(
        [
            {
                "recording_id": recording_id,
                "file_path_raw_signals": str(edf_path),
                "file_path_preprocessed_signals": str(preprocessed_path),
                "file_path_automated_state_annotation": str(automated_path),
                "file_path_state_probabilities": str(probabilities_path),
                "file_path_review_intervals": str(review_intervals_path),
                "file_path_manual_state_annotation": manual_for_somnotate,
                "sampling_frequency_in_hz": fs_out,
                "somnotate_epoch_sec": float(epoch_sec),
                # Channel-label columns expected by this Somnotate configuration.py
                "frontal_eeg_signal_label": "EEG",
                "emg_signal_label": "EMG",
                # Extra aliases for compatibility with other Somnotate configurations
                "EEG": "EEG",
                "EMG": "EMG",
                "eeg": "EEG",
                "emg": "EMG",
                "channel_eeg": "EEG",
                "channel_emg": "EMG",
            }
        ]
    )

    manifest_path = som_dir / f"somnotate_manifest_{tag}.csv"
    manifest.to_csv(manifest_path, index=False)

    # Current-run metadata lets later import-results infer the correct epoch length.
    run_meta = {
        "recording_id": recording_id,
        "somnotate_epoch_sec": float(epoch_sec),
        "target_fs": float(fs_out),
        "manifest_path": str(manifest_path),
        "automated_path": str(automated_path),
        "probabilities_path": str(probabilities_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(som_dir / f"somnotate_run_metadata_{tag}.json", run_meta)
    write_json(som_dir / "somnotate_current_run_metadata.json", run_meta)

    print()
    print("Prepared recording for Somnotate.")
    print("Recording:", recording_id)
    print("Epoch length:", f"{float(epoch_sec):g} s")
    print("EDF:", edf_path)
    print("Manifest:", manifest_path)
    print("Sampling rate:", fs_out)
    if manual_for_somnotate:
        print("Manual annotation:", manual_for_somnotate)
    else:
        print("Manual annotation: not found")
    return manifest_path


def _rebase_somnotate_manifest_row(project_root: Path, recording_id: str, row: dict[str, Any], epoch_sec: float) -> dict[str, Any]:
    """Repair generated Somnotate paths after a project is moved between machines."""
    tag = epoch_tag(epoch_sec)
    rec_dir = resolve_recording_dir(project_root, recording_id)
    som_dir = rec_dir / "somnotate"
    row = dict(row)
    canonical = {
        "file_path_raw_signals": som_dir / f"somnotate_input_{tag}.edf",
        "file_path_preprocessed_signals": som_dir / f"somnotate_preprocessed_{tag}.npy",
        "file_path_automated_state_annotation": som_dir / f"somnotate_automated_{tag}.tsv",
        "file_path_state_probabilities": som_dir / f"somnotate_state_probabilities_{tag}.npz",
        "file_path_review_intervals": som_dir / f"somnotate_review_intervals_{tag}.tsv",
    }
    for key, path in canonical.items():
        row[key] = str(path)
    manual = som_dir / "somnotate_manual.tsv"
    row["file_path_manual_state_annotation"] = str(manual) if manual.exists() else ""
    row["sampling_frequency_in_hz"] = float(row.get("sampling_frequency_in_hz", 512.0))
    row["somnotate_epoch_sec"] = float(epoch_sec)
    row["frontal_eeg_signal_label"] = "EEG"
    row["emg_signal_label"] = "EMG"
    return row


def combine_manifests(project_root: Path, recording_ids: list[str], out_path: Path, epoch_sec: float) -> Path:
    tag = epoch_tag(epoch_sec)
    rows = []
    for rec_id in recording_ids:
        rec_dir = resolve_recording_dir(
            project_root, rec_id, required_files=(f"somnotate/somnotate_manifest_{tag}.csv",)
        )
        manifest_path = rec_dir / "somnotate" / f"somnotate_manifest_{tag}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Somnotate manifest missing for {rec_id}: {manifest_path}. "
                f"Run Prepare first with Somnotate epoch sec = {float(epoch_sec):g}."
            )
        df = pd.read_csv(manifest_path)
        row = _rebase_somnotate_manifest_row(project_root, rec_id, df.iloc[0].to_dict(), epoch_sec)
        # Persist repaired paths so a copied Mac/Windows project remains portable.
        pd.DataFrame([row]).to_csv(manifest_path, index=False)
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


# =============================================================================
# TEMPORARY SOMNOTATE PIPELINE WITH APP-CONTROLLED TIME RESOLUTION
# =============================================================================


def find_somnotate_pipeline_dir(somnotate_root: Path) -> Path:
    """Accept /path/to/somnotate, /path/to/somnotate/example_pipeline, or a parent."""
    root = Path(somnotate_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Somnotate path does not exist: {root}")

    candidates: list[Path] = []
    candidates.append(root)
    candidates.append(root / "example_pipeline")
    for d in root.rglob("example_pipeline"):
        if d.is_dir():
            candidates.append(d)
    for f in root.rglob("01_preprocess_signals.py"):
        if f.is_file():
            candidates.append(f.parent)

    unique: list[Path] = []
    seen = set()
    for c in candidates:
        try:
            cr = c.resolve()
        except Exception:
            continue
        if cr not in seen:
            seen.add(cr)
            unique.append(cr)

    for c in unique:
        if all((c / name).exists() for name in [
            "01_preprocess_signals.py",
            "02_test_state_annotation.py",
            "03_train_state_annotation.py",
            "04_run_state_annotation.py",
            "07_compute_state_probabilities.py",
            "configuration.py",
            "data_io.py",
        ]):
            return c

    checked = "\n".join(str(c) for c in unique[:20])
    raise FileNotFoundError(
        "Could not locate Somnotate example_pipeline scripts.\n\n"
        f"Input path was: {root}\n\n"
        "Expected a folder containing at least:\n"
        " 01_preprocess_signals.py\n"
        " 02_test_state_annotation.py\n"
        " 03_train_state_annotation.py\n"
        " 04_run_state_annotation.py\n"
        " 07_compute_state_probabilities.py\n"
        " configuration.py\n"
        " data_io.py\n\n"
        "Checked candidate folders:\n"
        f"{checked}"
    )


def _copytree_ignore(dir_name: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".DS_Store"}
    return {name for name in names if name in ignored or name.endswith(".pyc")}


def patch_configuration_time_resolution(config_path: Path, epoch_sec: float) -> None:
    txt = config_path.read_text()
    new_line = f"time_resolution = {float(epoch_sec):g}"
    if re.search(r"^\s*time_resolution\s*=.*$", txt, flags=re.MULTILINE):
        txt = re.sub(r"^\s*time_resolution\s*=.*$", new_line, txt, flags=re.MULTILINE)
    else:
        txt += "\n\n# App-controlled Somnotate epoch length\n" + new_line + "\n"
    config_path.write_text(txt)


def patch_configuration_signals(config_path: Path) -> None:
    """Force the temporary Somnotate pipeline to use this app's EEG+EMG layout.

    The upstream Somnotate example configuration can change over time. Current
    versions may expect frontal EEG + occipital EEG + EMG, while this app writes
    a two-channel EDF (EEG, EMG) and the bundled legacy model was trained on the
    corresponding two-signal feature matrix.

    Only the temporary pipeline copy is modified; the user's Somnotate checkout
    remains untouched.
    """
    txt = config_path.read_text()

    replacements = {
        "state_annotation_signals": (
            "state_annotation_signals = [\n"
            "    'frontal_eeg_signal_label',\n"
            "    'emg_signal_label',\n"
            "]"
        ),
        "state_annotation_signal_labels": (
            "state_annotation_signal_labels = [\n"
            "    'EEG',\n"
            "    'EMG',\n"
            "]"
        ),
        "state_annotation_signal_frequency_bands": (
            "state_annotation_signal_frequency_bands = [\n"
            "    (0.5, 30.),  # EEG\n"
            "    (10., 45.),  # EMG\n"
            "]"
        ),
    }

    for name, replacement in replacements.items():
        pattern = rf"(?ms)^\s*{re.escape(name)}\s*=\s*\[.*?^\s*\]"
        txt, count = re.subn(pattern, replacement, txt, count=1)
        if count != 1:
            raise RuntimeError(
                f"Could not patch {name} in Somnotate configuration: {config_path}"
            )

    config_path.write_text(txt)


def patch_preprocess_integer_nperseg(preprocess_script: Path) -> None:
    """Make Somnotate/lspopt spectrogram window length an integer.

    The upstream example pipeline computes ``nperseg`` as
    ``sampling_frequency_in_hz * time_resolution_in_sec``. Values read from
    the CSV manifest are commonly floats (for example 512.0), so the product
    is also a float (for example 2560.0). Current NumPy/lspopt requires an
    integer window length and otherwise raises ``TypeError: 'float' object
    cannot be interpreted as an integer``.

    Patch only the temporary Somnotate pipeline copy.
    """
    txt = preprocess_script.read_text()
    old = "nperseg  = sampling_frequency_in_hz * time_resolution_in_sec,"
    new = "nperseg  = int(round(sampling_frequency_in_hz * time_resolution_in_sec)),"
    if old not in txt:
        raise RuntimeError(
            f"Could not patch integer nperseg in Somnotate preprocessing script: {preprocess_script}"
        )
    preprocess_script.write_text(txt.replace(old, new, 1))


def patch_review_interval_time_resolution(score_script: Path) -> None:
    """Fix original example script's low-confidence intervals in the temp copy.

    Some Somnotate versions define export_intervals_with_state_probability_below_threshold
    with a time_resolution argument but call it without passing the configured value. This
    does not affect scoring labels, but it makes review intervals wrong for 2 s epochs.
    """
    txt = score_script.read_text()
    old = "export_intervals_with_state_probability_below_threshold(dataset['file_path_review_intervals'],\n                                                                state_probability,\n                                                                threshold=0.99)"
    new = "export_intervals_with_state_probability_below_threshold(dataset['file_path_review_intervals'],\n                                                                state_probability,\n                                                                threshold=0.99,\n                                                                time_resolution=time_resolution)"
    if old in txt:
        txt = txt.replace(old, new)
        score_script.write_text(txt)


def create_epoch_pipeline_copy(somnotate_root: Path, project_root: Path, epoch_sec: float) -> Path:
    source_pipeline = find_somnotate_pipeline_dir(somnotate_root)
    tag = epoch_tag(epoch_sec)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tmp_dir = project_root / "somnotate_runs" / f"_tmp_example_pipeline_{tag}_{stamp}"
    tmp_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_pipeline, tmp_dir, ignore=_copytree_ignore)
    config_path = tmp_dir / "configuration.py"
    patch_configuration_time_resolution(config_path, epoch_sec)
    patch_configuration_signals(config_path)
    patch_preprocess_integer_nperseg(tmp_dir / "01_preprocess_signals.py")
    if (tmp_dir / "04_run_state_annotation.py").exists():
        patch_review_interval_time_resolution(tmp_dir / "04_run_state_annotation.py")

    print()
    print("Using temporary Somnotate pipeline copy:")
    print(tmp_dir)
    print("Original Somnotate pipeline is unchanged:")
    print(source_pipeline)
    print("App-controlled time_resolution:", f"{float(epoch_sec):g} s")
    print("App-controlled Somnotate signals: frontal EEG + EMG")
    print("App compatibility patch: integer spectrogram nperseg")
    return tmp_dir


def somnotate_scripts(tmp_pipeline_dir: Path) -> dict[str, Path]:
    required = {
        "preprocess": tmp_pipeline_dir / "01_preprocess_signals.py",
        "test": tmp_pipeline_dir / "02_test_state_annotation.py",
        "train": tmp_pipeline_dir / "03_train_state_annotation.py",
        "score": tmp_pipeline_dir / "04_run_state_annotation.py",
        "probabilities": tmp_pipeline_dir / "07_compute_state_probabilities.py",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing Somnotate {name} script in temporary pipeline: {path}")
    return required


def preprocess_manifest(py: str, tmp_pipeline_dir: Path, manifest_path: Path) -> None:
    scripts = somnotate_scripts(tmp_pipeline_dir)
    run_step([py, str(scripts["preprocess"]), str(manifest_path)], "Somnotate preprocessing", cwd=tmp_pipeline_dir)


def train_model(py: str, tmp_pipeline_dir: Path, training_manifest: Path, model_file: Path) -> None:
    scripts = somnotate_scripts(tmp_pipeline_dir)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    run_step([py, str(scripts["train"]), str(training_manifest), str(model_file)], "Somnotate model training", cwd=tmp_pipeline_dir)


def score_manifest(py: str, tmp_pipeline_dir: Path, manifest_path: Path, model_file: Path) -> None:
    scripts = somnotate_scripts(tmp_pipeline_dir)
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    run_step([py, str(scripts["score"]), str(manifest_path), str(model_file)], "Somnotate state annotation", cwd=tmp_pipeline_dir)


def probabilities_manifest(py: str, tmp_pipeline_dir: Path, manifest_path: Path, model_file: Path) -> None:
    scripts = somnotate_scripts(tmp_pipeline_dir)
    if not model_file.exists():
        raise FileNotFoundError(model_file)
    run_step([py, str(scripts["probabilities"]), str(manifest_path), str(model_file)], "Somnotate probability computation", cwd=tmp_pipeline_dir)


def test_model_manifest(
    py: str,
    tmp_pipeline_dir: Path,
    manifest_path: Path,
    save_file: Path,
    model_file: Path | None = None,
    title: str = "Somnotate model quality control",
) -> Path:
    """Run Somnotate's own recording-level validation script and save results."""
    scripts = somnotate_scripts(tmp_pipeline_dir)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [py, str(scripts["test"]), str(manifest_path), "--savefile", str(save_file)]
    if model_file is not None:
        cmd += ["--model", str(model_file)]
    run_step(cmd, title, cwd=tmp_pipeline_dir)
    if not save_file.exists():
        raise FileNotFoundError(f"Somnotate QC did not create expected results: {save_file}")
    return save_file


SOMNOTATE_STATE_TO_INT = {"undefined": 0, "awake": 1, "non-REM": 2, "REM": 3}
SOMNOTATE_INT_TO_APP_STATE = {0: "Undefined", 1: "Wake", 2: "NREM", 3: "REM"}


def _manual_state_ids_from_manifest(manifest_path: Path) -> list[int]:
    """Recover state IDs present in manual annotations, matching Somnotate ordering."""
    df = pd.read_csv(manifest_path)
    present: set[int] = set()
    for value in df.get("file_path_manual_state_annotation", pd.Series(dtype=object)).fillna(""):
        path = Path(str(value)).expanduser()
        if not str(value).strip() or not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[2:]:
            parts = line.split("\t")
            if not parts:
                continue
            state = parts[0].strip()
            if state in SOMNOTATE_STATE_TO_INT:
                present.add(SOMNOTATE_STATE_TO_INT[state])
    return sorted(present)


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def quality_report_from_npz(npz_path: Path, manifest_path: Path, kind: str) -> dict[str, Any]:
    """Convert Somnotate's accuracy/confusion arrays into a portable JSON report."""
    z = np.load(npz_path, allow_pickle=False)
    accuracy = np.asarray(z["accuracy"], dtype=float).ravel()
    confusion = np.asarray(z["confusion"], dtype=float)
    manifest = pd.read_csv(manifest_path)
    ids = manifest.get("recording_id", pd.Series([f"recording_{i+1}" for i in range(len(accuracy))])).astype(str).tolist()
    if len(ids) != len(accuracy):
        ids = [f"recording_{i+1}" for i in range(len(accuracy))]

    state_ids = _manual_state_ids_from_manifest(manifest_path)
    n_states = int(confusion.shape[-1]) if confusion.ndim == 3 else 0
    if len(state_ids) != n_states:
        # Somnotate orders confusion labels by sorted unique integer state IDs.
        # For standard Wake/NREM/REM data this is [1,2,3].
        state_ids = [1, 2, 3][:n_states] if n_states <= 3 else list(range(n_states))

    aggregate = confusion.sum(axis=0) if confusion.ndim == 3 and len(confusion) else np.zeros((n_states, n_states))
    per_state = []
    for i, state_id in enumerate(state_ids):
        tp = float(aggregate[i, i])
        actual = float(aggregate[i, :].sum())
        predicted = float(aggregate[:, i].sum())
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, actual)
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = float(2 * precision * recall / (precision + recall))
        per_state.append(
            {
                "state_id": int(state_id),
                "state": SOMNOTATE_INT_TO_APP_STATE.get(int(state_id), str(state_id)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(round(actual)),
            }
        )

    core = [x for x in per_state if x["state"] in {"Wake", "NREM", "REM"} and x["support"] > 0]
    recalls = [x["recall"] for x in core if x["recall"] is not None]
    f1s = [x["f1"] for x in core if x["f1"] is not None]

    per_recording = [
        {"recording_id": rid, "accuracy": float(acc)}
        for rid, acc in zip(ids, accuracy.tolist())
    ]
    report = {
        "kind": kind,
        "n_recordings": int(len(accuracy)),
        "mean_accuracy": float(np.mean(accuracy)) if len(accuracy) else None,
        "std_accuracy": float(np.std(accuracy)) if len(accuracy) else None,
        "worst_accuracy": float(np.min(accuracy)) if len(accuracy) else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "per_recording": per_recording,
        "per_state": per_state,
        "confusion_labels": [SOMNOTATE_INT_TO_APP_STATE.get(int(x), str(x)) for x in state_ids],
        "confusion_matrix": aggregate.astype(int).tolist(),
        "raw_results_npz": str(npz_path),
        "manifest_path": str(manifest_path),
    }
    return report


def print_quality_summary(report: dict[str, Any], heading: str) -> None:
    def pct(x):
        return "n/a" if x is None else f"{100 * float(x):.1f}%"

    print()
    print(heading)
    print("-" * len(heading))
    print("Recordings:", report.get("n_recordings", 0))
    print("Mean accuracy:", pct(report.get("mean_accuracy")))
    print("Worst-recording accuracy:", pct(report.get("worst_accuracy")))
    print("Balanced accuracy (Wake/NREM/REM):", pct(report.get("balanced_accuracy")))
    print("Macro F1 (Wake/NREM/REM):", pct(report.get("macro_f1")))
    print("Per-state performance:")
    for row in report.get("per_state", []):
        print(
            f"  {row['state']}: precision={pct(row.get('precision'))}, "
            f"recall={pct(row.get('recall'))}, F1={pct(row.get('f1'))}, support={row.get('support', 0)}"
        )


def _manifest_rows_with_manual(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    mask = []
    for value in df.get("file_path_manual_state_annotation", pd.Series([""] * len(df))).fillna(""):
        text = str(value).strip()
        mask.append(bool(text) and Path(text).expanduser().exists())
    return df.loc[mask].copy()


# =============================================================================
# MODEL METADATA AND COMPATIBILITY
# =============================================================================


def model_metadata_paths(model_file: Path) -> list[Path]:
    model_file = Path(model_file)
    return [
        model_file.with_suffix(".metadata.json"),
        model_file.with_name(model_file.name + ".metadata.json"),
    ]


def read_model_metadata(model_file: Path) -> dict[str, Any] | None:
    for p in model_metadata_paths(model_file):
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
    return None


def write_model_metadata(model_file: Path, data: dict[str, Any]) -> Path:
    p = model_metadata_paths(model_file)[0]
    write_json(p, data)
    return p


def query_runtime_versions(python_executable: str) -> dict[str, str]:
    """Read versions from the separate Somnotate interpreter without importing them here."""
    code = "\n".join([
        "import json, sys",
        "from importlib import metadata",
        "packages = ['somnotate', 'pomegranate', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'pyedflib', 'lspopt']",
        "out = {'python': sys.version.split()[0]}",
        "for name in packages:",
        "    try:",
        "        out[name] = metadata.version(name)",
        "    except Exception as exc:",
        "        out[name] = 'MISSING: ' + repr(exc)",
        "print(json.dumps(out, sort_keys=True))",
    ])
    result = subprocess.run(
        [str(python_executable), "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not inspect the Somnotate environment.\n"
            + (result.stdout or "")
            + "\n"
            + (result.stderr or "")
        )
    info = json.loads(result.stdout.strip().splitlines()[-1])
    missing = [k for k, v in info.items() if isinstance(v, str) and v.startswith("MISSING:")]
    if missing:
        raise RuntimeError(
            "Somnotate environment is incomplete. Missing/broken packages: "
            + ", ".join(missing)
            + ". Recreate it from environment_somnotate.yml."
        )
    return {str(k): str(v) for k, v in info.items()}


def somnotate_git_commit(somnotate_root: Path) -> str | None:
    git = shutil.which("git")
    root = Path(somnotate_root).expanduser().resolve()
    if not git or not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            [git, "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def preflight_somnotate_runtime(somnotate_root: Path, python_executable: str, model_file: Path | None = None) -> dict[str, str]:
    """Fail early on setup problems before regenerating EDFs or running long steps."""
    find_somnotate_pipeline_dir(somnotate_root)
    if model_file is not None and not Path(model_file).exists():
        raise FileNotFoundError(f"Somnotate model file does not exist: {model_file}")

    runtime = query_runtime_versions(python_executable)
    print()
    print("Somnotate runtime preflight OK")
    print("  Python:", runtime.get("python"))
    print("  Somnotate:", runtime.get("somnotate"))
    print("  scikit-learn:", runtime.get("scikit-learn"))
    print("  pomegranate:", runtime.get("pomegranate"))

    if runtime.get("somnotate") != TESTED_SOMNOTATE_VERSION:
        print(
            "WARNING: This app version was tested with Somnotate "
            f"{TESTED_SOMNOTATE_VERSION}; current environment has {runtime.get('somnotate')}."
        )

    head = somnotate_git_commit(somnotate_root)
    if head:
        print("  Somnotate git commit:", head)
        if head != TESTED_SOMNOTATE_COMMIT:
            print("WARNING: tested Somnotate commit is:", TESTED_SOMNOTATE_COMMIT)
    return runtime


def check_model_runtime_compatibility(
    model_file: Path,
    runtime: dict[str, str],
    allow_mismatch: bool = False,
) -> None:
    """Protect app-trained pickled models from unsupported runtime version drift."""
    meta = read_model_metadata(model_file) or {}
    expected = dict(meta.get("runtime_versions") or {})
    if not expected:
        serialized = meta.get("serialized_scikit_learn_version")
        if serialized:
            expected["scikit-learn"] = str(serialized)

    mismatches = []
    for key in ["python", "somnotate", "scikit-learn", "pomegranate"]:
        want = expected.get(key)
        have = runtime.get(key)
        if want and have:
            if key == "python":
                want_mm = ".".join(str(want).split(".")[:2])
                have_mm = ".".join(str(have).split(".")[:2])
                equal = want_mm == have_mm
            else:
                equal = str(have) == str(want)
            if not equal:
                mismatches.append((key, want, have))

    if not mismatches:
        return

    detail = "; ".join(f"{k}: model={want}, runtime={have}" for k, want, have in mismatches)
    legacy = bool(meta.get("legacy_model") or meta.get("legacy_unverified"))
    if legacy:
        print()
        print("WARNING: legacy Somnotate model runtime mismatch:", detail)
        print("This legacy model may be used for compatibility testing, but a version-matched/retrained model is recommended for final scientific analysis.")
        return

    msg = (
        "Somnotate model/runtime version mismatch. " + detail + "\n"
        "scikit-learn does not support loading pickled estimators across versions. "
        "Use the environment recorded in the model metadata or retrain the model in the release environment."
    )
    if allow_mismatch:
        print("WARNING:", msg)
    else:
        raise RuntimeError(msg)


def check_model_epoch_compatibility(model_file: Path, epoch_sec: float, allow_mismatch: bool = False) -> None:
    meta = read_model_metadata(model_file)
    if meta is None:
        print()
        print("WARNING: Selected Somnotate model has no epoch-length metadata.")
        print("Only use this legacy/unknown model if you know it was trained with")
        print(f"the selected Somnotate epoch length: {float(epoch_sec):g} s.")
        return

    model_epoch = meta.get("somnotate_epoch_sec", meta.get("epoch_sec", meta.get("time_resolution")))
    if model_epoch is None:
        print()
        print("WARNING: Selected Somnotate model metadata does not contain somnotate_epoch_sec.")
        print(f"Selected app epoch length is {float(epoch_sec):g} s.")
        return

    if not same_epoch(float(model_epoch), epoch_sec):
        msg = (
            "Somnotate epoch mismatch.\n"
            f"Model was trained with: {float(model_epoch):g} s epochs\n"
            f"Selected in app:       {float(epoch_sec):g} s epochs\n\n"
            "Use a matching model, or rerun with the model's epoch length. "
            "Do not directly mix 1 s and 2 s Somnotate models."
        )
        if allow_mismatch:
            print("WARNING:", msg)
        else:
            raise RuntimeError(msg)

    print()
    print("Model epoch metadata OK:", f"{float(model_epoch):g} s")


# =============================================================================
# IMPORT SOMNOTATE RESULTS INTO THE APP
# =============================================================================


def parse_somnotate_automated(path: Path) -> pd.DataFrame:
    rows = []
    prev_end = 0.0
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.replace(",", "\t").split()
        if len(parts) < 2:
            continue
        try:
            end_s = float(parts[-1])
        except Exception:
            continue
        label = " ".join(parts[:-1])
        state = normalize_state(label)
        if end_s > prev_end:
            rows.append({"t0_s": prev_end, "t1_s": end_s, "somnotate_state": state})
        prev_end = max(prev_end, end_s)
    return pd.DataFrame(rows)


def state_at_times(intervals: pd.DataFrame, t_mid: np.ndarray) -> np.ndarray:
    out = []
    if intervals is None or len(intervals) == 0:
        return np.full(len(t_mid), "Undefined", dtype=object)
    src = intervals.sort_values("t0_s").reset_index(drop=True)
    j = 0
    rows = src[["t0_s", "t1_s", "somnotate_state"]].to_numpy(object)
    for t in t_mid:
        while j < len(rows) and float(rows[j][1]) <= float(t):
            j += 1
        if j < len(rows) and float(rows[j][0]) <= float(t) < float(rows[j][1]):
            out.append(str(rows[j][2]))
        else:
            out.append("Undefined")
    return np.array(out, dtype=object)


def infer_epoch_from_current_metadata(som_dir: Path, default_epoch_sec: float) -> float:
    p = som_dir / "somnotate_current_run_metadata.json"
    if p.exists():
        try:
            meta = json.loads(p.read_text())
            return float(meta.get("somnotate_epoch_sec", default_epoch_sec))
        except Exception:
            pass
    return float(default_epoch_sec)


def import_one_recording(project_root: Path, recording_id: str, epoch_sec: float | None = None) -> Path:
    rec_dir = resolve_recording_dir(project_root, recording_id, required_files=("metadata.json",))
    som_dir = rec_dir / "somnotate"
    metadata = read_metadata(rec_dir)
    duration_s = float(metadata["duration_s"])
    epoch_sec = infer_epoch_from_current_metadata(som_dir, epoch_sec or 1.0)
    tag = epoch_tag(epoch_sec)

    auto_file = som_dir / f"somnotate_automated_{tag}.tsv"
    prob_file = som_dir / f"somnotate_state_probabilities_{tag}.npz"

    # Backward compatibility for old 1 s outputs.
    if not auto_file.exists() and tag == "1s" and (som_dir / "somnotate_automated.tsv").exists():
        auto_file = som_dir / "somnotate_automated.tsv"
    if not prob_file.exists() and tag == "1s" and (som_dir / "somnotate_state_probabilities.npz").exists():
        prob_file = som_dir / "somnotate_state_probabilities.npz"

    if not auto_file.exists() and not prob_file.exists():
        raise FileNotFoundError(
            f"No Somnotate outputs found for {recording_id} at {float(epoch_sec):g} s epochs.\n"
            f"Expected {auto_file} and/or {prob_file}"
        )

    n = int(np.ceil(duration_s / float(epoch_sec)))
    t0 = np.arange(n, dtype=float) * float(epoch_sec)
    t1 = np.minimum(t0 + float(epoch_sec), duration_s)
    t_mid = (t0 + t1) / 2.0

    out = pd.DataFrame(
        {
            "recording_id": recording_id,
            "t0_s": t0,
            "t1_s": t1,
            "time_min": t0 / 60.0,
            "somnotate_epoch_sec": float(epoch_sec),
        }
    )

    if auto_file.exists():
        intervals = parse_somnotate_automated(auto_file)
        if len(intervals) == 0:
            raise RuntimeError(f"Could not parse Somnotate automated annotation: {auto_file}")
        out["somnotate_state"] = state_at_times(intervals, t_mid)
    else:
        out["somnotate_state"] = "Undefined"

    if prob_file.exists():
        z = np.load(prob_file, allow_pickle=True)
        state_names = list(z.files)
        arrays = [np.asarray(z[k], dtype=float).ravel() for k in state_names]
        if arrays:
            m = min(len(a) for a in arrays)
            # One probability value per Somnotate sample; samples are spaced by epoch_sec.
            prob_t = np.arange(m, dtype=float) * float(epoch_sec)
            for state_name, arr in zip(state_names, arrays):
                clean = normalize_state(state_name)
                out[f"somnotate_P_{clean}"] = np.interp(t_mid, prob_t, arr[:m])

        prob_cols = [c for c in out.columns if c.startswith("somnotate_P_")]
        if prob_cols:
            out["somnotate_confidence"] = out[prob_cols].max(axis=1)
            out["somnotate_uncertainty"] = 1.0 - out["somnotate_confidence"]

    tagged_out = som_dir / f"somnotate_results_timeseries_{tag}.csv"
    current_out = som_dir / "somnotate_results_timeseries.csv"
    out.to_csv(tagged_out, index=False)
    out.to_csv(current_out, index=False)

    print()
    print("Imported Somnotate results.")
    print("Recording:", recording_id)
    print("Epoch length:", f"{float(epoch_sec):g} s")
    print("Tagged output:", tagged_out)
    print("Current output:", current_out)
    print()
    print(out["somnotate_state"].value_counts(normalize=True).mul(100).round(2).to_string())
    prob_cols = [c for c in out.columns if c.startswith("somnotate_P_")]
    if prob_cols:
        print()
        print("Probability columns:", ", ".join(prob_cols))
    return current_out


# =============================================================================
# HIGH-LEVEL WORKFLOWS
# =============================================================================


def workflow_use_existing_model(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    somnotate_root = Path(args.somnotate_root).expanduser().resolve()
    py = resolve_python(args.somnotate_python, args.somnotate_conda_env)
    model_file = Path(args.model_file).expanduser().resolve()
    recording_ids = split_ids(args.recording_ids)
    epoch_sec = float(args.epoch_sec)

    if not recording_ids:
        raise ValueError("No recording IDs provided.")

    runtime = preflight_somnotate_runtime(somnotate_root, py, model_file=model_file)
    check_model_epoch_compatibility(model_file, epoch_sec, allow_mismatch=args.allow_epoch_mismatch)
    check_model_runtime_compatibility(model_file, runtime, allow_mismatch=args.allow_version_mismatch)

    if args.prepare:
        for rec_id in recording_ids:
            prepare_one_recording(project_root, rec_id, args.target_fs, epoch_sec)

    tag = epoch_tag(epoch_sec)
    manifest_path = project_root / "somnotate_runs" / f"use_existing_model_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    combine_manifests(project_root, recording_ids, manifest_path, epoch_sec)

    tmp_pipeline_dir = None
    if args.preprocess or args.score or args.probabilities:
        tmp_pipeline_dir = create_epoch_pipeline_copy(somnotate_root, project_root, epoch_sec)

    if args.preprocess:
        preprocess_manifest(py, tmp_pipeline_dir, manifest_path)
    if args.score:
        score_manifest(py, tmp_pipeline_dir, manifest_path, model_file)
    if args.probabilities:
        probabilities_manifest(py, tmp_pipeline_dir, manifest_path, model_file)
    if args.import_results:
        for rec_id in recording_ids:
            import_one_recording(project_root, rec_id, epoch_sec)


def workflow_evaluate_model(args: argparse.Namespace) -> None:
    """Evaluate a fixed, already-trained model without retraining it."""
    project_root = Path(args.project_root).expanduser().resolve()
    somnotate_root = Path(args.somnotate_root).expanduser().resolve()
    py = resolve_python(args.somnotate_python, args.somnotate_conda_env)
    model_file = Path(args.model_file).expanduser().resolve()
    recording_ids = split_ids(args.recording_ids)
    epoch_sec = float(args.epoch_sec)

    if not recording_ids:
        raise ValueError("No evaluation recording IDs provided.")

    runtime = preflight_somnotate_runtime(somnotate_root, py, model_file=model_file)
    check_model_epoch_compatibility(model_file, epoch_sec, allow_mismatch=args.allow_epoch_mismatch)
    check_model_runtime_compatibility(model_file, runtime, allow_mismatch=args.allow_version_mismatch)

    if args.prepare:
        for rec_id in recording_ids:
            prepare_one_recording(project_root, rec_id, args.target_fs, epoch_sec)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep_tag = epoch_tag(epoch_sec)
    manifest_path = project_root / "somnotate_runs" / f"evaluate_existing_model_{ep_tag}_{stamp}.csv"
    combine_manifests(project_root, recording_ids, manifest_path, epoch_sec)

    manifest = pd.read_csv(manifest_path)
    manual_rows = _manifest_rows_with_manual(manifest_path)
    if len(manual_rows) != len(manifest):
        manual_ids = set(manual_rows.get("recording_id", pd.Series(dtype=str)).astype(str).tolist())
        all_ids = manifest.get("recording_id", pd.Series(recording_ids)).astype(str).tolist()
        missing = [rid for rid in all_ids if rid not in manual_ids]
        raise FileNotFoundError(
            "Model evaluation requires manual scoring for every selected recording. "
            "Missing manual annotations for: " + ", ".join(missing)
        )

    tmp_pipeline_dir = create_epoch_pipeline_copy(somnotate_root, project_root, epoch_sec)
    if args.preprocess:
        preprocess_manifest(py, tmp_pipeline_dir, manifest_path)

    qc_dir = project_root / "somnotate_model_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_file.stem).strip("_") or "somnotate_model"
    npz_path = qc_dir / f"{safe_model}_{ep_tag}_{stamp}.npz"
    test_model_manifest(
        py,
        tmp_pipeline_dir,
        manifest_path,
        npz_path,
        model_file=model_file,
        title="Somnotate existing-model evaluation",
    )
    evaluation = quality_report_from_npz(npz_path, manifest_path, "existing-model-evaluation")
    print_quality_summary(evaluation, "Existing-model evaluation")

    independent = args.evaluation_context == "independent-validation"
    guidance = [
        "Inspect the worst-performing recording, not only the mean accuracy.",
        "Inspect Wake, NREM and REM precision/recall/F1 separately; overall accuracy can hide weak minority-state performance.",
        "Check the confusion matrix for systematic errors, especially REM confused with Wake or NREM.",
        "Visually review representative EEG/EMG and disagreement periods before relying on automated labels.",
        "There is no universal accuracy threshold that guarantees a scientifically valid sleep-stage model.",
    ]
    if independent:
        guidance.insert(0, "These recordings were declared independent of model training, so this report can be interpreted as held-out validation.")
    else:
        guidance.insert(0, "These recordings may have been used for training (or their status is unknown), so this report is descriptive and may be optimistic.")

    report = {
        "report_type": "existing-model-evaluation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_file": str(model_file),
        "somnotate_epoch_sec": float(epoch_sec),
        "target_fs": float(args.target_fs),
        "recording_ids": recording_ids,
        "evaluation_context": args.evaluation_context,
        "evaluation": evaluation,
        "somnotate_root": str(somnotate_root),
        "somnotate_git_commit": somnotate_git_commit(somnotate_root),
        "runtime_versions": runtime,
        "guidance": guidance,
    }
    report_path = qc_dir / f"{safe_model}_{ep_tag}_{stamp}.quality.json"
    write_json(report_path, report)

    print()
    print("Existing model was evaluated without retraining.")
    print("Quality report saved here:")
    print(report_path)


def workflow_train_model(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    somnotate_root = Path(args.somnotate_root).expanduser().resolve()
    py = resolve_python(args.somnotate_python, args.somnotate_conda_env)
    train_ids = split_ids(args.train_recording_ids)
    test_ids = split_ids(args.test_recording_ids)
    epoch_sec = float(args.epoch_sec)

    if not train_ids:
        raise ValueError("No training recording IDs provided.")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(
            "Training and held-out test recordings must be separate. "
            f"Remove these IDs from one set: {', '.join(overlap)}"
        )

    runtime = preflight_somnotate_runtime(somnotate_root, py)

    print()
    print("Training Somnotate model with app-controlled epoch length:", f"{epoch_sec:g} s")
    print("This model should later be used only with matching Somnotate epochs.")

    all_ids = train_ids + test_ids
    if args.prepare:
        for rec_id in all_ids:
            prepare_one_recording(project_root, rec_id, args.target_fs, epoch_sec)

    for rec_id in train_ids:
        rec_dir = resolve_recording_dir(
            project_root, rec_id,
            required_files=(f"somnotate/somnotate_manifest_{epoch_tag(epoch_sec)}.csv",),
        )
        manifest_path = rec_dir / "somnotate" / f"somnotate_manifest_{epoch_tag(epoch_sec)}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Somnotate manifest missing for training recording {rec_id}: {manifest_path}. Run Prepare first."
            )
        df = pd.read_csv(manifest_path)
        manual_path = str(df.iloc[0].get("file_path_manual_state_annotation", ""))
        if not manual_path or manual_path == "nan" or not Path(manual_path).exists():
            raise FileNotFoundError(
                f"Training recording {rec_id} has no manual annotation for Somnotate. "
                "Import a recording with manual scoring first."
            )

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep_tag = epoch_tag(epoch_sec)
    safe_name = str(args.model_name or "model").strip().replace(" ", "_").replace("/", "_").replace("\\\\", "_")
    if ep_tag not in safe_name:
        safe_name = f"{safe_name}_{ep_tag}"

    train_manifest = project_root / "somnotate_runs" / f"{safe_name}_training_manifest_{tag}.csv"
    model_file = project_root / "somnotate_models" / f"{safe_name}_{tag}.pickle"
    combine_manifests(project_root, train_ids, train_manifest, epoch_sec)
    present_state_ids = set(_manual_state_ids_from_manifest(train_manifest))
    missing_states = [SOMNOTATE_INT_TO_APP_STATE[x] for x in (1, 2, 3) if x not in present_state_ids]
    if missing_states:
        raise ValueError(
            "Training annotations do not contain all three sleep states. "
            f"Missing across the training set: {', '.join(missing_states)}. "
            "A Wake/NREM/REM model should be trained with examples of every state."
        )

    tmp_pipeline_dir = create_epoch_pipeline_copy(somnotate_root, project_root, epoch_sec)

    if args.preprocess:
        preprocess_manifest(py, tmp_pipeline_dir, train_manifest)

    quality: dict[str, Any] = {
        "report_type": "training-qc",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_file": str(model_file),
        "somnotate_epoch_sec": float(epoch_sec),
        "train_recording_ids": train_ids,
        "test_recording_ids": test_ids,
        "guidance": [
            "Prefer recording/animal-level validation; do not randomly split neighbouring epochs from the same recording.",
            "Check the worst-performing recording, not only the mean accuracy.",
            "Inspect Wake, NREM and REM precision/recall/F1 separately; overall accuracy can hide weak minority-state performance.",
            "Whenever possible, reserve independent held-out recordings/animals that were not used for final model fitting.",
            "There is no universal accuracy threshold that guarantees a scientifically valid model; visually review disagreements and representative EEG/EMG before relying on it.",
        ],
    }

    if args.quality_control:
        if len(train_ids) >= 2:
            cv_npz = model_file.with_name(model_file.stem + "_cv.npz")
            test_model_manifest(
                py,
                tmp_pipeline_dir,
                train_manifest,
                cv_npz,
                model_file=None,
                title="Somnotate leave-one-recording-out cross-validation",
            )
            quality["cross_validation"] = quality_report_from_npz(cv_npz, train_manifest, "leave-one-recording-out")
            print_quality_summary(quality["cross_validation"], "Training-set cross-validation")
        else:
            quality["cross_validation"] = None
            print()
            print("Model QC warning: leave-one-recording-out cross-validation requires at least 2 training recordings.")

    train_model(py, tmp_pipeline_dir, train_manifest, model_file)

    meta_path = write_model_metadata(
        model_file,
        {
            "created_by_app": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "app_version": APP_VERSION,
            "model_file": str(model_file),
            "somnotate_epoch_sec": float(epoch_sec),
            "target_fs": float(args.target_fs),
            "signals": ["EEG", "EMG"],
            "somnotate_configuration": "frontal EEG + EMG",
            "train_recording_ids": train_ids,
            "test_recording_ids": test_ids,
            "somnotate_root": str(somnotate_root),
            "somnotate_git_commit": somnotate_git_commit(somnotate_root),
            "runtime_versions": runtime,
        },
    )

    print()
    print("New model saved here:")
    print(model_file)
    print("Model metadata saved here:")
    print(meta_path)

    if test_ids:
        test_manifest = project_root / "somnotate_runs" / f"{safe_name}_test_manifest_{tag}.csv"
        combine_manifests(project_root, test_ids, test_manifest, epoch_sec)
        if args.preprocess:
            preprocess_manifest(py, tmp_pipeline_dir, test_manifest)

        manual_test = _manifest_rows_with_manual(test_manifest)
        if args.quality_control and len(manual_test):
            heldout_manifest = project_root / "somnotate_runs" / f"{safe_name}_heldout_qc_manifest_{tag}.csv"
            manual_test.to_csv(heldout_manifest, index=False)
            heldout_npz = model_file.with_name(model_file.stem + "_heldout_test.npz")
            test_model_manifest(
                py,
                tmp_pipeline_dir,
                heldout_manifest,
                heldout_npz,
                model_file=model_file,
                title="Somnotate held-out model quality control",
            )
            quality["heldout_test"] = quality_report_from_npz(heldout_npz, heldout_manifest, "held-out")
            print_quality_summary(quality["heldout_test"], "Held-out test performance")
        elif args.quality_control:
            quality["heldout_test"] = None
            print()
            print("Held-out QC not calculated: none of the test recordings has manual scoring.")

        score_manifest(py, tmp_pipeline_dir, test_manifest, model_file)
        probabilities_manifest(py, tmp_pipeline_dir, test_manifest, model_file)
        for rec_id in test_ids:
            import_one_recording(project_root, rec_id, epoch_sec)

    if args.quality_control:
        quality_path = model_file.with_suffix(".quality.json")
        write_json(quality_path, quality)
        meta = read_model_metadata(model_file) or {}
        meta["quality_report_file"] = str(quality_path)
        write_model_metadata(model_file, meta)
        print()
        print("Model quality report saved here:")
        print(quality_path)


def workflow_attach_outputs(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    rec_dir = resolve_recording_dir(project_root, args.recording_id)
    som_dir = rec_dir / "somnotate"
    som_dir.mkdir(parents=True, exist_ok=True)
    epoch_sec = float(args.epoch_sec)
    tag = epoch_tag(epoch_sec)

    auto_in = Path(args.automated_file).expanduser() if args.automated_file else None
    prob_in = Path(args.probability_file).expanduser() if args.probability_file else None
    if not auto_in and not prob_in:
        raise ValueError("Provide at least --automated-file or --probability-file.")

    if auto_in:
        if not auto_in.exists():
            raise FileNotFoundError(auto_in)
        dst = som_dir / f"somnotate_automated_{tag}.tsv"
        if auto_in.resolve() != dst.resolve():
            shutil.copy2(auto_in, dst)
    if prob_in:
        if not prob_in.exists():
            raise FileNotFoundError(prob_in)
        dst = som_dir / f"somnotate_state_probabilities_{tag}.npz"
        if prob_in.resolve() != dst.resolve():
            shutil.copy2(prob_in, dst)

    write_json(
        som_dir / "somnotate_current_run_metadata.json",
        {
            "recording_id": args.recording_id,
            "somnotate_epoch_sec": epoch_sec,
            "attached_outputs": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    import_one_recording(project_root, args.recording_id, epoch_sec)


# =============================================================================
# CLI
# =============================================================================


def add_epoch_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--epoch-sec",
        type=float,
        default=1.0,
        choices=[1.0, 2.0, 5.0],
        help="Somnotate epoch length/time resolution in seconds. Models are epoch-specific.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Somnotate layer for Sleep Stage QC app.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-ids", required=True)
    p.add_argument("--target-fs", type=float, default=512.0)
    add_epoch_arg(p)

    p = sub.add_parser("use-existing-model")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-ids", required=True)
    p.add_argument("--somnotate-root", required=True)
    p.add_argument("--somnotate-python", default="")
    p.add_argument("--somnotate-conda-env", default="somnotate_env")
    p.add_argument("--model-file", required=True)
    p.add_argument("--target-fs", type=float, default=512.0)
    add_epoch_arg(p)
    p.add_argument("--allow-epoch-mismatch", action="store_true", help="Advanced/debug only: do not block model epoch mismatch.")
    p.add_argument("--allow-version-mismatch", action="store_true", help="Advanced/debug only: do not block version mismatch for app-trained models.")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--probabilities", action="store_true")
    p.add_argument("--import-results", action="store_true")

    p = sub.add_parser("evaluate-model")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-ids", required=True)
    p.add_argument("--somnotate-root", required=True)
    p.add_argument("--somnotate-python", default="")
    p.add_argument("--somnotate-conda-env", default="somnotate_env")
    p.add_argument("--model-file", required=True)
    p.add_argument("--target-fs", type=float, default=512.0)
    add_epoch_arg(p)
    p.add_argument(
        "--evaluation-context",
        choices=["independent-validation", "training-or-unknown"],
        default="independent-validation",
        help="Whether the manually scored evaluation recordings were independent of model training.",
    )
    p.add_argument("--allow-epoch-mismatch", action="store_true", help="Advanced/debug only: do not block model epoch mismatch.")
    p.add_argument("--allow-version-mismatch", action="store_true", help="Advanced/debug only: do not block version mismatch for app-trained models.")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--preprocess", action="store_true")

    p = sub.add_parser("train-model")
    p.add_argument("--project-root", required=True)
    p.add_argument("--train-recording-ids", required=True)
    p.add_argument("--test-recording-ids", default="")
    p.add_argument("--somnotate-root", required=True)
    p.add_argument("--somnotate-python", default="")
    p.add_argument("--somnotate-conda-env", default="somnotate_env")
    p.add_argument("--model-name", required=True)
    p.add_argument("--target-fs", type=float, default=512.0)
    add_epoch_arg(p)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument(
        "--quality-control",
        action="store_true",
        help="Run leave-one-recording-out CV and held-out evaluation (when manual test labels exist).",
    )

    p = sub.add_parser("attach-outputs")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-id", required=True)
    p.add_argument("--automated-file", default="")
    p.add_argument("--probability-file", default="")
    add_epoch_arg(p)

    p = sub.add_parser("import-results")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-ids", required=True)
    add_epoch_arg(p)

    args = parser.parse_args()

    if args.command == "prepare":
        project_root = Path(args.project_root).expanduser().resolve()
        for rec_id in split_ids(args.recording_ids):
            prepare_one_recording(project_root, rec_id, args.target_fs, float(args.epoch_sec))
    elif args.command == "use-existing-model":
        workflow_use_existing_model(args)
    elif args.command == "evaluate-model":
        workflow_evaluate_model(args)
    elif args.command == "train-model":
        workflow_train_model(args)
    elif args.command == "attach-outputs":
        workflow_attach_outputs(args)
    elif args.command == "import-results":
        project_root = Path(args.project_root).expanduser().resolve()
        for rec_id in split_ids(args.recording_ids):
            import_one_recording(project_root, rec_id, float(args.epoch_sec))


if __name__ == "__main__":
    main()
