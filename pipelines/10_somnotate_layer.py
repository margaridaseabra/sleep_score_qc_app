from __future__ import annotations

import argparse
import json
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
    result = subprocess.run(
        [str(x) for x in cmd],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(f"Failed: {title}")
    if result.stderr:
        # Some Somnotate scripts emit warnings to stderr even on success.
        print("[stderr]")
        print(result.stderr)
    return result


def resolve_python(python_executable: str = "", conda_env: str = "") -> str:
    if python_executable:
        p = Path(python_executable).expanduser()
        if p.exists():
            return str(p)
    if conda_env:
        candidates = [
            Path.home() / "anaconda3" / "envs" / conda_env / "bin" / "python",
            Path.home() / "miniconda3" / "envs" / conda_env / "bin" / "python",
            Path.home() / "mambaforge" / "envs" / conda_env / "bin" / "python",
            Path.home() / "micromamba" / "envs" / conda_env / "bin" / "python",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return sys.executable


def split_ids(x: str | None) -> list[str]:
    return [s.strip() for s in str(x or "").split(",") if s.strip()]


def read_metadata(rec_dir: Path) -> dict[str, Any]:
    path = rec_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def normalize_state(x: Any) -> str:
    x = str(x).strip()
    mapping = {
        "Awake": "Wake",
        "Wake": "Wake",
        "WK": "Wake",
        "W": "Wake",
        "NREM": "NREM",
        "SWS": "NREM",
        "REM": "REM",
        "PS": "REM",
        "Sleep": "NREM",
        "Undefined": "Undefined",
        "ND": "Undefined",
        "TR": "Undefined",
        "Artifact": "Undefined",
        "Artf": "Undefined",
        "nan": "Undefined",
        "NaN": "Undefined",
        "None": "Undefined",
        "": "Undefined",
    }
    return mapping.get(x, x)


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


def export_manual_for_somnotate(manual_csv: Path, out_path: Path) -> str:
    """Export app manual scoring as Somnotate-style state/end-time TSV.

    This format is interval based, so it works for both 1 s and 2 s Somnotate
    epochs. Somnotate converts it to a model-sample vector using time_resolution.
    """
    if not manual_csv.exists():
        return ""
    manual = pd.read_csv(manual_csv)
    if "manual_state" not in manual.columns or not {"t0_s", "t1_s"}.issubset(manual.columns):
        return ""

    manual = manual.sort_values("t0_s").reset_index(drop=True)
    rows: list[tuple[str, float]] = []
    current_state: str | None = None
    current_end: float | None = None

    for _, r in manual.iterrows():
        state = normalize_state(r["manual_state"])
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for state, end_s in rows:
            f.write(f"{state}\t{end_s:.6f}\n")
    return str(out_path)


def prepare_one_recording(project_root: Path, recording_id: str, target_fs: float, epoch_sec: float) -> Path:
    rec_dir = project_root / "recordings" / recording_id
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


def combine_manifests(project_root: Path, recording_ids: list[str], out_path: Path, epoch_sec: float) -> Path:
    tag = epoch_tag(epoch_sec)
    rows = []
    for rec_id in recording_ids:
        manifest_path = project_root / "recordings" / rec_id / "somnotate" / f"somnotate_manifest_{tag}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Somnotate manifest missing for {rec_id}: {manifest_path}. "
                f"Run Prepare first with Somnotate epoch sec = {float(epoch_sec):g}."
            )
        df = pd.read_csv(manifest_path)
        rows.append(df.iloc[0].to_dict())
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
    patch_configuration_time_resolution(tmp_dir / "configuration.py", epoch_sec)
    if (tmp_dir / "04_run_state_annotation.py").exists():
        patch_review_interval_time_resolution(tmp_dir / "04_run_state_annotation.py")

    print()
    print("Using temporary Somnotate pipeline copy:")
    print(tmp_dir)
    print("Original Somnotate pipeline is unchanged:")
    print(source_pipeline)
    print("App-controlled time_resolution:", f"{float(epoch_sec):g} s")
    return tmp_dir


def somnotate_scripts(tmp_pipeline_dir: Path) -> dict[str, Path]:
    required = {
        "preprocess": tmp_pipeline_dir / "01_preprocess_signals.py",
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
    rec_dir = project_root / "recordings" / recording_id
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

    check_model_epoch_compatibility(model_file, epoch_sec, allow_mismatch=args.allow_epoch_mismatch)

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


def workflow_train_model(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    somnotate_root = Path(args.somnotate_root).expanduser().resolve()
    py = resolve_python(args.somnotate_python, args.somnotate_conda_env)
    train_ids = split_ids(args.train_recording_ids)
    test_ids = split_ids(args.test_recording_ids)
    epoch_sec = float(args.epoch_sec)

    if not train_ids:
        raise ValueError("No training recording IDs provided.")

    print()
    print("Training Somnotate model with app-controlled epoch length:", f"{epoch_sec:g} s")
    print("This model should later be used only with matching Somnotate epochs.")

    all_ids = train_ids + test_ids
    if args.prepare:
        for rec_id in all_ids:
            prepare_one_recording(project_root, rec_id, args.target_fs, epoch_sec)

    for rec_id in train_ids:
        manifest_path = project_root / "recordings" / rec_id / "somnotate" / f"somnotate_manifest_{epoch_tag(epoch_sec)}.csv"
        df = pd.read_csv(manifest_path)
        manual_path = str(df.iloc[0].get("file_path_manual_state_annotation", ""))
        if not manual_path or manual_path == "nan" or not Path(manual_path).exists():
            raise FileNotFoundError(
                f"Training recording {rec_id} has no manual annotation for Somnotate. "
                "Import a recording with manual scoring first."
            )

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep_tag = epoch_tag(epoch_sec)
    safe_name = str(args.model_name or "model").strip().replace(" ", "_").replace("/", "_")
    # Include epoch tag in the filename unless the user already did it.
    if ep_tag not in safe_name:
        safe_name = f"{safe_name}_{ep_tag}"

    train_manifest = project_root / "somnotate_runs" / f"{safe_name}_training_manifest_{tag}.csv"
    model_file = project_root / "somnotate_models" / f"{safe_name}_{tag}.pickle"
    combine_manifests(project_root, train_ids, train_manifest, epoch_sec)

    tmp_pipeline_dir = create_epoch_pipeline_copy(somnotate_root, project_root, epoch_sec)

    if args.preprocess:
        preprocess_manifest(py, tmp_pipeline_dir, train_manifest)

    train_model(py, tmp_pipeline_dir, train_manifest, model_file)

    meta_path = write_model_metadata(
        model_file,
        {
            "created_by_app": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_file": str(model_file),
            "somnotate_epoch_sec": float(epoch_sec),
            "target_fs": float(args.target_fs),
            "train_recording_ids": train_ids,
            "test_recording_ids": test_ids,
            "somnotate_root": str(somnotate_root),
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
        score_manifest(py, tmp_pipeline_dir, test_manifest, model_file)
        probabilities_manifest(py, tmp_pipeline_dir, test_manifest, model_file)
        for rec_id in test_ids:
            import_one_recording(project_root, rec_id, epoch_sec)


def workflow_attach_outputs(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).expanduser().resolve()
    rec_dir = project_root / "recordings" / args.recording_id
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
        choices=[1.0, 2.0],
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
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--probabilities", action="store_true")
    p.add_argument("--import-results", action="store_true")

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
