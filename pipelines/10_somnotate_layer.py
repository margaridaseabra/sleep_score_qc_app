from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def run_step(cmd, title, cwd=None):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(" ".join(str(x) for x in cmd))
    print()

    result = subprocess.run(
        cmd,
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

    return result


def resolve_python(python_executable="", conda_env=""):
    if python_executable and Path(python_executable).exists():
        return str(Path(python_executable).expanduser())

    if conda_env:
        candidate = Path.home() / "anaconda3" / "envs" / conda_env / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    return sys.executable


def split_ids(x):
    return [s.strip() for s in str(x).split(",") if s.strip()]


def read_metadata(rec_dir: Path):
    path = rec_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def normalize_state(x):
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
        "": "Undefined",
    }

    return mapping.get(x, x)


def find_probability_script(somnotate_root: Path):
    for p in sorted(somnotate_root.glob("*.py")):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue

        if "predict_all_probabilities" in txt and "file_path_state_probabilities" in txt:
            return p

    raise FileNotFoundError(
        f"Could not find Somnotate probability script in {somnotate_root}. "
        "Expected a script containing predict_all_probabilities."
    )


# =============================================================================
# PREPARE RECORDINGS FOR SOMNOTATE
# =============================================================================

def resample_if_needed(x, fs_in, fs_out):
    x = np.asarray(x, dtype=np.float32)

    if fs_out <= 0 or abs(float(fs_in) - float(fs_out)) < 1e-6:
        return x, float(fs_in)

    frac = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
    y = resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)

    return y, float(fs_out)


def write_edf(edf_path, eeg, emg, fs):
    try:
        import pyedflib
    except ImportError as e:
        raise ImportError(
            "pyedflib is required to write EDF files. Install it in sleep_app:\n"
            "python -m pip install pyedflib"
        ) from e

    edf_path = Path(edf_path)
    edf_path.parent.mkdir(parents=True, exist_ok=True)

    signals = [
        np.asarray(eeg, dtype=np.float64),
        np.asarray(emg, dtype=np.float64),
    ]

    labels = ["EEG", "EMG"]
    headers = []

    for label, sig in zip(labels, signals):
        physical_min = float(np.nanmin(sig))
        physical_max = float(np.nanmax(sig))

        if not np.isfinite(physical_min) or not np.isfinite(physical_max):
            raise ValueError(f"{label} contains non-finite values.")

        if physical_min == physical_max:
            physical_min -= 1
            physical_max += 1

        headers.append({
            "label": label,
            "dimension": "uV",
            "sample_frequency": float(fs),
            "physical_min": physical_min,
            "physical_max": physical_max,
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "",
            "prefilter": "",
        })

    with pyedflib.EdfWriter(
        str(edf_path),
        n_channels=2,
        file_type=pyedflib.FILETYPE_EDFPLUS,
    ) as f:
        f.setSignalHeaders(headers)
        f.writeSamples(signals)


def export_manual_for_somnotate(manual_csv: Path, out_path: Path):
    if not manual_csv.exists():
        return ""

    manual = pd.read_csv(manual_csv)

    if "manual_state" not in manual.columns:
        return ""

    manual = manual.sort_values("t0_s").reset_index(drop=True)

    rows = []
    current_state = None
    current_end = None

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
            rows.append((current_state, current_end))
            current_state = state
            current_end = end_s

    if current_state is not None:
        rows.append((current_state, current_end))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for state, end_s in rows:
            f.write(f"{state}\t{end_s:.6f}\n")

    return str(out_path)


def prepare_one_recording(project_root: Path, recording_id: str, target_fs: float):
    rec_dir = project_root / "recordings" / recording_id
    som_dir = rec_dir / "somnotate"
    som_dir.mkdir(parents=True, exist_ok=True)

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

    edf_path = som_dir / "somnotate_input.edf"
    preprocessed_path = som_dir / "somnotate_preprocessed.npy"
    automated_path = som_dir / "somnotate_automated.tsv"
    probabilities_path = som_dir / "somnotate_state_probabilities.npz"
    review_intervals_path = som_dir / "somnotate_review_intervals.tsv"
    manual_somnotate_path = som_dir / "somnotate_manual.tsv"

    write_edf(edf_path, eeg_rs, emg_rs, fs_out)

    manual_csv = rec_dir / "manual_scoring_aligned.csv"
    manual_for_somnotate = export_manual_for_somnotate(
        manual_csv=manual_csv,
        out_path=manual_somnotate_path,
    )

    manifest = pd.DataFrame([{
        "recording_id": recording_id,

        "file_path_raw_signals": str(edf_path),
        "file_path_preprocessed_signals": str(preprocessed_path),
        "file_path_automated_state_annotation": str(automated_path),
        "file_path_state_probabilities": str(probabilities_path),
        "file_path_review_intervals": str(review_intervals_path),
        "file_path_manual_state_annotation": manual_for_somnotate,

        "sampling_frequency_in_hz": fs_out,

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
    }])

    manifest_path = som_dir / "somnotate_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print()
    print("Prepared recording for Somnotate.")
    print("Recording:", recording_id)
    print("EDF:", edf_path)
    print("Manifest:", manifest_path)
    print("Sampling rate:", fs_out)

    if manual_for_somnotate:
        print("Manual annotation:", manual_for_somnotate)

    return manifest_path


def combine_manifests(project_root: Path, recording_ids, out_path: Path):
    rows = []

    for rec_id in recording_ids:
        manifest_path = project_root / "recordings" / rec_id / "somnotate" / "somnotate_manifest.csv"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Somnotate manifest missing for {rec_id}: {manifest_path}"
            )

        df = pd.read_csv(manifest_path)
        rows.append(df.iloc[0].to_dict())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    return out_path


# =============================================================================
# RUN SOMNOTATE
# =============================================================================

def find_somnotate_pipeline_dir(somnotate_root: Path):
    """
    Accept common Somnotate paths.

    The user may provide:
    - /path/to/somnotate
    - /path/to/somnotate/example_pipeline
    - any folder inside a cloned Somnotate repo

    The function searches for the folder containing the example pipeline scripts.
    """
    root = Path(somnotate_root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Somnotate path does not exist: {root}")

    candidates = []

    # Case 1: user directly selected example_pipeline.
    candidates.append(root)

    # Case 2: user selected main cloned repo.
    candidates.append(root / "example_pipeline")

    # Case 3: user selected some parent folder; search for example_pipeline.
    for d in root.rglob("example_pipeline"):
        if d.is_dir():
            candidates.append(d)

    # Case 4: search for the preprocessing script and use its parent folder.
    for f in root.rglob("01_preprocess_signals.py"):
        if f.is_file():
            candidates.append(f.parent)

    # Remove duplicates while preserving order.
    unique = []
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
        has_preprocess = (c / "01_preprocess_signals.py").exists()
        has_score = (c / "04_run_state_annotation.py").exists()
        has_config = (c / "configuration.py").exists()
        has_data_io = (c / "data_io.py").exists()

        if has_preprocess and has_score and has_config and has_data_io:
            return c

    checked = "\n".join(str(c) for c in unique[:20])

    raise FileNotFoundError(
        "Could not locate Somnotate example_pipeline scripts.\n\n"
        f"Input path was: {root}\n\n"
        "Expected a folder containing at least:\n"
        "  01_preprocess_signals.py\n"
        "  03_train_state_annotation.py\n"
        "  04_run_state_annotation.py\n"
        "  07_compute_state_probabilities.py\n"
        "  configuration.py\n"
        "  data_io.py\n\n"
        "Checked candidate folders:\n"
        f"{checked}"
    )


def find_script_by_name_or_content(pipeline_dir: Path, exact_names, required_text=None):
    """
    First try expected file names. If those are missing, search all .py files
    by content. This makes the wrapper more robust to small Somnotate changes.
    """
    pipeline_dir = Path(pipeline_dir)

    for name in exact_names:
        p = pipeline_dir / name
        if p.exists():
            return p

    if required_text:
        if isinstance(required_text, str):
            required_text = [required_text]

        for p in sorted(pipeline_dir.glob("*.py")):
            try:
                txt = p.read_text(errors="ignore")
            except Exception:
                continue

            if all(marker in txt for marker in required_text):
                return p

    raise FileNotFoundError(
        f"Could not find script in {pipeline_dir}.\n"
        f"Tried exact names: {exact_names}\n"
        f"Required text markers: {required_text}"
    )


def somnotate_paths(somnotate_root: Path):
    """
    Resolve all Somnotate example pipeline scripts from either the main repo path
    or the example_pipeline path.
    """
    pipeline_dir = find_somnotate_pipeline_dir(somnotate_root)

    preprocess_script = find_script_by_name_or_content(
        pipeline_dir,
        ["01_preprocess_signals.py"],
        required_text=["load_raw_signals", "file_path_preprocessed_signals"],
    )

    train_script = find_script_by_name_or_content(
        pipeline_dir,
        ["03_train_state_annotation.py"],
        required_text=["StateAnnotator", "train"],
    )

    score_script = find_script_by_name_or_content(
        pipeline_dir,
        ["04_run_state_annotation.py"],
        required_text=["StateAnnotator", "file_path_automated_state_annotation"],
    )

    probability_script = find_script_by_name_or_content(
        pipeline_dir,
        ["07_compute_state_probabilities.py"],
        required_text=["predict_all_probabilities", "file_path_state_probabilities"],
    )

    print()
    print("Resolved Somnotate example pipeline:")
    print(pipeline_dir)
    print("Preprocess script:", preprocess_script.name)
    print("Train script:     ", train_script.name)
    print("Score script:     ", score_script.name)
    print("Prob script:      ", probability_script.name)

    return pipeline_dir, preprocess_script, train_script, score_script, probability_script


def preprocess_manifest(py, somnotate_root: Path, manifest_path: Path):
    pipeline_dir, preprocess_script, _, _, _ = somnotate_paths(somnotate_root)

    run_step(
        [py, str(preprocess_script), str(manifest_path)],
        "Somnotate preprocessing",
        cwd=pipeline_dir,
    )


def train_model(py, somnotate_root: Path, training_manifest: Path, model_file: Path):
    pipeline_dir, _, train_script, _, _ = somnotate_paths(somnotate_root)

    model_file.parent.mkdir(parents=True, exist_ok=True)

    run_step(
        [py, str(train_script), str(training_manifest), str(model_file)],
        "Somnotate model training",
        cwd=pipeline_dir,
    )


def score_manifest(py, somnotate_root: Path, manifest_path: Path, model_file: Path):
    pipeline_dir, _, _, score_script, _ = somnotate_paths(somnotate_root)

    if not model_file.exists():
        raise FileNotFoundError(model_file)

    run_step(
        [py, str(score_script), str(manifest_path), str(model_file)],
        "Somnotate state annotation",
        cwd=pipeline_dir,
    )


def probabilities_manifest(py, somnotate_root: Path, manifest_path: Path, model_file: Path):
    pipeline_dir, _, _, _, probability_script = somnotate_paths(somnotate_root)

    if not model_file.exists():
        raise FileNotFoundError(model_file)

    run_step(
        [py, str(probability_script), str(manifest_path), str(model_file)],
        "Somnotate probability computation",
        cwd=pipeline_dir,
    )



# =============================================================================
# IMPORT SOMNOTATE RESULTS INTO THE APP
# =============================================================================

def parse_somnotate_automated(path: Path):
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
            rows.append({
                "t0_s": prev_end,
                "t1_s": end_s,
                "somnotate_state": state,
            })
            prev_end = end_s

    return pd.DataFrame(rows)


def state_at_times(intervals, t_mid):
    out = []

    for t in t_mid:
        hit = intervals[(intervals["t0_s"] <= t) & (intervals["t1_s"] > t)]

        if len(hit):
            out.append(hit.iloc[0]["somnotate_state"])
        else:
            out.append("Undefined")

    return np.array(out, dtype=object)


def import_one_recording(project_root: Path, recording_id: str):
    rec_dir = project_root / "recordings" / recording_id
    som_dir = rec_dir / "somnotate"

    metadata = read_metadata(rec_dir)
    duration_s = float(metadata["duration_s"])

    auto_file = som_dir / "somnotate_automated.tsv"
    prob_file = som_dir / "somnotate_state_probabilities.npz"

    if not auto_file.exists() and not prob_file.exists():
        raise FileNotFoundError(
            f"No Somnotate outputs found for {recording_id}. "
            f"Expected {auto_file} and/or {prob_file}"
        )

    n = int(np.ceil(duration_s))
    t0 = np.arange(n, dtype=float)
    t1 = t0 + 1.0
    t_mid = (t0 + t1) / 2

    out = pd.DataFrame({
        "recording_id": recording_id,
        "t0_s": t0,
        "t1_s": t1,
        "time_min": t0 / 60,
    })

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
        arrays = [np.asarray(z[k], dtype=float) for k in state_names]

        m = min(len(a) for a in arrays)
        prob_t = np.linspace(0, duration_s, m, endpoint=False)

        for state_name, arr in zip(state_names, arrays):
            clean = normalize_state(state_name)
            out[f"somnotate_P_{clean}"] = np.interp(
                out["t0_s"].to_numpy(dtype=float),
                prob_t,
                arr[:m],
            )

        prob_cols = [c for c in out.columns if c.startswith("somnotate_P_")]

        if prob_cols:
            out["somnotate_confidence"] = out[prob_cols].max(axis=1)
            out["somnotate_uncertainty"] = 1.0 - out["somnotate_confidence"]

    out_path = som_dir / "somnotate_results_timeseries.csv"
    out.to_csv(out_path, index=False)

    print()
    print("Imported Somnotate results.")
    print("Recording:", recording_id)
    print("Output:", out_path)
    print()
    print(out["somnotate_state"].value_counts(normalize=True).mul(100).round(2).to_string())

    prob_cols = [c for c in out.columns if c.startswith("somnotate_P_")]
    if prob_cols:
        print()
        print("Probability columns:", ", ".join(prob_cols))

    return out_path


# =============================================================================
# HIGH-LEVEL WORKFLOWS
# =============================================================================

def workflow_use_existing_model(args):
    project_root = Path(args.project_root).expanduser().resolve()
    somnotate_root = Path(args.somnotate_root).expanduser().resolve()
    py = resolve_python(args.somnotate_python, args.somnotate_conda_env)
    model_file = Path(args.model_file).expanduser().resolve()

    recording_ids = split_ids(args.recording_ids)

    if not recording_ids:
        raise ValueError("No recording IDs provided.")

    if args.prepare:
        for rec_id in recording_ids:
            prepare_one_recording(project_root, rec_id, args.target_fs)

    manifest_path = project_root / "somnotate_runs" / f"use_existing_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    combine_manifests(project_root, recording_ids, manifest_path)

    if args.preprocess:
        preprocess_manifest(py, somnotate_root, manifest_path)

    if args.score:
        score_manifest(py, somnotate_root, manifest_path, model_file)

    if args.probabilities:
        probabilities_manifest(py, somnotate_root, manifest_path, model_file)

    if args.import_results:
        for rec_id in recording_ids:
            import_one_recording(project_root, rec_id)


def workflow_train_model(args):
    project_root = Path(args.project_root).expanduser().resolve()
    somnotate_root = Path(args.somnotate_root).expanduser().resolve()
    py = resolve_python(args.somnotate_python, args.somnotate_conda_env)

    train_ids = split_ids(args.train_recording_ids)
    test_ids = split_ids(args.test_recording_ids)

    if not train_ids:
        raise ValueError("No training recording IDs provided.")

    all_ids = train_ids + test_ids

    if args.prepare:
        for rec_id in all_ids:
            prepare_one_recording(project_root, rec_id, args.target_fs)

    for rec_id in train_ids:
        manifest_path = project_root / "recordings" / rec_id / "somnotate" / "somnotate_manifest.csv"
        df = pd.read_csv(manifest_path)

        manual_path = str(df.iloc[0].get("file_path_manual_state_annotation", ""))

        if not manual_path or manual_path == "nan" or not Path(manual_path).exists():
            raise FileNotFoundError(
                f"Training recording {rec_id} has no manual annotation for Somnotate. "
                "Import a .mat file with manual scoring first."
            )

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.model_name.strip().replace(" ", "_").replace("/", "_")

    train_manifest = project_root / "somnotate_runs" / f"{safe_name}_training_manifest_{tag}.csv"
    model_file = project_root / "somnotate_models" / f"{safe_name}_{tag}.pickle"

    combine_manifests(project_root, train_ids, train_manifest)

    if args.preprocess:
        preprocess_manifest(py, somnotate_root, train_manifest)

    train_model(py, somnotate_root, train_manifest, model_file)

    print()
    print("New model saved here:")
    print(model_file)

    if test_ids:
        test_manifest = project_root / "somnotate_runs" / f"{safe_name}_test_manifest_{tag}.csv"
        combine_manifests(project_root, test_ids, test_manifest)

        if args.preprocess:
            preprocess_manifest(py, somnotate_root, test_manifest)

        score_manifest(py, somnotate_root, test_manifest, model_file)
        probabilities_manifest(py, somnotate_root, test_manifest, model_file)

        for rec_id in test_ids:
            import_one_recording(project_root, rec_id)


def workflow_attach_outputs(args):
    project_root = Path(args.project_root).expanduser().resolve()
    rec_dir = project_root / "recordings" / args.recording_id
    som_dir = rec_dir / "somnotate"
    som_dir.mkdir(parents=True, exist_ok=True)

    auto_in = Path(args.automated_file).expanduser() if args.automated_file else None
    prob_in = Path(args.probability_file).expanduser() if args.probability_file else None

    if not auto_in and not prob_in:
        raise ValueError("Provide at least --automated-file or --probability-file.")

    if auto_in:
        if not auto_in.exists():
            raise FileNotFoundError(auto_in)

        dst = som_dir / "somnotate_automated.tsv"
        if auto_in.resolve() != dst.resolve():
            shutil.copy2(auto_in, dst)

    if prob_in:
        if not prob_in.exists():
            raise FileNotFoundError(prob_in)

        dst = som_dir / "somnotate_state_probabilities.npz"
        if prob_in.resolve() != dst.resolve():
            shutil.copy2(prob_in, dst)

    import_one_recording(project_root, args.recording_id)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Somnotate layer for Sleep Stage QC v2."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------
    p = sub.add_parser("prepare")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-ids", required=True)
    p.add_argument("--target-fs", type=float, default=512.0)

    # ------------------------------------------------------------------
    # Use existing model
    # ------------------------------------------------------------------
    p = sub.add_parser("use-existing-model")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-ids", required=True)
    p.add_argument("--somnotate-root", required=True)
    p.add_argument("--somnotate-python", default="")
    p.add_argument("--somnotate-conda-env", default="somnotate_env")
    p.add_argument("--model-file", required=True)
    p.add_argument("--target-fs", type=float, default=512.0)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--probabilities", action="store_true")
    p.add_argument("--import-results", action="store_true")

    # ------------------------------------------------------------------
    # Train new model
    # ------------------------------------------------------------------
    p = sub.add_parser("train-model")
    p.add_argument("--project-root", required=True)
    p.add_argument("--train-recording-ids", required=True)
    p.add_argument("--test-recording-ids", default="")
    p.add_argument("--somnotate-root", required=True)
    p.add_argument("--somnotate-python", default="")
    p.add_argument("--somnotate-conda-env", default="somnotate_env")
    p.add_argument("--model-name", required=True)
    p.add_argument("--target-fs", type=float, default=512.0)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--preprocess", action="store_true")

    # ------------------------------------------------------------------
    # Attach already-created outputs
    # ------------------------------------------------------------------
    p = sub.add_parser("attach-outputs")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-id", required=True)
    p.add_argument("--automated-file", default="")
    p.add_argument("--probability-file", default="")

    # ------------------------------------------------------------------
    # Import existing local somnotate folder outputs
    # ------------------------------------------------------------------
    p = sub.add_parser("import-results")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-ids", required=True)

    args = parser.parse_args()

    if args.command == "prepare":
        project_root = Path(args.project_root).expanduser().resolve()
        for rec_id in split_ids(args.recording_ids):
            prepare_one_recording(project_root, rec_id, args.target_fs)

    elif args.command == "use-existing-model":
        workflow_use_existing_model(args)

    elif args.command == "train-model":
        workflow_train_model(args)

    elif args.command == "attach-outputs":
        workflow_attach_outputs(args)

    elif args.command == "import-results":
        project_root = Path(args.project_root).expanduser().resolve()
        for rec_id in split_ids(args.recording_ids):
            import_one_recording(project_root, rec_id)


if __name__ == "__main__":
    main()
