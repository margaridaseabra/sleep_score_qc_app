from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


def resample_if_needed(x, fs_in, fs_out):
    x = np.asarray(x, dtype=np.float32)

    if fs_out <= 0 or abs(fs_in - fs_out) < 1e-6:
        return x, fs_in

    frac = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
    y = resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)
    return y, float(fs_out)


def write_edf(edf_path, eeg, emg, fs):
    try:
        import pyedflib
    except ImportError as e:
        raise ImportError(
            "pyedflib is required to write EDF files. Install it in your app environment:\n"
            "python -m pip install pyedflib"
        ) from e

    edf_path = Path(edf_path)
    edf_path.parent.mkdir(parents=True, exist_ok=True)

    signals = [np.asarray(eeg, dtype=np.float64), np.asarray(emg, dtype=np.float64)]
    labels = ["EEG", "EMG"]

    signal_headers = []

    for label, sig in zip(labels, signals):
        physical_min = float(np.nanmin(sig))
        physical_max = float(np.nanmax(sig))

        if physical_min == physical_max:
            physical_min -= 1
            physical_max += 1

        signal_headers.append({
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

    with pyedflib.EdfWriter(str(edf_path), n_channels=2, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setSignalHeaders(signal_headers)
        f.writeSamples(signals)


def export_manual_for_somnotate(manual_csv, out_path):
    manual = pd.read_csv(manual_csv)

    if "manual_state" not in manual.columns:
        return ""

    manual = manual.sort_values("t0_s").reset_index(drop=True)

    rows = []
    current_state = None
    current_end = None

    for _, r in manual.iterrows():
        state = str(r["manual_state"])

        if state == "Undefined" or state == "nan":
            state = "Undefined"

        if current_state is None:
            current_state = state
            current_end = float(r["t1_s"])
            continue

        if state == current_state:
            current_end = float(r["t1_s"])
        else:
            rows.append((current_state, current_end))
            current_state = state
            current_end = float(r["t1_s"])

    if current_state is not None:
        rows.append((current_state, current_end))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for state, end_s in rows:
            f.write(f"{state}\t{end_s:.6f}\n")

    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--target-fs", type=float, default=512.0)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    rec_dir = project_root / "recordings" / args.recording_id
    som_dir = rec_dir / "somnotate"
    som_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = rec_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(metadata_path.read_text())

    fs = float(metadata["sampling_rate_hz"])
    eeg = np.load(rec_dir / "eeg.npy", mmap_mode="r")
    emg = np.load(rec_dir / "emg.npy", mmap_mode="r")

    n = min(len(eeg), len(emg))
    eeg = np.asarray(eeg[:n], dtype=np.float32)
    emg = np.asarray(emg[:n], dtype=np.float32)

    eeg_rs, fs_out = resample_if_needed(eeg, fs, args.target_fs)
    emg_rs, fs_out = resample_if_needed(emg, fs, args.target_fs)

    n2 = min(len(eeg_rs), len(emg_rs))
    eeg_rs = eeg_rs[:n2]
    emg_rs = emg_rs[:n2]

    edf_path = som_dir / "somnotate_input.edf"
    preprocessed_path = som_dir / "somnotate_preprocessed.npy"
    automated_path = som_dir / "somnotate_automated.tsv"
    probabilities_path = som_dir / "somnotate_state_probabilities.npz"
    review_intervals_path = som_dir / "somnotate_review_intervals.tsv"
    manual_som_path = som_dir / "somnotate_manual.tsv"

    write_edf(edf_path, eeg_rs, emg_rs, fs_out)

    manual_csv = rec_dir / "manual_scoring_aligned.csv"
    manual_for_som = ""

    if manual_csv.exists():
        manual_for_som = export_manual_for_somnotate(manual_csv, manual_som_path)

    manifest = pd.DataFrame([{
        "recording_id": args.recording_id,

        # Core Somnotate paths
        "file_path_raw_signals": str(edf_path),
        "file_path_preprocessed_signals": str(preprocessed_path),
        "file_path_automated_state_annotation": str(automated_path),
        "file_path_state_probabilities": str(probabilities_path),
        "file_path_review_intervals": str(review_intervals_path),
        "file_path_manual_state_annotation": manual_for_som,

        # Sampling
        "sampling_frequency_in_hz": fs_out,

        # Channel labels: include several names to be compatible with different configuration.py versions
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
    print("EDF:", edf_path)
    print("Manifest:", manifest_path)
    print("Sampling rate:", fs_out)

    if manual_for_som:
        print("Manual annotation for Somnotate:", manual_for_som)


if __name__ == "__main__":
    main()
