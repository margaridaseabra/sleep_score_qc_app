from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def clean_mat_dict(d):
    return {k: v for k, v in d.items() if not k.startswith("__")}


def squeeze_1d(x, name="signal"):
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Expected 1D {name} after squeeze, got shape {x.shape}")
    return x.astype(np.float32)


def load_mat_any(path: Path):
    try:
        return clean_mat_dict(loadmat(path))
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if hasattr(obj, "shape"):
                    out[name] = np.array(obj).squeeze()
            f.visititems(visit)
        return out


def scalar_from_key(mat, key):
    if not key:
        return None
    if key not in mat:
        return None
    val = np.asarray(mat[key]).squeeze()
    if val.size != 1:
        return None
    return float(val)


def infer_fs(mat, user_fs=None, fs_key=None):
    if fs_key:
        val = scalar_from_key(mat, fs_key)
        if val is not None and val > 0:
            return float(val)

    if user_fs is not None and float(user_fs) > 0:
        return float(user_fs)

    for key in [
        "fs",
        "Fs",
        "sampling_rate",
        "sampling_frequency",
        "srate",
        "eeg_frequency",
    ]:
        val = scalar_from_key(mat, key)
        if val is not None and val > 0:
            return float(val)

    raise ValueError(
        "Sampling frequency not provided and could not be inferred from .mat. "
        "Use --fs or --eeg-fs-key."
    )


def make_manual_scoring(scoring, epoch_sec, code_map, recording_id):
    scoring = np.asarray(scoring).squeeze()

    if scoring.ndim != 1:
        raise ValueError(f"Scoring must be 1D after squeeze, got {scoring.shape}")

    rows = []

    for i, code in enumerate(scoring):
        try:
            raw_code = str(int(code)) if float(code).is_integer() else str(code)
        except Exception:
            raw_code = str(code)

        label = code_map.get(raw_code, code_map.get(str(code), "Undefined"))

        rows.append({
            "recording_id": recording_id,
            "epoch_id": i,
            "t0_s": i * epoch_sec,
            "t1_s": (i + 1) * epoch_sec,
            "manual_code": raw_code,
            "manual_state": label,
            "manual_wake_sleep": (
                "Wake"
                if label == "Wake"
                else ("Sleep" if label in ["NREM", "REM", "Sleep"] else "Uncertain")
            ),
        })

    return pd.DataFrame(rows)


def update_manifest(project_root, row):
    manifest_path = project_root / "recordings_manifest.csv"

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["recording_id"].astype(str) != str(row["recording_id"])]
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    else:
        manifest = pd.DataFrame([row])

    manifest.to_csv(manifest_path, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mat-file", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--recording-id", required=True)

    parser.add_argument("--eeg-key", required=True)
    parser.add_argument("--emg-key", required=True)
    parser.add_argument("--ach-key", default="")
    parser.add_argument("--scoring-key", default="")

    parser.add_argument("--fs", type=float, default=None)
    parser.add_argument("--eeg-fs-key", default="")
    parser.add_argument("--ach-fs-key", default="")
    parser.add_argument("--epoch-sec", type=float, default=1.0)

    parser.add_argument("--mouse-id", default="")
    parser.add_argument("--group", default="")
    parser.add_argument("--condition", default="")
    parser.add_argument("--week", default="")

    parser.add_argument(
        "--code-map",
        default='{"0":"Wake","1":"NREM","2":"REM","15":"Wake","-1":"Undefined"}',
    )

    args = parser.parse_args()

    mat_file = Path(args.mat_file).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()

    if not mat_file.exists():
        raise FileNotFoundError(mat_file)

    mat = load_mat_any(mat_file)

    print("Available .mat variables:")
    for k, v in mat.items():
        try:
            print(f"  {k}: shape={np.asarray(v).shape}")
        except Exception:
            print(f"  {k}")

    if args.eeg_key not in mat:
        raise KeyError(f"EEG key not found: {args.eeg_key}")

    if args.emg_key not in mat:
        raise KeyError(f"EMG key not found: {args.emg_key}")

    eeg = squeeze_1d(mat[args.eeg_key], name=args.eeg_key)
    emg = squeeze_1d(mat[args.emg_key], name=args.emg_key)

    eeg_fs = infer_fs(mat, user_fs=args.fs, fs_key=args.eeg_fs_key)
    emg_fs = eeg_fs

    n = min(len(eeg), len(emg))
    eeg = eeg[:n]
    emg = emg[:n]

    duration_s = n / eeg_fs

    rec_dir = project_root / "recordings" / args.recording_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    eeg_path = rec_dir / "eeg.npy"
    emg_path = rec_dir / "emg.npy"
    meta_path = rec_dir / "metadata.json"

    np.save(eeg_path, eeg)
    np.save(emg_path, emg)

    ach_path = ""
    ach_fs = None
    ach_key_used = ""

    if args.ach_key:
        if args.ach_key in mat:
            ach = squeeze_1d(mat[args.ach_key], name=args.ach_key)

            # ACh can have a different sampling rate and length from EEG/EMG.
            ach_fs = infer_fs(
                mat,
                user_fs=eeg_fs,
                fs_key=args.ach_fs_key,
            )

            ach_path_obj = rec_dir / "ach.npy"
            np.save(ach_path_obj, ach)

            ach_path = str(ach_path_obj)
            ach_key_used = args.ach_key

            print(f"ACh/photometry saved from key '{args.ach_key}':", ach_path_obj)
            print("ACh sampling rate:", ach_fs)

        else:
            print(f"Warning: ACh key requested but not found: {args.ach_key}")

    metadata = {
        "recording_id": args.recording_id,
        "source_mat_file": str(mat_file),
        "mouse_id": args.mouse_id,
        "group": args.group,
        "condition": args.condition,
        "week": args.week,

        "eeg_key": args.eeg_key,
        "emg_key": args.emg_key,
        "ach_key": ach_key_used,
        "scoring_key": args.scoring_key,

        "eeg_fs_key": args.eeg_fs_key,
        "ach_fs_key": args.ach_fs_key,

        "eeg_file": str(eeg_path),
        "emg_file": str(emg_path),
        "ach_file": ach_path,

        "sampling_rate_hz": eeg_fs,
        "eeg_sampling_rate_hz": eeg_fs,
        "emg_sampling_rate_hz": emg_fs,

        "ach_sampling_rate_hz": ach_fs,
        "photometry_sampling_rate_hz": ach_fs,
        "ne_frequency": ach_fs,

        "n_samples": int(n),
        "duration_s": float(duration_s),
    }

    meta_path.write_text(json.dumps(metadata, indent=2))

    manual_file = ""

    if args.scoring_key:
        if args.scoring_key not in mat:
            raise KeyError(f"Scoring key not found: {args.scoring_key}")

        code_map = json.loads(args.code_map)

        manual = make_manual_scoring(
            scoring=mat[args.scoring_key],
            epoch_sec=args.epoch_sec,
            code_map=code_map,
            recording_id=args.recording_id,
        )

        manual_path = rec_dir / "manual_scoring_aligned.csv"
        manual.to_csv(manual_path, index=False)
        manual_file = str(manual_path)

    manifest_row = {
        "recording_id": args.recording_id,
        "recording_dir": str(rec_dir),
        "source_mat_file": str(mat_file),
        "mouse_id": args.mouse_id,
        "group": args.group,
        "condition": args.condition,
        "week": args.week,
        "duration_s": float(duration_s),
        "sampling_rate_hz": eeg_fs,
        "eeg_sampling_rate_hz": eeg_fs,
        "emg_sampling_rate_hz": emg_fs,
        "ach_sampling_rate_hz": ach_fs,
        "ach_file": ach_path,
        "manual_scoring_file": manual_file,
        "preprocessing_done": True,
    }

    update_manifest(project_root, manifest_row)

    print()
    print("Imported .mat recording.")
    print("Recording folder:", rec_dir)
    print("EEG:", eeg_path)
    print("EMG:", emg_path)
    print("Metadata:", meta_path)

    if ach_path:
        print("ACh:", ach_path)

    if manual_file:
        print("Manual scoring:", manual_file)


if __name__ == "__main__":
    main()
