#!/usr/bin/env python3
"""Import an EDF/BDF recording into the sleep scoring QC app project format.

This script mirrors the output structure expected by the Dash app:

project_root/
├── recordings_manifest.csv
└── recordings/<recording_id>/
    ├── metadata.json
    ├── eeg.npy
    ├── emg.npy
    ├── ach.npy                         # optional
    └── manual_scoring_aligned.csv       # optional, if EDF annotations map to states

The downstream app steps can then run unchanged:
    2. Compute epoch features
    3. Run Layer 1 Wake/Sleep

Examples
--------
python pipelines/01_import_edf_recording.py \
  --edf-file /path/to/recording.edf \
  --project-root /path/to/project \
  --recording-id mouse01_baseline \
  --eeg-channel EEG \
  --emg-channel EMG \
  --epoch-sec 1

Channel arguments can be either an exact label, a case-insensitive substring,
or a zero-based channel index.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

try:
    import pyedflib
except Exception as exc:  # pragma: no cover - handled at runtime
    pyedflib = None
    _PYEDFLIB_IMPORT_ERROR = exc
else:
    _PYEDFLIB_IMPORT_ERROR = None


KNOWN_STATE_LABELS = {
    "wake": "Wake",
    "w": "Wake",
    "awake": "Wake",
    "nrem": "NREM",
    "non-rem": "NREM",
    "non rem": "NREM",
    "n": "NREM",
    "sws": "NREM",
    "sleep": "NREM",
    "rem": "REM",
    "r": "REM",
    "paradoxical": "REM",
    "undefined": "Undefined",
    "uncertain": "Undefined",
    "artifact": "Artifact",
    "artefact": "Artifact",
}


FINAL_CODE = {
    "Wake": 0,
    "NREM": 1,
    "REM": 2,
    "Undefined": -1,
    "Uncertain": -1,
    "Artifact": -2,
}


def _load_json_map(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    try:
        raw = json.loads(text)
    except Exception:
        return {}
    return {str(k): str(v) for k, v in dict(raw).items()}


def _normalise_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def resolve_channel(labels: list[str], requested: str | None, *, kind: str, required: bool = True) -> int | None:
    """Resolve a channel by index, exact label, or substring."""
    requested = str(requested or "").strip()

    if requested:
        if requested.isdigit():
            idx = int(requested)
            if 0 <= idx < len(labels):
                return idx
            raise ValueError(f"{kind} channel index {idx} is outside available channels 0..{len(labels)-1}.")

        wanted = _normalise_for_match(requested)
        normalised = [_normalise_for_match(x) for x in labels]

        # Exact case-insensitive label match first.
        for i, lab in enumerate(normalised):
            if lab == wanted:
                return i

        # Then substring match in either direction.
        candidates = [i for i, lab in enumerate(normalised) if wanted in lab or lab in wanted]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            pretty = ", ".join([f"{i}:{labels[i]}" for i in candidates])
            raise ValueError(f"{kind} channel '{requested}' is ambiguous. Matching channels: {pretty}")

        if required:
            pretty = ", ".join([f"{i}:{x}" for i, x in enumerate(labels)])
            raise ValueError(f"Could not find {kind} channel '{requested}'. Available channels: {pretty}")
        return None

    # No requested channel: make a conservative guess.
    kind_lower = kind.lower()
    guess_terms = {
        "eeg": ["eeg", "frontal", "cortical"],
        "emg": ["emg", "muscle"],
        "ach": ["ach", "465", "photometry", "fiber", "ne"],
    }.get(kind_lower, [kind_lower])

    normalised = [_normalise_for_match(x) for x in labels]
    for term in guess_terms:
        for i, lab in enumerate(normalised):
            if term in lab:
                return i

    if required:
        pretty = ", ".join([f"{i}:{x}" for i, x in enumerate(labels)])
        raise ValueError(f"Could not guess {kind} channel. Available channels: {pretty}")
    return None


def read_signal(reader: Any, channel_index: int) -> tuple[np.ndarray, float, str]:
    labels = list(reader.getSignalLabels())
    fs_values = np.asarray(reader.getSampleFrequencies(), dtype=float)
    fs = float(fs_values[channel_index])
    y = np.asarray(reader.readSignal(channel_index), dtype=np.float32)
    y = np.ravel(y)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return y, fs, str(labels[channel_index])


def resample_to_fs(y: np.ndarray, src_fs: float, target_fs: float) -> np.ndarray:
    if abs(float(src_fs) - float(target_fs)) < 1e-6:
        return np.asarray(y, dtype=np.float32)

    ratio = Fraction(float(target_fs) / float(src_fs)).limit_denominator(1000)
    out = resample_poly(np.asarray(y, dtype=np.float32), ratio.numerator, ratio.denominator)
    return np.asarray(out, dtype=np.float32)


def normalise_state_label(raw_label: str, annotation_map: dict[str, str]) -> str | None:
    text = str(raw_label).strip()
    if not text:
        return None

    # Direct user map first. This lets Sirenia-specific labels/codes be handled.
    if text in annotation_map:
        return annotation_map[text]

    lower = _normalise_for_match(text)
    if lower in annotation_map:
        return annotation_map[lower]

    # Try numeric code extraction, e.g. "Stage 1" or "Sleep stage: 2".
    numeric = re.fullmatch(r"[-+]?\d+(?:\.0+)?", lower)
    if numeric and lower in annotation_map:
        return annotation_map[lower]

    # Common direct labels.
    if lower in KNOWN_STATE_LABELS:
        return KNOWN_STATE_LABELS[lower]

    # Common annotation strings with state words embedded.
    for token, state in [
        ("wake", "Wake"),
        ("nrem", "NREM"),
        ("non-rem", "NREM"),
        ("non rem", "NREM"),
        ("sws", "NREM"),
        ("rem", "REM"),
        ("artifact", "Artifact"),
        ("artefact", "Artifact"),
    ]:
        if token in lower:
            if token == "rem" and ("nrem" in lower or "non-rem" in lower or "non rem" in lower):
                continue
            return state

    return None


def extract_manual_scoring_from_annotations(
    reader: Any,
    recording_id: str,
    epoch_sec: float,
    annotation_map: dict[str, str],
) -> pd.DataFrame | None:
    try:
        onsets, durations, descriptions = reader.readAnnotations()
    except Exception:
        return None

    if len(descriptions) == 0:
        return None

    rows: list[dict[str, Any]] = []
    for onset, duration, desc in zip(onsets, durations, descriptions):
        state = normalise_state_label(str(desc), annotation_map)
        if state is None:
            continue
        dur = float(duration) if np.isfinite(duration) and float(duration) > 0 else float(epoch_sec)
        t0 = max(0.0, float(onset))
        t1 = max(t0 + 1e-6, t0 + dur)
        rows.append(
            {
                "recording_id": recording_id,
                "epoch_id": len(rows),
                "t0_s": t0,
                "t1_s": t1,
                "manual_state": state,
                "manual_code": FINAL_CODE.get(state, -1),
                "source_annotation": str(desc),
            }
        )

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values(["t0_s", "t1_s"]).reset_index(drop=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def update_manifest(project_root: Path, row: dict[str, Any]) -> None:
    manifest_path = project_root / "recordings_manifest.csv"
    row_df = pd.DataFrame([row])

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if "recording_id" in manifest.columns:
            manifest = manifest[manifest["recording_id"].astype(str) != str(row["recording_id"])]
        manifest = pd.concat([manifest, row_df], ignore_index=True, sort=False)
    else:
        manifest = row_df

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)


def import_edf(args: argparse.Namespace) -> None:
    if pyedflib is None:
        raise RuntimeError(
            "pyedflib is required for EDF/BDF import but could not be imported. "
            f"Original import error: {_PYEDFLIB_IMPORT_ERROR!r}"
        )

    edf_path = Path(args.edf_file).expanduser().resolve()
    if not edf_path.exists():
        raise FileNotFoundError(edf_path)

    project_root = Path(args.project_root).expanduser().resolve()
    recording_dir = project_root / "recordings" / str(args.recording_id)
    recording_dir.mkdir(parents=True, exist_ok=True)

    annotation_map = _load_json_map(args.annotation_map)

    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = list(reader.getSignalLabels())
        eeg_idx = resolve_channel(labels, args.eeg_channel, kind="EEG", required=True)
        emg_idx = resolve_channel(labels, args.emg_channel, kind="EMG", required=True)
        ach_idx = resolve_channel(labels, args.ach_channel, kind="ACh", required=False) if args.ach_channel else None

        eeg, eeg_fs, eeg_label = read_signal(reader, int(eeg_idx))
        emg, emg_fs, emg_label = read_signal(reader, int(emg_idx))

        target_fs = float(eeg_fs)
        emg = resample_to_fs(emg, emg_fs, target_fs)

        n = min(len(eeg), len(emg))
        eeg = np.asarray(eeg[:n], dtype=np.float32)
        emg = np.asarray(emg[:n], dtype=np.float32)
        duration_s = float(n / target_fs)

        np.save(recording_dir / "eeg.npy", eeg)
        np.save(recording_dir / "emg.npy", emg)

        metadata: dict[str, Any] = {
            "recording_id": str(args.recording_id),
            "source_file": str(edf_path),
            "source_format": edf_path.suffix.lower().lstrip("."),
            "sampling_rate_hz": target_fs,
            "duration_s": duration_s,
            "epoch_sec": float(args.epoch_sec),
            "eeg_file": "eeg.npy",
            "emg_file": "emg.npy",
            "eeg_channel": eeg_label,
            "emg_channel": emg_label,
            "eeg_original_sampling_rate_hz": float(eeg_fs),
            "emg_original_sampling_rate_hz": float(emg_fs),
            "imported_at": datetime.now().isoformat(timespec="seconds"),
            "mouse_id": str(args.mouse_id or ""),
            "group": str(args.group or ""),
            "condition": str(args.condition or ""),
            "week": str(args.week or ""),
        }

        if ach_idx is not None:
            try:
                ach, ach_fs, ach_label = read_signal(reader, int(ach_idx))
                np.save(recording_dir / "ach.npy", np.asarray(ach, dtype=np.float32))
                metadata.update(
                    {
                        "ach_file": "ach.npy",
                        "ach_sampling_rate_hz": float(ach_fs),
                        "ach_channel": ach_label,
                    }
                )
            except Exception as exc:
                print(f"WARNING: could not import optional ACh/photometry channel: {exc}")

        manual = extract_manual_scoring_from_annotations(
            reader=reader,
            recording_id=str(args.recording_id),
            epoch_sec=float(args.epoch_sec),
            annotation_map=annotation_map,
        )
        if manual is not None and len(manual):
            manual.to_csv(recording_dir / "manual_scoring_aligned.csv", index=False)
            metadata["manual_scoring_file"] = "manual_scoring_aligned.csv"
            metadata["manual_scoring_source"] = "edf_annotations"
            metadata["manual_scoring_rows"] = int(len(manual))

        write_json(recording_dir / "metadata.json", metadata)

        update_manifest(
            project_root,
            {
                "recording_id": str(args.recording_id),
                "recording_dir": str(recording_dir),
                "source_file": str(edf_path),
                "source_format": metadata["source_format"],
                "sampling_rate_hz": target_fs,
                "duration_s": duration_s,
                "mouse_id": str(args.mouse_id or ""),
                "group": str(args.group or ""),
                "condition": str(args.condition or ""),
                "week": str(args.week or ""),
                "imported_at": metadata["imported_at"],
            },
        )

        print(f"Imported EDF/BDF recording: {args.recording_id}")
        print(f"  Source: {edf_path}")
        print(f"  Output: {recording_dir}")
        print(f"  EEG: {eeg_label} at {eeg_fs:g} Hz -> eeg.npy")
        print(f"  EMG: {emg_label} at {emg_fs:g} Hz -> emg.npy at {target_fs:g} Hz")
        if ach_idx is not None and "ach_file" in metadata:
            print(f"  ACh/photometry: {metadata['ach_channel']} at {metadata['ach_sampling_rate_hz']:g} Hz -> ach.npy")
        if manual is not None and len(manual):
            print(f"  Manual scoring annotations imported: {len(manual)} rows")
        else:
            print("  No EDF annotations were imported as manual scoring.")
        print("Next steps in the app: Compute epoch features -> Run Layer 1 Wake/Sleep.")

    finally:
        try:
            reader.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import EDF/BDF into the sleep scoring QC app project format.")
    parser.add_argument("--edf-file", required=True, help="Full path to .edf or .bdf recording.")
    parser.add_argument("--project-root", required=True, help="Project root folder used by the app.")
    parser.add_argument("--recording-id", required=True, help="Recording ID to create/update.")
    parser.add_argument("--eeg-channel", default="eeg", help="EEG channel label, substring, or zero-based index.")
    parser.add_argument("--emg-channel", default="emg", help="EMG channel label, substring, or zero-based index.")
    parser.add_argument("--ach-channel", default="", help="Optional ACh/photometry channel label, substring, or zero-based index.")
    parser.add_argument("--epoch-sec", type=float, default=1.0, help="Epoch length used downstream by features/Layer 1.")
    parser.add_argument("--annotation-map", default="{}", help="JSON mapping from EDF annotation text/code to Wake/NREM/REM/etc.")
    parser.add_argument("--mouse-id", default="")
    parser.add_argument("--group", default="")
    parser.add_argument("--condition", default="")
    parser.add_argument("--week", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import_edf(args)


if __name__ == "__main__":
    main()
