"""Time-aligned, descriptive EEG/EMG/photometry physiology for dissociation QC.

The inputs are existing epoch_features.csv and (optionally) a photometry .npy.
Physiology is *not* used to certify a dissociation or alter scoring labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Already produced by pipelines/02_compute_epoch_features.py.
FEATURE_COLUMNS = (
    "eeg_delta", "eeg_theta", "eeg_sigma", "eeg_broadband",
    "eeg_theta_delta_ratio", "eeg_delta_z", "eeg_theta_z",
    "eeg_sigma_z", "eeg_broadband_z", "eeg_theta_delta_ratio_z",
    "emg_rms", "emg_abs_p95", "emg_abs_p99",
    "emg_rms_z", "emg_abs_p95_z", "emg_abs_p99_z",
)
PROFILE_COLUMNS = (
    "eeg_delta_z", "eeg_theta_delta_ratio_z", "eeg_sigma_z",
    "emg_rms_z", "emg_abs_p95_z", "photometry_mean_z",
)
EVENT_RAW_COLUMNS = (
    "eeg_delta", "eeg_theta", "eeg_sigma", "eeg_theta_delta_ratio",
    "emg_rms", "emg_abs_p95", "photometry_mean_raw", "photometry_std_raw",
)
PROFILE_LABELS = {
    "eeg_delta_z": "EEG delta (0.5–4 Hz), log-power z",
    "eeg_theta_delta_ratio_z": "EEG theta/delta (5–10 / 0.5–4 Hz), log-ratio z",
    "eeg_sigma_z": "EEG sigma (10–15 Hz), log-power z",
    "emg_rms_z": "EMG RMS, log-amplitude z",
    "emg_abs_p95_z": "EMG |amplitude| P95, log-amplitude z",
    "photometry_mean_z": "Photometry epoch mean, robust z",
}


def robust_z(values):
    """Median/MAD normalized across *this recording*, finite values only."""
    a = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(a)
    out = np.full(len(a), np.nan)
    if finite.sum() < 10:
        return out
    med = np.median(a[finite])
    mad = np.median(np.abs(a[finite] - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < 1e-12:
        sigma = np.std(a[finite])
    if not np.isfinite(sigma) or sigma < 1e-12:
        return out  # constant signal: no meaningful standardized variation
    out[finite] = (a[finite] - med) / sigma
    return out


def _z_from_raw(value, *, log_transform=True):
    v = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
    if log_transform:
        v = np.where(np.isfinite(v) & (v > 0), np.log10(np.maximum(v, 1e-12)), np.nan)
    return robust_z(v)


def align_epoch_features(epochs: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Overlap-weighted feature alignment on *time*, not unverified epoch_id.

    This also handles 1s features against a different Layer 1 epoch duration.
    Require >=80% temporal AND finite-feature coverage for each value.
    """
    out = pd.DataFrame(index=epochs.index)
    for key in FEATURE_COLUMNS:
        out[key] = np.nan
    if not {"t0_s", "t1_s"}.issubset(feature_df):
        return out

    src = feature_df.copy()
    for key in ("t0_s", "t1_s", *FEATURE_COLUMNS):
        if key in src:
            src[key] = pd.to_numeric(src[key], errors="coerce")
    src = src[np.isfinite(src["t0_s"]) & np.isfinite(src["t1_s"]) &
              (src["t1_s"] > src["t0_s"])].sort_values("t0_s").reset_index(drop=True)
    if src.empty:
        return out

    starts = src["t0_s"].to_numpy(dtype=float)
    ends = src["t1_s"].to_numpy(dtype=float)
    names = [c for c in FEATURE_COLUMNS if c in src]
    if not names:
        return out
    values = src[names].to_numpy(dtype=float)
    target_start = pd.to_numeric(epochs["t0_s"], errors="coerce").to_numpy(dtype=float)
    target_end = pd.to_numeric(epochs["t1_s"], errors="coerce").to_numpy(dtype=float)
    result = np.full((len(epochs), len(names)), np.nan, dtype=float)
    for i, (left, right) in enumerate(zip(target_start, target_end)):
        if not (np.isfinite(left) and np.isfinite(right) and right > left):
            continue
        lo = int(np.searchsorted(ends, left, side="right"))
        hi = int(np.searchsorted(starts, right, side="left"))
        if lo >= hi:
            continue
        weights = np.minimum(right, ends[lo:hi]) - np.maximum(left, starts[lo:hi])
        weights = np.maximum(0, weights)
        if not weights.sum() or weights.sum() > (right - left) * 1.01:
            # Overlapping source intervals: don't double-count physiology.
            continue
        segment = values[lo:hi]
        valid = np.isfinite(segment)
        coverage = (valid * weights[:, None]).sum(axis=0)
        eligible = coverage >= 0.8 * (right - left)
        if eligible.any():
            weighted = np.where(valid, segment, 0) * weights[:, None]
            numerators = weighted.sum(axis=0)
            result[i, eligible] = numerators[eligible] / coverage[eligible]
    for j, name in enumerate(names):
        out[name] = result[:, j]
    # Work with older feature files with raw metrics but missing robust z.
    for raw, z in (
        ("eeg_delta", "eeg_delta_z"), ("eeg_theta", "eeg_theta_z"),
        ("eeg_sigma", "eeg_sigma_z"), ("eeg_broadband", "eeg_broadband_z"),
        ("eeg_theta_delta_ratio", "eeg_theta_delta_ratio_z"),
        ("emg_rms", "emg_rms_z"), ("emg_abs_p95", "emg_abs_p95_z"),
        ("emg_abs_p99", "emg_abs_p99_z"),
    ):
        if not np.isfinite(out[z]).any() and np.isfinite(out[raw]).any():
            out[z] = _z_from_raw(out[raw])
    return out


def _photometry_location(rec_dir: Path, metadata: dict):
    for key in ("photometry_file", "ach_file", "ne_file"):
        name = metadata.get(key)
        if name and isinstance(name, str):
            path = Path(name).expanduser()
            if not path.is_absolute():
                path = rec_dir / path
            if path.is_file():
                return path
    for name in ("ach.npy", "ne.npy", "photometry.npy", "fiber_photometry.npy"):
        path = rec_dir / name
        if path.is_file():
            return path
    return None


def align_photometry(epochs: pd.DataFrame, rec_dir: Path, metadata: dict):
    """Mean/std of raw auxiliary signal in each epoch; no assumed ΔF/F units."""
    n = len(epochs)
    means = np.full(n, np.nan)
    std = np.full(n, np.nan)
    info = {"available": False, "source": None, "sampling_rate_hz": None,
            "units": "unverified raw units", "status": "No photometry signal found."}

    path = _photometry_location(rec_dir, metadata)
    if path is None:
        return means, std, info
    fs = None
    for key in ("ach_sampling_rate_hz", "photometry_sampling_rate_hz", "ne_frequency"):
        value = metadata.get(key)
        try:
            candidate = float(value)
        except (ValueError, TypeError):
            continue
        if np.isfinite(candidate) and candidate > 0:
            fs = candidate
            break
    if fs is None:
        info["status"] = "Photometry file present, but no explicit photometry sampling rate. Skipped."
        return means, std, info

    # Optional recording-relative signal onset: +10 means photometry starts
    # 10 seconds after EEG t=0. Video offsets are deliberately unrelated.
    try:
        onset_s = float(metadata.get("photometry_offset_s",
                                    metadata.get("ach_offset_s", 0.0)))
    except (ValueError, TypeError):
        info["status"] = "Invalid photometry_offset_s / ach_offset_s. Skipped."
        return means, std, info
    if not np.isfinite(onset_s):
        info["status"] = "Non-finite photometry time offset. Skipped."
        return means, std, info

    try:
        signal = np.load(path, mmap_mode="r", allow_pickle=False)
        if signal.ndim != 1:
            raise ValueError("expected 1D photometry array")
        signal_duration = len(signal) / fs
        reference_duration = float(pd.to_numeric(epochs["t1_s"], errors="coerce").max())
        expected_duration = reference_duration - onset_s
        if (not np.isfinite(reference_duration) or reference_duration <= 0 or
            expected_duration <= 0 or signal_duration <= 0 or
            abs(signal_duration - expected_duration) > max(2, 0.05 * reference_duration)):
            info["status"] = (
                f"Photometry time base unverified: {signal_duration:.1f}s from array/fs "
                f"vs expected {expected_duration:.1f}s (scoring {reference_duration:.1f}s, "
                f"photometry onset {onset_s:+.1f}s). Skipped to prevent misalignment."
            )
            return means, std, info

        start = pd.to_numeric(epochs["t0_s"], errors="coerce").to_numpy(dtype=float)
        end = pd.to_numeric(epochs["t1_s"], errors="coerce").to_numpy(dtype=float)
        # A sample at k/fs belongs to the half-open interval [t0, t1).
        for i, (t0, t1) in enumerate(zip(start, end)):
            if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
                continue
            a = min(len(signal), max(0, int(np.ceil((t0 - onset_s) * fs - 1e-8))))
            b = min(len(signal), max(0, int(np.ceil((t1 - onset_s) * fs - 1e-8))))
            if b <= a:
                continue
            slice_ = np.asarray(signal[a:b], dtype=float)
            finite = slice_[np.isfinite(slice_)]
            if len(finite) < max(1, int(np.ceil((b - a) * 0.8))):
                continue
            means[i] = float(finite.mean())
            std[i] = float(finite.std())

        source = str(metadata.get("ach_key") or metadata.get("ach_channel") or
                     metadata.get("photometry_channel") or path.stem)
        unit = str(metadata.get("photometry_units") or metadata.get("ach_units") or
                   metadata.get("ne_units") or "unverified raw units")
        info.update(available=bool(np.isfinite(means).any()), source=source,
                    sampling_rate_hz=fs, onset_s=onset_s, units=unit, status="aligned")
    except (OSError, ValueError, TypeError) as exc:
        info["status"] = f"Could not safely read photometry: {exc}"
    return means, std, info


def load_epoch_physiology(rec_dir: Path, epochs: pd.DataFrame):
    """Return the epoch table with physiological measurements and provenance."""
    out = epochs.copy()
    info = {"eeg_emg": "epoch_features.csv not found", "photometry": {}}
    for name in FEATURE_COLUMNS:
        out[name] = np.nan
    features_file = rec_dir / "epoch_features.csv"
    if features_file.is_file():
        features = pd.read_csv(features_file)
        aligned = align_epoch_features(out, features)
        for key in FEATURE_COLUMNS:
            out[key] = aligned[key].to_numpy(dtype=float)
        n_eeg = int(np.isfinite(out["eeg_delta"]).sum())
        n_emg = int(np.isfinite(out["emg_rms"]).sum())
        info["eeg_emg"] = f"EEG delta coverage: {n_eeg}/{len(out)} epochs; EMG RMS: {n_emg}/{len(out)}."
    else:
        info["eeg_emg"] = "No epoch_features.csv; compute epoch features first."

    try:
        metadata = json.loads((rec_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metadata = {}
    photo, photo_std, photo_info = align_photometry(out, rec_dir, metadata)
    out["photometry_mean_raw"] = photo
    out["photometry_std_raw"] = photo_std
    out["photometry_mean_z"] = robust_z(photo)
    info["photometry"] = photo_info
    return out, info


def summarize_event_physiology(epochs: pd.DataFrame, events: pd.DataFrame, threshold=0.2):
    """Event medians and contrasts to unflagged *same-state* background epochs.

    Core-event epochs are compared with other epochs of the same Somnotate state
    (fall back to Final if no valid Somnotate label). These are within-recording
    descriptive contrasts, *not* significance tests or true controls.
    """
    available = (*PROFILE_COLUMNS, *EVENT_RAW_COLUMNS)
    output_columns = ["event_id", "core_start_s", "core_end_s", "n_physiology_epochs",
                      "reference_source", "reference_state", "reference_n_epochs"]
    for col in available:
        output_columns.append("event_" + col)
    for col in PROFILE_COLUMNS:
        output_columns.extend(("background_" + col, "delta_" + col))
    if events is None or events.empty:
        return pd.DataFrame(columns=output_columns)

    t0 = pd.to_numeric(epochs["t0_s"], errors="coerce").to_numpy(dtype=float)
    t1 = pd.to_numeric(epochs["t1_s"], errors="coerce").to_numpy(dtype=float)
    mid = (t0 + t1) / 2
    score = pd.to_numeric(epochs["dissociation_index"], errors="coerce").to_numpy(dtype=float)
    in_any_event = np.zeros(len(epochs), dtype=bool)
    for _, ev in events.iterrows():
        in_any_event |= (mid >= float(ev["core_start_s"])) & (mid < float(ev["core_end_s"]))

    records = []
    for _, ev in events.iterrows():
        core = (mid >= float(ev["core_start_s"])) & (mid < float(ev["core_end_s"]))
        row = {"event_id": ev["event_id"], "core_start_s": ev["core_start_s"],
               "core_end_s": ev["core_end_s"], "n_physiology_epochs": int(core.sum())}
        reference_source = ""
        reference_state = ""
        for source in ("somnotate_state", "final_state", "manual_state"):
            if source not in epochs.columns:
                continue
            values = epochs.loc[core, source].astype(str)
            defined = values[values.isin(["Wake", "NREM", "REM"])]
            if len(defined):
                reference_state = str(defined.mode().iloc[0])
                reference_source = source
                break
        row["reference_source"] = reference_source
        row["reference_state"] = reference_state

        background = np.zeros(len(epochs), dtype=bool)
        if reference_source:
            background = (
                (~in_any_event) & np.isfinite(score) & (score < float(threshold)) &
                epochs[reference_source].astype(str).eq(reference_state).to_numpy()
            )
        row["reference_n_epochs"] = int(background.sum())
        for col in available:
            vals = pd.to_numeric(epochs[col], errors="coerce").to_numpy(dtype=float) if col in epochs else np.full(len(epochs), np.nan)
            finite_core = vals[core & np.isfinite(vals)]
            row["event_" + col] = float(np.median(finite_core)) if len(finite_core) else np.nan
            if col in PROFILE_COLUMNS:
                valid_ref = vals[background & np.isfinite(vals)]
                # Do not report a contrast to a tiny and unrepresentative reference pool.
                if len(valid_ref) >= 10 and len(finite_core) > 0:
                    med = float(np.median(valid_ref))
                    row["background_" + col] = med
                    row["delta_" + col] = row["event_" + col] - med
                else:
                    row["background_" + col] = np.nan
                    row["delta_" + col] = np.nan
        records.append(row)
    return pd.DataFrame(records, columns=output_columns)
