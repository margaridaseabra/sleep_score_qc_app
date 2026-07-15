from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if not np.isfinite(mad) or mad == 0:
        mad = np.nanstd(x)

    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)

    return (x - med) / (1.4826 * mad)


def bandpower(x, fs, fmin, fmax):
    if len(x) < 4:
        return np.nan

    nperseg = min(len(x), int(fs))
    f, pxx = welch(x, fs=fs, nperseg=nperseg)

    mask = (f >= fmin) & (f < fmax)

    if not np.any(mask):
        return np.nan

    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(pxx[mask], f[mask]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--epoch-sec", type=float, default=1.0)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    rec_dir = project_root / "recordings" / args.recording_id

    metadata = json.loads((rec_dir / "metadata.json").read_text())

    fs = float(metadata["sampling_rate_hz"])
    eeg = np.load(rec_dir / "eeg.npy", mmap_mode="r")
    emg = np.load(rec_dir / "emg.npy", mmap_mode="r")

    samples_per_epoch = int(round(args.epoch_sec * fs))
    n_epochs = min(len(eeg), len(emg)) // samples_per_epoch

    rows = []

    for i in range(n_epochs):
        a = i * samples_per_epoch
        b = a + samples_per_epoch

        eeg_ep = np.asarray(eeg[a:b], dtype=float)
        emg_ep = np.asarray(emg[a:b], dtype=float)

        emg_rms = float(np.sqrt(np.mean(emg_ep ** 2)))
        emg_abs_p95 = float(np.percentile(np.abs(emg_ep), 95))
        emg_abs_p99 = float(np.percentile(np.abs(emg_ep), 99))

        delta = bandpower(eeg_ep, fs, 0.5, 4)
        theta = bandpower(eeg_ep, fs, 5, 10)
        sigma = bandpower(eeg_ep, fs, 10, 15)
        broadband = bandpower(eeg_ep, fs, 0.5, 30)

        rows.append({
            "recording_id": args.recording_id,
            "epoch_id": i,
            "t0_s": i * args.epoch_sec,
            "t1_s": (i + 1) * args.epoch_sec,
            "emg_rms": emg_rms,
            "emg_abs_p95": emg_abs_p95,
            "emg_abs_p99": emg_abs_p99,
            "eeg_delta": delta,
            "eeg_theta": theta,
            "eeg_sigma": sigma,
            "eeg_broadband": broadband,
            "eeg_theta_delta_ratio": theta / delta if delta and np.isfinite(delta) and delta > 0 else np.nan,
        })

    df = pd.DataFrame(rows)

    for col in ["emg_rms", "emg_abs_p95", "emg_abs_p99"]:
        df[f"log_{col}"] = np.log10(df[col].clip(lower=1e-12))
        df[f"{col}_z"] = robust_z(df[f"log_{col}"])

    for col in ["eeg_delta", "eeg_theta", "eeg_sigma", "eeg_broadband", "eeg_theta_delta_ratio"]:
        df[f"log_{col}"] = np.log10(df[col].clip(lower=1e-12))
        df[f"{col}_z"] = robust_z(df[f"log_{col}"])

    out = rec_dir / "epoch_features.csv"
    df.to_csv(out, index=False)

    print("Wrote:", out)
    print("Epochs:", len(df))


if __name__ == "__main__":
    main()
