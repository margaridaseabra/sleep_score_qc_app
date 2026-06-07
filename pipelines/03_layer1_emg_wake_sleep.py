from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def rolling_mean(x, win):
    x = np.asarray(x, dtype=float)
    win = max(1, int(win))

    if win == 1:
        return x.copy()

    s = pd.Series(x)
    return s.rolling(win, center=True, min_periods=1).mean().to_numpy()


def rolling_median(x, win):
    x = np.asarray(x, dtype=float)
    win = max(1, int(win))

    if win == 1:
        return x.copy()

    s = pd.Series(x)
    return s.rolling(win, center=True, min_periods=1).median().to_numpy()


def fill_short_gaps(mask, max_gap_epochs):
    """
    Fill short False gaps inside True regions.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    n = len(mask)

    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue

        start = i
        while i < n and not mask[i]:
            i += 1
        end = i

        left_true = start > 0 and mask[start - 1]
        right_true = end < n and mask[end]

        if left_true and right_true and (end - start) <= max_gap_epochs:
            mask[start:end] = True

    return mask


def remove_short_true_bouts(mask, min_len_epochs):
    """
    Remove short True bouts.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    n = len(mask)

    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue

        start = i
        while i < n and mask[i]:
            i += 1
        end = i

        if (end - start) < min_len_epochs:
            mask[start:end] = False

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--recording-id", required=True)

    parser.add_argument("--epoch-sec", type=float, default=1.0)

    # EMG/GMM thresholds
    parser.add_argument("--wake-prob-threshold", type=float, default=0.70)
    parser.add_argument("--sleep-prob-threshold", type=float, default=0.35)
    parser.add_argument("--wake-emg-z", type=float, default=1.25)
    parser.add_argument("--strong-wake-emg-z", type=float, default=2.50)
    parser.add_argument("--sleep-emg-z", type=float, default=0.80)

    # Temporal context
    parser.add_argument("--context-window-s", type=float, default=20.0)
    parser.add_argument("--merge-gap-s", type=float, default=10.0)
    parser.add_argument("--min-wake-bout-s", type=float, default=8.0)
    parser.add_argument("--min-sleep-bout-s", type=float, default=10.0)

    args = parser.parse_args()

    rec_dir = Path(args.project_root) / "recordings" / args.recording_id
    features_path = rec_dir / "epoch_features.csv"

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    features = pd.read_csv(features_path)

    if "log_emg_rms" not in features.columns:
        raise ValueError("epoch_features.csv must contain log_emg_rms")

    if "emg_rms_z" not in features.columns:
        raise ValueError("epoch_features.csv must contain emg_rms_z")

    x = features["log_emg_rms"].to_numpy(dtype=float)
    emg_z = features["emg_rms_z"].to_numpy(dtype=float)

    valid = np.isfinite(x)

    if valid.sum() < 20:
        raise RuntimeError("Not enough valid EMG epochs to fit Layer 1.")

    # ------------------------------------------------------------------
    # 1. Fit 2-component GMM on log EMG RMS.
    #    High-EMG component = Wake-like component.
    # ------------------------------------------------------------------
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(x[valid].reshape(-1, 1))

    means = gmm.means_.squeeze()
    wake_component = int(np.argmax(means))

    p_wake_raw = np.full(len(features), np.nan)
    p_wake_raw[valid] = gmm.predict_proba(x[valid].reshape(-1, 1))[:, wake_component]

    # ------------------------------------------------------------------
    # 2. Add temporal context.
    #    Awake periods should usually form bouts, not random isolated 1 s points.
    # ------------------------------------------------------------------
    context_epochs = max(1, int(round(args.context_window_s / args.epoch_sec)))

    p_wake_smooth = rolling_mean(p_wake_raw, context_epochs)
    emg_z_smooth = rolling_median(emg_z, context_epochs)

    # Active fraction: how much of the local context has elevated EMG?
    active_epoch = emg_z > args.wake_emg_z
    active_fraction = rolling_mean(active_epoch.astype(float), context_epochs)

    # Combined wake evidence.
    # This gives weight to both probabilistic high-EMG cluster and sustained EMG activity.
    p_wake_context = (
        0.55 * p_wake_smooth
        + 0.30 * np.clip(active_fraction, 0, 1)
        + 0.15 * np.clip((emg_z_smooth + 1.0) / 4.0, 0, 1)
    )

    # Strong isolated EMG can still be Wake, but most Wake calls should be contextual.
    wake_core = (
        (p_wake_context >= args.wake_prob_threshold)
        | (emg_z_smooth >= args.wake_emg_z)
        | (emg_z >= args.strong_wake_emg_z)
    )

    sleep_core = (
        (p_wake_context <= args.sleep_prob_threshold)
        & (emg_z_smooth < args.sleep_emg_z)
        & (active_fraction < 0.20)
    )

    # ------------------------------------------------------------------
    # 3. Bout cleanup.
    # ------------------------------------------------------------------
    merge_gap_epochs = max(1, int(round(args.merge_gap_s / args.epoch_sec)))
    min_wake_epochs = max(1, int(round(args.min_wake_bout_s / args.epoch_sec)))
    min_sleep_epochs = max(1, int(round(args.min_sleep_bout_s / args.epoch_sec)))

    wake_bouts = fill_short_gaps(wake_core, merge_gap_epochs)
    wake_bouts = remove_short_true_bouts(wake_bouts, min_wake_epochs)

    sleep_bouts = fill_short_gaps(sleep_core, merge_gap_epochs)
    sleep_bouts = remove_short_true_bouts(sleep_bouts, min_sleep_epochs)

    label = np.array(["Uncertain"] * len(features), dtype=object)

    label[sleep_bouts] = "Sleep"
    label[wake_bouts] = "Wake"

    # Wake wins if both happen after smoothing.
    label[wake_bouts] = "Wake"

    # Conservative probability-like outputs.
    p_wake_final = np.clip(p_wake_context, 0, 1)
    p_sleep_final = 1 - p_wake_final

    confidence = np.maximum(p_wake_final, p_sleep_final)
    confidence[label == "Uncertain"] = np.minimum(confidence[label == "Uncertain"], 0.60)

    out = features[["recording_id", "epoch_id", "t0_s", "t1_s"]].copy()
    out["layer1_P_Wake_raw"] = p_wake_raw
    out["layer1_P_Wake"] = p_wake_final
    out["layer1_P_Sleep"] = p_sleep_final
    out["layer1_confidence"] = confidence
    out["layer1_uncertainty"] = 1 - confidence
    out["layer1_label"] = label
    out["layer1_emg_z_smooth"] = emg_z_smooth
    out["layer1_active_fraction"] = active_fraction
    out["layer1_reason"] = np.where(
        label == "Wake",
        "sustained_high_EMG_or_high_EMG_bout",
        np.where(label == "Sleep", "sustained_low_EMG", "ambiguous_EMG_context")
    )

    out_path = rec_dir / "layer1_wake_sleep.csv"
    out.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print()
    print("Layer 1 distribution:")
    print(out["layer1_label"].value_counts().to_string())

    print()
    print("GMM log EMG RMS means:", means)
    print("Wake component:", wake_component)


if __name__ == "__main__":
    main()
