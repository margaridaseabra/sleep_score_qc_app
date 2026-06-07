from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATE_ORDER = ["Wake", "NREM", "REM", "Sleep", "Uncertain", "Undefined", "Artifact"]


def normalize_state(x):
    x = str(x).strip()

    mapping = {
        "Awake": "Wake",
        "Wake": "Wake",
        "W": "Wake",
        "WK": "Wake",
        "NREM": "NREM",
        "SWS": "NREM",
        "Sleep": "NREM",
        "REM": "REM",
        "PS": "REM",
        "Uncertain": "Uncertain",
        "Undefined": "Undefined",
        "nan": "Undefined",
        "NaN": "Undefined",
        "None": "Undefined",
        "": "Undefined",
        "-1": "Undefined",
        "Artifact": "Artifact",
    }

    return mapping.get(x, x)


def wake_sleep_state(x):
    x = normalize_state(x)

    if x == "Wake":
        return "Wake"

    if x in ["NREM", "REM", "Sleep"]:
        return "Sleep"

    return "Uncertain"


def labels_at_epoch_midpoints(epoch_df, interval_df, label_col, default="Undefined"):
    if interval_df is None or len(interval_df) == 0 or label_col not in interval_df.columns:
        return np.array([default] * len(epoch_df), dtype=object)

    starts = interval_df["t0_s"].to_numpy(dtype=float)
    ends = interval_df["t1_s"].to_numpy(dtype=float)
    labels = interval_df[label_col].fillna(default).astype(str).to_numpy()

    mids = (
        epoch_df["t0_s"].to_numpy(dtype=float)
        + epoch_df["t1_s"].to_numpy(dtype=float)
    ) / 2

    idx = np.searchsorted(starts, mids, side="right") - 1
    out = np.array([default] * len(epoch_df), dtype=object)

    valid = (idx >= 0) & (idx < len(starts))
    valid_idx = idx[valid]
    valid_mid = mids[valid]

    inside = valid_mid < ends[valid_idx]
    out[np.where(valid)[0][inside]] = labels[valid_idx[inside]]

    return np.array([normalize_state(x) for x in out], dtype=object)


def confidence_from_probability_columns(df, prefix):
    prob_cols = [c for c in df.columns if c.startswith(prefix)]

    if not prob_cols:
        return None

    vals = df[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if vals.size == 0:
        return None

    return np.nanmax(vals, axis=1)


def confidence_aligned(epoch_df, source_df, direct_col, prob_prefix):
    if source_df is None:
        return np.full(len(epoch_df), np.nan)

    if direct_col in source_df.columns:
        tmp = source_df[["t0_s", "t1_s", direct_col]].rename(columns={direct_col: "value"})
        vals = labels_at_epoch_midpoints(epoch_df, tmp, "value", default=np.nan)
        return pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)

    conf = confidence_from_probability_columns(source_df, prob_prefix)

    if conf is None:
        return np.full(len(epoch_df), np.nan)

    tmp = source_df[["t0_s", "t1_s"]].copy()
    tmp["value"] = conf
    vals = labels_at_epoch_midpoints(epoch_df, tmp, "value", default=np.nan)

    return pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)


def add_disagreement_column(out, col_a, col_b, out_col, mode="3state"):
    a = out[col_a].astype(str).to_numpy()
    b = out[col_b].astype(str).to_numpy()

    if mode == "wake_sleep":
        a2 = np.array([wake_sleep_state(x) for x in a], dtype=object)
        b2 = np.array([wake_sleep_state(x) for x in b], dtype=object)
    else:
        a2 = np.array([normalize_state(x) for x in a], dtype=object)
        b2 = np.array([normalize_state(x) for x in b], dtype=object)

    comparable = ~np.isin(a2, ["Undefined", "Uncertain", "Artifact"]) & ~np.isin(
        b2, ["Undefined", "Uncertain", "Artifact"]
    )

    disagree = comparable & (a2 != b2)

    out[out_col] = disagree
    out[out_col + "_comparable"] = comparable

    return out


def summarize_pair(out, pair_name, col_a, col_b, disagree_col, mode="3state"):
    comparable = out[disagree_col + "_comparable"].astype(bool)
    disagree = out[disagree_col].astype(bool)

    n_comp = int(comparable.sum())
    n_dis = int(disagree.sum())
    pct = 100 * n_dis / n_comp if n_comp else np.nan

    top_patterns = ""

    if n_dis:
        tmp = out.loc[disagree, [col_a, col_b]].copy()

        if mode == "wake_sleep":
            tmp[col_a] = tmp[col_a].map(wake_sleep_state)
            tmp[col_b] = tmp[col_b].map(wake_sleep_state)

        vc = tmp.value_counts().head(5)
        top_patterns = "; ".join([f"{idx[0]}→{idx[1]}: {count}" for idx, count in vc.items()])

    return {
        "pair": pair_name,
        "mode": mode,
        "n_comparable_epochs": n_comp,
        "n_disagree_epochs": n_dis,
        "percent_disagree": pct,
        "top_disagreement_patterns": top_patterns,
    }


def state_disagreement_summary(out, pair_name, ref_col, other_col, disagree_col, mode="3state"):
    rows = []

    if mode == "wake_sleep":
        ref = out[ref_col].map(wake_sleep_state)
        other = out[other_col].map(wake_sleep_state)
        states = ["Wake", "Sleep"]
    else:
        ref = out[ref_col].map(normalize_state)
        other = out[other_col].map(normalize_state)
        states = ["Wake", "NREM", "REM"]

    comparable = out[disagree_col + "_comparable"].astype(bool)
    disagree = out[disagree_col].astype(bool)

    for state in states:
        mask = comparable & (ref == state)
        n_ref = int(mask.sum())
        n_dis = int((mask & disagree).sum())
        pct = 100 * n_dis / n_ref if n_ref else np.nan

        if n_dis:
            most_common_other = other[mask & disagree].value_counts().idxmax()
        else:
            most_common_other = ""

        rows.append({
            "pair": pair_name,
            "reference_source": ref_col,
            "reference_state": state,
            "n_reference_epochs": n_ref,
            "n_disagree_epochs": n_dis,
            "percent_disagree_within_state": pct,
            "most_common_other_state_when_disagree": most_common_other,
        })

    return rows


def merge_dissociation_events(epoch_df, min_index=0.20, max_gap_s=5, pad_s=10):
    flagged = epoch_df[epoch_df["dissociation_index"] >= float(min_index)].copy()

    if len(flagged) == 0:
        return pd.DataFrame(columns=[
            "event_id", "start_s", "end_s", "start_min", "end_min",
            "duration_s", "max_dissociation_index", "mean_dissociation_index",
            "main_reason", "states_at_peak",
            "n_epochs",
        ])

    flagged = flagged.sort_values("t0_s").reset_index(drop=True)

    events = []
    current = None

    for _, r in flagged.iterrows():
        if current is None:
            current = {
                "start_s": float(r["t0_s"]),
                "end_s": float(r["t1_s"]),
                "rows": [r],
            }
            continue

        gap = float(r["t0_s"]) - current["end_s"]

        if gap <= max_gap_s:
            current["end_s"] = max(current["end_s"], float(r["t1_s"]))
            current["rows"].append(r)
        else:
            events.append(current)
            current = {
                "start_s": float(r["t0_s"]),
                "end_s": float(r["t1_s"]),
                "rows": [r],
            }

    if current is not None:
        events.append(current)

    rows = []

    for i, ev in enumerate(events):
        ev_df = pd.DataFrame(ev["rows"])
        peak_idx = ev_df["dissociation_index"].astype(float).idxmax()
        peak = ev_df.loc[peak_idx]

        reason_counts = {}

        for reason_string in ev_df["dissociation_reason"].fillna("").astype(str):
            for reason in reason_string.split("; "):
                reason = reason.strip()
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1

        main_reason = ""

        if reason_counts:
            main_reason = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[0][0]

        state_cols = [c for c in ["manual_state", "somnotate_state", "layer1_3state", "final_state"] if c in ev_df.columns]
        states_at_peak = "; ".join([f"{c}={peak[c]}" for c in state_cols])

        start_s = max(0.0, float(ev["start_s"]) - pad_s)
        end_s = float(ev["end_s"]) + pad_s

        rows.append({
            "event_id": f"dissociation_{i:05d}",
            "start_s": start_s,
            "end_s": end_s,
            "start_min": start_s / 60,
            "end_min": end_s / 60,
            "duration_s": end_s - start_s,
            "max_dissociation_index": float(ev_df["dissociation_index"].max()),
            "mean_dissociation_index": float(ev_df["dissociation_index"].mean()),
            "main_reason": main_reason,
            "states_at_peak": states_at_peak,
            "n_epochs": int(len(ev_df)),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["max_dissociation_index", "mean_dissociation_index", "duration_s"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    out["rank"] = np.arange(1, len(out) + 1)

    return out


def build_analysis(project_root, recording_id, threshold=0.20):
    project_root = Path(project_root)
    rec_dir = project_root / "recordings" / recording_id

    layer1_file = rec_dir / "layer1_wake_sleep.csv"

    if not layer1_file.exists():
        raise FileNotFoundError(layer1_file)

    layer1 = pd.read_csv(layer1_file)
    out = layer1[["t0_s", "t1_s"]].copy()
    out["recording_id"] = recording_id
    out["epoch_id"] = np.arange(len(out))

    # Layer 1 states
    out["layer1_label"] = layer1["layer1_label"].fillna("Undefined").astype(str)
    out["layer1_wake_sleep"] = out["layer1_label"].map(wake_sleep_state)
    out["layer1_3state"] = out["layer1_label"].map(
        lambda x: "Wake" if str(x) == "Wake" else ("NREM" if str(x) == "Sleep" else "Undefined")
    )

    out["layer1_confidence"] = np.nan

    if "layer1_confidence" in layer1.columns:
        out["layer1_confidence"] = pd.to_numeric(layer1["layer1_confidence"], errors="coerce")
    else:
        conf = confidence_from_probability_columns(layer1, "layer1_P_")
        if conf is not None:
            out["layer1_confidence"] = conf

    # Manual
    manual_file = rec_dir / "manual_scoring_aligned.csv"
    manual = None

    if manual_file.exists():
        manual = pd.read_csv(manual_file)
        out["manual_state"] = labels_at_epoch_midpoints(out, manual, "manual_state")
    else:
        out["manual_state"] = "Undefined"

    # Somnotate
    som_file = rec_dir / "somnotate" / "somnotate_results_timeseries.csv"
    som = None

    if som_file.exists():
        som = pd.read_csv(som_file)
        out["somnotate_state"] = labels_at_epoch_midpoints(out, som, "somnotate_state")
        out["somnotate_confidence"] = confidence_aligned(out, som, "somnotate_confidence", "somnotate_P_")
    else:
        out["somnotate_state"] = "Undefined"
        out["somnotate_confidence"] = np.nan

    # Final/app scoring
    final_file = rec_dir / "final_scoring.csv"
    final = None

    if final_file.exists():
        final = pd.read_csv(final_file)
        out["final_state"] = labels_at_epoch_midpoints(out, final, "final_state")
        if "review_status" in final.columns:
            tmp = final[["t0_s", "t1_s", "review_status"]].copy()
            out["final_review_status"] = labels_at_epoch_midpoints(out, tmp, "review_status", default="not_reviewed")
        else:
            out["final_review_status"] = "unknown"
    else:
        out["final_state"] = "Undefined"
        out["final_review_status"] = "missing"

    # EMG features
    feat_file = rec_dir / "epoch_features.csv"

    if feat_file.exists():
        feat = pd.read_csv(feat_file)
        if "epoch_id" in feat.columns:
            for col in ["emg_rms_z", "emg_abs_p95_z"]:
                if col in feat.columns:
                    out = out.merge(feat[["epoch_id", col]], on="epoch_id", how="left")
    if "emg_rms_z" not in out.columns:
        out["emg_rms_z"] = np.nan

    pair_rows = []
    state_rows = []

    # Pairwise disagreement definitions
    pair_defs = []

    if som is not None and manual is not None:
        pair_defs.append(("Manual vs Somnotate", "manual_state", "somnotate_state", "manual_vs_somnotate_3state", "3state"))

    if som is not None:
        pair_defs.append(("Layer 1 vs Somnotate", "layer1_3state", "somnotate_state", "layer1_vs_somnotate_3state", "3state"))
        pair_defs.append(("Layer 1 vs Somnotate WS", "layer1_label", "somnotate_state", "layer1_vs_somnotate_ws", "wake_sleep"))

    if manual is not None:
        pair_defs.append(("Layer 1 vs Manual WS", "layer1_label", "manual_state", "layer1_vs_manual_ws", "wake_sleep"))

    if final is not None and manual is not None:
        pair_defs.append(("App final vs Manual", "final_state", "manual_state", "final_vs_manual_3state", "3state"))

    if final is not None and som is not None:
        pair_defs.append(("App final vs Somnotate", "final_state", "somnotate_state", "final_vs_somnotate_3state", "3state"))

    for pair_name, col_a, col_b, out_col, mode in pair_defs:
        out = add_disagreement_column(out, col_a, col_b, out_col, mode=mode)
        pair_rows.append(summarize_pair(out, pair_name, col_a, col_b, out_col, mode=mode))
        state_rows.extend(state_disagreement_summary(out, pair_name, col_a, col_b, out_col, mode=mode))

    # Dissociation index
    score = np.zeros(len(out), dtype=float)
    max_possible = np.zeros(len(out), dtype=float)
    reasons = [[] for _ in range(len(out))]

    weights = {
        "3state": 2.0,
        "wake_sleep": 1.5,
        "som_uncertainty": 1.5,
        "layer1_uncertainty": 1.0,
        "rem_high_emg": 1.5,
        "missing": 1.0,
    }

    for _, _, _, out_col, mode in pair_defs:
        w = weights["wake_sleep"] if mode == "wake_sleep" else weights["3state"]
        comparable = out[out_col + "_comparable"].astype(bool).to_numpy()
        disagree = out[out_col].astype(bool).to_numpy()

        max_possible[comparable] += w
        score[disagree] += w

        for idx in np.where(disagree)[0]:
            reasons[idx].append(out_col)

    if "somnotate_confidence" in out.columns:
        conf = pd.to_numeric(out["somnotate_confidence"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(conf)
        max_possible[valid] += weights["som_uncertainty"]
        add = np.zeros(len(out), dtype=float)
        add[valid] = (1 - np.clip(conf[valid], 0, 1)) * weights["som_uncertainty"]
        score += add
        for idx in np.where(add > 0.4)[0]:
            reasons[idx].append("low_somnotate_confidence")

    if "layer1_confidence" in out.columns:
        conf = pd.to_numeric(out["layer1_confidence"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(conf)
        max_possible[valid] += weights["layer1_uncertainty"]
        add = np.zeros(len(out), dtype=float)
        add[valid] = (1 - np.clip(conf[valid], 0, 1)) * weights["layer1_uncertainty"]
        score += add
        for idx in np.where(add > 0.3)[0]:
            reasons[idx].append("low_layer1_confidence")

    # REM with high EMG from any REM-like source
    emg = pd.to_numeric(out["emg_rms_z"], errors="coerce").to_numpy(dtype=float)
    rem_any = (
        out["somnotate_state"].astype(str).eq("REM").to_numpy()
        | out["manual_state"].astype(str).eq("REM").to_numpy()
        | out["final_state"].astype(str).eq("REM").to_numpy()
    )
    high_emg = rem_any & np.isfinite(emg) & (emg > 2.0)

    max_possible[rem_any] += weights["rem_high_emg"]
    score[high_emg] += weights["rem_high_emg"]

    for idx in np.where(high_emg)[0]:
        reasons[idx].append("REM_with_high_EMG")

    missing = (
        out["somnotate_state"].astype(str).isin(["Undefined", "Uncertain"]).to_numpy()
        | out["layer1_label"].astype(str).isin(["Undefined", "Uncertain"]).to_numpy()
    )
    max_possible[missing] += weights["missing"]
    score[missing] += weights["missing"]

    for idx in np.where(missing)[0]:
        reasons[idx].append("missing_or_uncertain_model_state")

    out["raw_dissociation_score"] = score
    out["dissociation_index"] = np.divide(
        score,
        np.maximum(max_possible, 1e-9),
        out=np.zeros_like(score),
        where=max_possible > 0,
    )
    out["dissociation_reason"] = ["; ".join(sorted(set(r))) for r in reasons]

    events = merge_dissociation_events(out, min_index=threshold, max_gap_s=5, pad_s=10)

    analysis_dir = rec_dir / "dissociation_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    epochs_path = analysis_dir / "dissociation_epochs.csv"
    events_path = analysis_dir / "dissociation_events.csv"
    pair_path = analysis_dir / "dissociation_pairwise_summary.csv"
    state_path = analysis_dir / "dissociation_state_summary.csv"

    out.to_csv(epochs_path, index=False)
    events.to_csv(events_path, index=False)
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
    pd.DataFrame(state_rows).to_csv(state_path, index=False)

    print("Dissociation analysis complete.")
    print("Epochs:", epochs_path)
    print("Events:", events_path)
    print("Pairwise summary:", pair_path)
    print("State summary:", state_path)
    print("Number of dissociation events:", len(events))

    if len(events):
        print(events[["rank", "event_id", "start_min", "end_min", "max_dissociation_index", "main_reason"]].head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--threshold", type=float, default=0.20)

    args = parser.parse_args()

    build_analysis(
        project_root=args.project_root,
        recording_id=args.recording_id,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
