from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


FINAL_CODE_MAP = {
    "Wake": 0,
    "NREM": 1,
    "REM": 2,
    "Sleep": 1,
    "Uncertain": -1,
    "Undefined": -1,
    "Artifact": -2,
}


def normalize_state(x):
    x = str(x).strip()

    mapping = {
        "Awake": "Wake",
        "Wake": "Wake",
        "W": "Wake",
        "WK": "Wake",
        "NREM": "NREM",
        "SWS": "NREM",
        "Sleep": "Sleep",
        "REM": "REM",
        "PS": "REM",
        "Uncertain": "Uncertain",
        "Undefined": "Undefined",
        "nan": "Undefined",
        "NaN": "Undefined",
        "": "Undefined",
        "-1": "Undefined",
        "Artifact": "Artifact",
    }

    return mapping.get(x, x)


def wake_sleep(state):
    state = normalize_state(state)

    if state == "Wake":
        return "Wake"

    if state in ["NREM", "REM", "Sleep"]:
        return "Sleep"

    return "Uncertain"


def load_metadata(rec_dir: Path):
    path = rec_dir / "metadata.json"

    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text())


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


def load_layers(project_root: Path, recording_id: str):
    rec_dir = project_root / "recordings" / recording_id

    metadata = load_metadata(rec_dir)
    duration_s = float(metadata["duration_s"])

    n = int(np.ceil(duration_s))

    base = pd.DataFrame({
        "recording_id": recording_id,
        "epoch_id": np.arange(n),
        "t0_s": np.arange(n, dtype=float),
        "t1_s": np.arange(n, dtype=float) + 1.0,
    })

    # Layer 1
    l1_path = rec_dir / "layer1_wake_sleep.csv"

    if l1_path.exists():
        l1 = pd.read_csv(l1_path)
        base["layer1_label"] = labels_at_epoch_midpoints(base, l1, "layer1_label")

        for col in ["layer1_P_Wake", "layer1_P_Sleep", "layer1_confidence", "layer1_uncertainty"]:
            if col in l1.columns:
                tmp = l1[["t0_s", "t1_s", col]].rename(columns={col: "value"})
                base[col] = pd.to_numeric(
                    labels_at_epoch_midpoints(base, tmp, "value", default=np.nan),
                    errors="coerce",
                )
    else:
        base["layer1_label"] = "Undefined"

    # Manual
    man_path = rec_dir / "manual_scoring_aligned.csv"

    if man_path.exists():
        man = pd.read_csv(man_path)
        base["manual_state"] = labels_at_epoch_midpoints(base, man, "manual_state")
    else:
        base["manual_state"] = "Undefined"

    # Somnotate
    som_path = rec_dir / "somnotate" / "somnotate_results_timeseries.csv"

    if som_path.exists():
        som = pd.read_csv(som_path)
        base["somnotate_state"] = labels_at_epoch_midpoints(base, som, "somnotate_state")

        for col in som.columns:
            if col.startswith("somnotate_P_") or col in ["somnotate_confidence", "somnotate_uncertainty"]:
                tmp = som[["t0_s", "t1_s", col]].rename(columns={col: "value"})
                base[col] = pd.to_numeric(
                    labels_at_epoch_midpoints(base, tmp, "value", default=np.nan),
                    errors="coerce",
                )
    else:
        base["somnotate_state"] = "Undefined"

    # Features
    feat_path = rec_dir / "epoch_features.csv"

    if feat_path.exists():
        feat = pd.read_csv(feat_path)

        if "epoch_id" in feat.columns:
            keep = [c for c in ["epoch_id", "emg_rms_z", "emg_abs_p95_z"] if c in feat.columns]
            base = base.merge(feat[keep], on="epoch_id", how="left")

    return base


def choose_initial_final(row):
    manual = normalize_state(row.get("manual_state", "Undefined"))
    som = normalize_state(row.get("somnotate_state", "Undefined"))
    l1 = normalize_state(row.get("layer1_label", "Undefined"))

    if manual not in ["Undefined", "Uncertain"]:
        return manual, "manual"

    if som not in ["Undefined", "Uncertain"]:
        return som, "somnotate"

    if l1 == "Wake":
        return "Wake", "layer1"

    if l1 == "Sleep":
        return "NREM", "layer1_sleep_as_nrem"

    return "Undefined", "none"


def initialize_final(project_root: Path, recording_id: str, overwrite=False):
    rec_dir = project_root / "recordings" / recording_id
    out_path = rec_dir / "final_scoring.csv"

    if out_path.exists() and not overwrite:
        print("final_scoring.csv already exists:", out_path)
        return out_path

    layers = load_layers(project_root, recording_id)

    final_states = []
    sources = []

    for _, row in layers.iterrows():
        state, source = choose_initial_final(row)
        final_states.append(state)
        sources.append(source)

    out = layers[["recording_id", "epoch_id", "t0_s", "t1_s"]].copy()
    out["final_state"] = final_states
    out["final_code"] = [FINAL_CODE_MAP.get(s, -1) for s in final_states]
    out["final_source"] = sources
    out["review_status"] = "not_reviewed"
    out["review_notes"] = ""

    out.to_csv(out_path, index=False)

    print("Initialized final scoring:", out_path)
    print(out["final_state"].value_counts().to_string())

    return out_path


def add_flag(flags, mask, reason, priority):
    mask = np.asarray(mask, dtype=bool)

    flags.loc[mask, "flagged"] = True
    flags.loc[mask, "priority_score"] += priority

    def append_reason(x):
        x = "" if pd.isna(x) else str(x)

        if not x:
            return reason

        if reason in x.split("; "):
            return x

        return x + "; " + reason

    flags.loc[mask, "reason_for_review"] = flags.loc[mask, "reason_for_review"].apply(append_reason)


def flag_transitions(flags, state_col, reason, priority, window_s=5):
    states = flags[state_col].astype(str).to_numpy()
    change = np.where(states[1:] != states[:-1])[0] + 1

    mask = np.zeros(len(flags), dtype=bool)

    for idx in change:
        t = float(flags.iloc[idx]["t0_s"])
        mask |= (
            (flags["t0_s"] >= t - window_s)
            & (flags["t1_s"] <= t + window_s)
        ).to_numpy()

    add_flag(flags, mask, reason, priority)


def merge_flagged_epochs(flags, max_gap_s=5, pad_s=10):
    flagged = flags[flags["flagged"]].copy()

    if len(flagged) == 0:
        return pd.DataFrame(columns=[
            "review_id", "recording_id", "start_s", "end_s",
            "start_min", "end_min", "duration_s",
            "reason_for_review", "priority_score", "n_epochs",
        ])

    flagged = flagged.sort_values("t0_s").reset_index(drop=True)

    intervals = []
    current = None

    for _, r in flagged.iterrows():
        if current is None:
            current = {
                "recording_id": r["recording_id"],
                "start_s": float(r["t0_s"]),
                "end_s": float(r["t1_s"]),
                "reasons": set(str(r["reason_for_review"]).split("; ")),
                "priority_score": float(r["priority_score"]),
                "n_epochs": 1,
            }
            continue

        gap = float(r["t0_s"]) - current["end_s"]

        if gap <= max_gap_s:
            current["end_s"] = max(current["end_s"], float(r["t1_s"]))
            current["reasons"].update(str(r["reason_for_review"]).split("; "))
            current["priority_score"] += float(r["priority_score"])
            current["n_epochs"] += 1
        else:
            intervals.append(current)
            current = {
                "recording_id": r["recording_id"],
                "start_s": float(r["t0_s"]),
                "end_s": float(r["t1_s"]),
                "reasons": set(str(r["reason_for_review"]).split("; ")),
                "priority_score": float(r["priority_score"]),
                "n_epochs": 1,
            }

    if current is not None:
        intervals.append(current)

    rows = []

    for i, iv in enumerate(intervals):
        start_s = max(0.0, iv["start_s"] - pad_s)
        end_s = iv["end_s"] + pad_s

        rows.append({
            "review_id": f"review_{i:05d}",
            "recording_id": iv["recording_id"],
            "start_s": start_s,
            "end_s": end_s,
            "start_min": start_s / 60,
            "end_min": end_s / 60,
            "duration_s": end_s - start_s,
            "reason_for_review": "; ".join(sorted([x for x in iv["reasons"] if x and x != "nan"])),
            "priority_score": iv["priority_score"],
            "n_epochs": iv["n_epochs"],
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["priority_score", "start_s"], ascending=[False, True]).reset_index(drop=True)

    return out


def build_queue(project_root: Path, recording_id: str, layer1_conf_thr=0.70, som_conf_thr=0.70, high_emg_z=2.0):
    rec_dir = project_root / "recordings" / recording_id
    layers = load_layers(project_root, recording_id)

    flags = layers.copy()
    flags["flagged"] = False
    flags["reason_for_review"] = ""
    flags["priority_score"] = 0.0

    l1_ws = flags["layer1_label"].map(wake_sleep)
    som_ws = flags["somnotate_state"].map(wake_sleep)
    man_ws = flags["manual_state"].map(wake_sleep)

    add_flag(flags, (l1_ws != "Uncertain") & (som_ws != "Uncertain") & (l1_ws != som_ws),
             "Layer 1 vs Somnotate Wake/Sleep disagreement", 4.0)

    add_flag(flags, (man_ws != "Uncertain") & (som_ws != "Uncertain") & (man_ws != som_ws),
             "Manual vs Somnotate disagreement", 4.0)

    add_flag(flags, (man_ws != "Uncertain") & (l1_ws != "Uncertain") & (man_ws != l1_ws),
             "Manual vs Layer 1 disagreement", 3.5)

    if "layer1_confidence" in flags.columns:
        add_flag(flags, pd.to_numeric(flags["layer1_confidence"], errors="coerce") < layer1_conf_thr,
                 "low Layer 1 confidence", 2.0)

    if "somnotate_confidence" in flags.columns:
        add_flag(flags, pd.to_numeric(flags["somnotate_confidence"], errors="coerce") < som_conf_thr,
                 "low Somnotate confidence", 2.5)

    add_flag(flags, flags["somnotate_state"].astype(str).isin(["Undefined", "Uncertain"]),
             "missing/undefined Somnotate", 1.5)

    flag_transitions(flags, "somnotate_state", "Somnotate state transition", 1.0, window_s=5)
    flag_transitions(flags, "layer1_label", "Layer 1 state transition", 0.75, window_s=5)

    if "emg_rms_z" in flags.columns:
        add_flag(
            flags,
            (flags["somnotate_state"].astype(str) == "REM")
            & (pd.to_numeric(flags["emg_rms_z"], errors="coerce") > high_emg_z),
            "REM with high EMG",
            5.0,
        )

    rem = flags["somnotate_state"].astype(str).eq("REM").to_numpy()
    short_rem_mask = np.zeros(len(flags), dtype=bool)

    i = 0
    while i < len(rem):
        if not rem[i]:
            i += 1
            continue

        start = i

        while i < len(rem) and rem[i]:
            i += 1

        end = i
        dur = float(flags.iloc[end - 1]["t1_s"]) - float(flags.iloc[start]["t0_s"])

        if dur <= 20:
            short_rem_mask[start:end] = True

    add_flag(flags, short_rem_mask, "very short REM fragment", 3.0)

    queue = merge_flagged_epochs(flags, max_gap_s=5, pad_s=10)

    flags_path = rec_dir / "review_flagged_epochs.csv"
    queue_path = rec_dir / "review_queue.csv"

    flags.to_csv(flags_path, index=False)
    queue.to_csv(queue_path, index=False)

    print("Built review queue:", queue_path)
    print("Review intervals:", len(queue))

    if len(queue):
        print(queue[["review_id", "start_min", "end_min", "reason_for_review", "priority_score"]].head(20).to_string(index=False))

    return queue_path


def apply_edit(project_root: Path, recording_id: str, start_s: float, end_s: float, mode: str, label: str = "", notes: str = ""):
    rec_dir = project_root / "recordings" / recording_id
    final_path = rec_dir / "final_scoring.csv"

    if not final_path.exists():
        initialize_final(project_root, recording_id, overwrite=False)

    final = pd.read_csv(final_path)
    layers = load_layers(project_root, recording_id)

    mask = (
        (final["t0_s"].astype(float) < float(end_s))
        & (final["t1_s"].astype(float) > float(start_s))
    )

    if mask.sum() == 0:
        raise ValueError("No epochs found in selected interval.")

    if mode == "manual_label":
        new_states = np.array([normalize_state(label)] * int(mask.sum()), dtype=object)
        source = "manual_edit"

    elif mode == "use_manual":
        new_states = layers.loc[mask, "manual_state"].map(normalize_state).to_numpy()
        source = "accepted_manual"

    elif mode == "use_somnotate":
        new_states = layers.loc[mask, "somnotate_state"].map(normalize_state).to_numpy()
        source = "accepted_somnotate"

    elif mode == "use_layer1":
        vals = layers.loc[mask, "layer1_label"].map(normalize_state).to_numpy()
        new_states = []

        for v in vals:
            if v == "Wake":
                new_states.append("Wake")
            elif v == "Sleep":
                new_states.append("NREM")
            else:
                new_states.append("Undefined")

        new_states = np.array(new_states, dtype=object)
        source = "accepted_layer1"

    else:
        raise ValueError(f"Unknown edit mode: {mode}")

    final.loc[mask, "final_state"] = new_states
    final.loc[mask, "final_code"] = [FINAL_CODE_MAP.get(s, -1) for s in new_states]
    final.loc[mask, "final_source"] = source
    final.loc[mask, "review_status"] = "reviewed"
    final.loc[mask, "review_notes"] = notes

    final.to_csv(final_path, index=False)

    log_path = rec_dir / "review_edit_log.csv"

    log_row = pd.DataFrame([{
        "recording_id": recording_id,
        "start_s": start_s,
        "end_s": end_s,
        "mode": mode,
        "label": label,
        "source": source,
        "notes": notes,
        "n_epochs": int(mask.sum()),
        "saved_at": pd.Timestamp.now().isoformat(),
    }])

    if log_path.exists():
        log = pd.read_csv(log_path)
        log = pd.concat([log, log_row], ignore_index=True)
    else:
        log = log_row

    log.to_csv(log_path, index=False)

    print("Applied edit.")
    print("Interval:", start_s, "to", end_s, "s")
    print("Mode:", mode)
    print("Epochs:", int(mask.sum()))
    print("Final scoring:", final_path)


def export_final(project_root: Path, recording_id: str):
    rec_dir = project_root / "recordings" / recording_id
    final_path = rec_dir / "final_scoring.csv"

    if not final_path.exists():
        raise FileNotFoundError(final_path)

    final = pd.read_csv(final_path)
    final["final_state"] = final["final_state"].map(normalize_state)
    final["final_code"] = final["final_state"].map(FINAL_CODE_MAP).fillna(-1).astype(int)

    csv_out = rec_dir / "final_scoring_export.csv"
    mat_out = rec_dir / "final_scoring_export.mat"

    final.to_csv(csv_out, index=False)

    savemat(mat_out, {
        "scoring": final["final_code"].to_numpy(dtype=np.int16),
        "scoring_state": final["final_state"].astype(str).to_numpy(dtype=object),
        "t0_s": final["t0_s"].to_numpy(dtype=float),
        "t1_s": final["t1_s"].to_numpy(dtype=float),
    })

    print("Exported:")
    print(csv_out)
    print(mat_out)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-final")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-id", required=True)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("build-queue")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-id", required=True)
    p.add_argument("--layer1-confidence-threshold", type=float, default=0.70)
    p.add_argument("--somnotate-confidence-threshold", type=float, default=0.70)
    p.add_argument("--high-emg-z", type=float, default=2.0)

    p = sub.add_parser("apply-edit")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-id", required=True)
    p.add_argument("--start-s", type=float, required=True)
    p.add_argument("--end-s", type=float, required=True)
    p.add_argument("--mode", required=True, choices=["manual_label", "use_manual", "use_somnotate", "use_layer1"])
    p.add_argument("--label", default="")
    p.add_argument("--notes", default="")

    p = sub.add_parser("export-final")
    p.add_argument("--project-root", required=True)
    p.add_argument("--recording-id", required=True)

    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()

    if args.command == "init-final":
        initialize_final(project_root, args.recording_id, overwrite=args.overwrite)

    elif args.command == "build-queue":
        build_queue(
            project_root,
            args.recording_id,
            layer1_conf_thr=args.layer1_confidence_threshold,
            som_conf_thr=args.somnotate_confidence_threshold,
            high_emg_z=args.high_emg_z,
        )

    elif args.command == "apply-edit":
        apply_edit(
            project_root,
            args.recording_id,
            start_s=args.start_s,
            end_s=args.end_s,
            mode=args.mode,
            label=args.label,
            notes=args.notes,
        )

    elif args.command == "export-final":
        export_final(project_root, args.recording_id)


if __name__ == "__main__":
    main()
