from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--automated-file", default="")
    parser.add_argument("--probability-file", default="")
    parser.add_argument("--copy-files", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    rec_dir = project_root / "recordings" / args.recording_id
    som_dir = rec_dir / "somnotate"
    som_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = rec_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(metadata_path.read_text())
    duration_s = float(metadata["duration_s"])

    automated_in = Path(args.automated_file).expanduser() if args.automated_file else None
    probability_in = Path(args.probability_file).expanduser() if args.probability_file else None

    if automated_in and not automated_in.exists():
        raise FileNotFoundError(f"Automated Somnotate file not found: {automated_in}")

    if probability_in and not probability_in.exists():
        raise FileNotFoundError(f"Somnotate probability file not found: {probability_in}")

    if automated_in is None and probability_in is None:
        raise ValueError("Provide at least --automated-file or --probability-file.")

    automated_path = som_dir / "somnotate_automated.tsv"
    probability_path = som_dir / "somnotate_state_probabilities.npz"

    if automated_in:
        if args.copy_files:
            shutil.copy2(automated_in, automated_path)
        else:
            automated_path = automated_in

    if probability_in:
        if args.copy_files:
            shutil.copy2(probability_in, probability_path)
        else:
            probability_path = probability_in

    n = int(np.ceil(duration_s))
    t0 = np.arange(n, dtype=float)
    t1 = t0 + 1.0
    t_mid = (t0 + t1) / 2

    out = pd.DataFrame({
        "recording_id": args.recording_id,
        "t0_s": t0,
        "t1_s": t1,
        "time_min": t0 / 60,
    })

    if automated_in:
        intervals = parse_somnotate_automated(automated_path)

        if len(intervals) == 0:
            raise RuntimeError(f"Could not parse Somnotate automated file: {automated_path}")

        out["somnotate_state"] = state_at_times(intervals, t_mid)
    else:
        out["somnotate_state"] = "Undefined"

    if probability_in:
        z = np.load(probability_path, allow_pickle=True)
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
    print("Attached external Somnotate outputs.")
    print("Recording:", args.recording_id)
    print("Output:", out_path)

    if automated_in:
        print("Automated scoring:", automated_path)

    if probability_in:
        print("Probabilities:", probability_path)

    print()
    print("Somnotate state distribution:")
    print(out["somnotate_state"].value_counts(normalize=True).mul(100).round(2).to_string())

    prob_cols = [c for c in out.columns if c.startswith("somnotate_P_")]
    if prob_cols:
        print()
        print("Probability columns:")
        print(", ".join(prob_cols))


if __name__ == "__main__":
    main()
