from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATE_COL = "somnotate_state"
PROB_COLS = ["somnotate_P_Wake", "somnotate_P_NREM", "somnotate_P_REM"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two imported Somnotate result CSVs, e.g. Windows vs macOS.")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    a = pd.read_csv(args.first)
    b = pd.read_csv(args.second)
    if len(a) != len(b):
        print(f"FAIL: row count differs: {len(a)} vs {len(b)}")
        return 1

    failures = 0
    if STATE_COL in a.columns and STATE_COL in b.columns:
        sa = a[STATE_COL].fillna("Undefined").astype(str).to_numpy()
        sb = b[STATE_COL].fillna("Undefined").astype(str).to_numpy()
        mismatches = int(np.sum(sa != sb))
        print(f"State mismatches: {mismatches} / {len(a)}")
        failures += int(mismatches > 0)
    else:
        print("WARN: state column missing from one file")

    for col in PROB_COLS:
        if col not in a.columns or col not in b.columns:
            print(f"WARN: probability column missing: {col}")
            continue
        x = pd.to_numeric(a[col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(b[col], errors="coerce").to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            print(f"WARN: no finite values for {col}")
            continue
        diff = np.abs(x[mask] - y[mask])
        print(f"{col}: mean abs diff={diff.mean():.12g}, max abs diff={diff.max():.12g}")
        failures += int(diff.max() > args.atol)

    if failures:
        print("RESULT: DIFFERENCES FOUND")
        return 1
    print("RESULT: IDENTICAL WITHIN TOLERANCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
