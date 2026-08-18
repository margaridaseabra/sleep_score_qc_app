from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("somnotate_layer_qc", ROOT / "pipelines" / "10_somnotate_layer.py")
assert SPEC and SPEC.loader
som_layer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(som_layer)


def test_manual_export_uses_somnotate_visbrain_format(tmp_path: Path):
    manual_csv = tmp_path / "manual.csv"
    out_file = tmp_path / "manual.tsv"
    pd.DataFrame(
        {
            "t0_s": [0.0, 5.0, 10.0, 15.0],
            "t1_s": [5.0, 10.0, 15.0, 20.0],
            "manual_state": ["Wake", "Wake", "NREM", "REM"],
        }
    ).to_csv(manual_csv, index=False)

    som_layer.export_manual_for_somnotate(manual_csv, out_file)
    lines = out_file.read_text(encoding="utf-8").splitlines()

    assert lines[0].startswith("*Duration_sec\t20")
    assert lines[1] == "*Datafile\tUnspecified"
    assert lines[2].startswith("awake\t10")
    assert lines[3].startswith("non-REM\t15")
    assert lines[4].startswith("REM\t20")


def test_quality_report_computes_recording_and_state_metrics(tmp_path: Path):
    manual1 = tmp_path / "m1.tsv"
    manual2 = tmp_path / "m2.tsv"
    content = "*Duration_sec\t15\n*Datafile\tUnspecified\nawake\t5\nnon-REM\t10\nREM\t15\n"
    manual1.write_text(content, encoding="utf-8")
    manual2.write_text(content, encoding="utf-8")

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "recording_id": ["m1", "m2"],
            "file_path_manual_state_annotation": [str(manual1), str(manual2)],
        }
    ).to_csv(manifest, index=False)

    npz = tmp_path / "qc.npz"
    confusion = np.array(
        [
            [[8, 1, 1], [1, 8, 1], [0, 1, 9]],
            [[9, 1, 0], [2, 7, 1], [1, 0, 9]],
        ],
        dtype=float,
    )
    np.savez(npz, accuracy=np.array([0.83, 0.86]), confusion=confusion)

    report = som_layer.quality_report_from_npz(npz, manifest, "leave-one-recording-out")
    assert report["n_recordings"] == 2
    assert abs(report["mean_accuracy"] - 0.845) < 1e-9
    assert report["worst_accuracy"] == 0.83
    assert report["confusion_labels"] == ["Wake", "NREM", "REM"]
    assert {row["state"] for row in report["per_state"]} == {"Wake", "NREM", "REM"}
    assert report["balanced_accuracy"] is not None
    assert report["macro_f1"] is not None
