"""Regression tests for biologically interpretable dissociation analysis."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dash_app.dissociation_viz import (
    confusion_figure, score_timeline_figure, hourly_burden_figure,
    event_raster_figure, final_coverage_figure,
)
from dash_app.app import (
    attach_dissociation_review_status, persist_dissociation_review_status,
)

PIPELINE = Path(__file__).resolve().parents[1] / "pipelines" / "30_dissociation_analysis.py"
spec = importlib.util.spec_from_file_location("dissociation_pipeline", PIPELINE)
diss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diss)


def _recording(tmp_path, layers, som_state, conf=None, emg=None):
    folder = tmp_path / "recordings" / "sample"
    (folder / "somnotate").mkdir(parents=True)
    n = len(layers)
    start = np.arange(n, dtype=float) * 5
    geometry = {"t0_s": start, "t1_s": start + 5}
    pd.DataFrame({**geometry, "layer1_label": layers}).to_csv(
        folder / "layer1_wake_sleep.csv", index=False
    )
    som = pd.DataFrame({**geometry, "somnotate_state": som_state})
    if conf is not None:
        som["somnotate_confidence"] = conf
    som.to_csv(folder / "somnotate" / "somnotate_results_timeseries.csv", index=False)
    if emg is not None:
        pd.DataFrame({"epoch_id": list(range(n)), "emg_rms_z": emg}).to_csv(
            folder / "epoch_features.csv", index=False
        )
    return folder


def _outputs(folder):
    directory = folder / "dissociation_analysis"
    return (
        pd.read_csv(directory / "dissociation_epochs.csv"),
        pd.read_csv(directory / "dissociation_events.csv"),
        pd.read_csv(directory / "dissociation_pairwise_summary.csv"),
    )


def test_sleep_is_not_mislabeled_nrem_and_rem_is_not_false_conflict(tmp_path):
    folder = _recording(
        tmp_path, ["Sleep", "Wake", "Sleep", "Undefined"],
        ["REM", "REM", "NREM", "Undefined"]
    )
    diss.build_analysis(tmp_path, "sample", evidence="disagreement", threshold=0.20)
    epochs, events, summary = _outputs(folder)

    # A binary Sleep prediction agrees with either NREM or REM after collapsing.
    assert epochs["layer1_vs_somnotate_ws"].tolist() == [False, True, False, False]
    assert epochs["layer1_3state"].tolist() == ["Undefined", "Wake", "Undefined", "Undefined"]
    assert "layer1_vs_somnotate_3state" not in epochs
    assert len(events) == 1
    assert events.iloc[0]["core_start_s"] == 5
    assert events.iloc[0]["core_end_s"] == 10
    assert events.iloc[0]["duration_s"] == 5
    assert events.iloc[0]["review_duration_s"] == 20  # 10 s padded each side, clipped at 0.
    assert events.iloc[0]["start_s"] == 0
    assert events.iloc[0]["end_s"] == 20

    ws = summary.loc[summary["pair"] == "Layer 1 vs Somnotate Wake/Sleep"].iloc[0]
    assert ws["n_comparable_epochs"] == 3
    assert ws["n_disagree_epochs"] == 1
    # Missing labels are a coverage issue, never a scored biological conflict.
    assert bool(epochs.iloc[3]["missing_model_state"])
    assert epochs.iloc[3]["dissociation_index"] == 0
    assert not bool(epochs.iloc[3]["evidence_available"])


def test_evidence_mode_and_min_duration_do_not_create_phantom_events(tmp_path):
    folder = _recording(
        tmp_path, ["Sleep", "Sleep", "Wake"], ["REM", "NREM", "REM"],
        conf=[0.98, 0.99, 0.99], emg=[3.1, 0.0, 0.0],
    )
    diss.build_analysis(tmp_path, "sample", evidence="physiology",
                        threshold=0.2, min_duration_s=0)
    epochs, events, _ = _outputs(folder)
    assert epochs["rem_high_emg"].tolist() == [True, False, False]
    assert len(events) == 1
    assert events.iloc[0]["main_reason"] == "REM_with_high_EMG"
    assert events.iloc[0]["duration_s"] == 5
    settings = json.loads((folder / "dissociation_analysis" / "analysis_settings.json").read_text())
    assert settings["evidence"] == "physiology"

    diss.build_analysis(tmp_path, "sample", evidence="physiology",
                        threshold=0.2, min_duration_s=10)
    _, events, _ = _outputs(folder)
    assert events.empty
    assert "core_start_s" in events.columns
    assert "rank" in events.columns


def test_dissociation_plot_helpers_use_comparable_epochs_and_core_times():
    epochs = pd.DataFrame({
        "t0_s": [0., 5., 10., 15.],
        "dissociation_index": [0.1, 0.8, 0.3, 0.0],
        "evidence_available": [True, True, True, False],
        "somnotate_state": ["REM", "REM", "NREM", "Undefined"],
        "final_state": ["REM", "Wake", "Undefined", "Undefined"],
    })
    events = pd.DataFrame({
        "event_id": ["dissociation_00000"], "start_s": [0.], "end_s": [20.],
        "core_start_s": [5.], "core_end_s": [10.],
        "max_dissociation_index": [0.8],
        "main_reason": ["final_vs_somnotate_3state"],
    })
    chart = confusion_figure(epochs, "somnotate_state", "final_state", "Somnotate", "Final")
    # REM -> Wake and REM -> REM are both present, but Undefined Final excluded.
    assert chart is not None
    assert np.asarray(chart.data[0]["z"]).sum() == 100
    assert chart.data[0]["customdata"][2][0] == 1  # REM -> Wake
    assert score_timeline_figure(epochs, events) is not None
    assert hourly_burden_figure(epochs) is not None
    assert event_raster_figure(events) is not None
    assert final_coverage_figure(epochs) is not None


def test_review_decisions_are_persisted_without_changing_scoring(tmp_path):
    folder = _recording(tmp_path, ["Sleep"], ["REM"])
    diss.build_analysis(tmp_path, "sample", evidence="confidence", threshold=0.2)
    # Confidence mode has no available confidence; create a known event for this test.
    analysis = folder / "dissociation_analysis"
    pd.DataFrame([{
        "event_id": "event_a", "start_s": 0., "end_s": 25.,
        "core_start_s": 5., "core_end_s": 15.,
    }]).to_csv(analysis / "dissociation_events.csv", index=False)
    event = pd.read_csv(analysis / "dissociation_events.csv")
    assert attach_dissociation_review_status(event, analysis)["qc_review_status"].iloc[0] == "pending"
    persist_dissociation_review_status(tmp_path, "sample", "event_a", "reviewed")
    assert attach_dissociation_review_status(event, analysis)["qc_review_status"].iloc[0] == "reviewed"
    persist_dissociation_review_status(tmp_path, "sample", "event_a", "pending")
    assert attach_dissociation_review_status(event, analysis)["qc_review_status"].iloc[0] == "pending"
    assert not (folder / "final_scoring.csv").exists()
