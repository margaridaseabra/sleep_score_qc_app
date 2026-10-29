"""Dissociation physiology: alignment, cautious comparisons and dashboard regression."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.dissociation_physiology import (
    align_epoch_features, align_photometry, robust_z, load_epoch_physiology,
    summarize_event_physiology,
)
from dash_app.dissociation_physiology_viz import (
    physiology_overview_figure, event_contrast_heatmap,
    event_detail_figure, event_metric_rows,
)

PIPELINE = Path(__file__).resolve().parents[1] / "pipelines" / "30_dissociation_analysis.py"
spec = importlib.util.spec_from_file_location("dissociation_pipeline_with_physiology", PIPELINE)
diss = importlib.util.module_from_spec(spec)
# Script-mode import of its sibling module is explicit here because pytest
# imports the file as a module rather than calling python pipelines/30_...py.
import sys
sys.path.insert(0, str(PIPELINE.parent))
spec.loader.exec_module(diss)


def test_epoch_features_use_time_not_untrusted_epoch_ids():
    epochs = pd.DataFrame({"t0_s": [0, 2, 4], "t1_s": [2, 4, 6]})
    features = pd.DataFrame({
        "epoch_id": [999, 998, 997, 996, 995, 994],
        "t0_s": [0, 1, 2, 3, 4, 5], "t1_s": [1, 2, 3, 4, 5, 6],
        "eeg_delta": [1, 3, 5, 7, 9, 11],
        "emg_rms": [2, 4, 6, 8, 10, 12],
        "eeg_theta_delta_ratio": [1, 1, 2, 2, 3, 3],
    })
    aligned = align_epoch_features(epochs, features)
    assert aligned["eeg_delta"].tolist() == [2, 6, 10]
    assert aligned["emg_rms"].tolist() == [3, 7, 11]
    assert aligned["eeg_theta_delta_ratio"].tolist() == [1, 2, 3]


def test_partial_coverage_is_not_silently_extrapolated():
    epochs = pd.DataFrame({"t0_s": [0, 2], "t1_s": [2, 4]})
    features = pd.DataFrame({"t0_s": [0], "t1_s": [1],
                             "eeg_delta": [200]})
    aligned = align_epoch_features(epochs, features)
    assert aligned["eeg_delta"].isna().all()


def test_photometry_uses_its_own_fs_and_raw_scale(tmp_path):
    epochs = pd.DataFrame({"t0_s": [0, 2, 4, 6, 8],
                           "t1_s": [2, 4, 6, 8, 10]})
    np.save(tmp_path / "ach.npy", np.arange(20, dtype=np.float32))
    means, stds, info = align_photometry(
        epochs, tmp_path,
        {"ach_file": "ach.npy", "ach_sampling_rate_hz": 2.0,
         "ach_key": "ne", "ne_units": "volts"},
    )
    assert means.tolist() == pytest.approx([1.5, 5.5, 9.5, 13.5, 17.5])
    assert stds.tolist() == pytest.approx([np.std(np.arange(k, k + 4)) for k in range(0, 20, 4)])
    assert info["source"] == "ne"
    assert info["units"] == "volts"
    assert info["sampling_rate_hz"] == 2.0


def test_explicit_photometry_onset_aligns_sample_time(tmp_path):
    epochs = pd.DataFrame({"t0_s": [0., 2., 4.], "t1_s": [2., 4., 6.]})
    # Auxiliary signal starts at EEG t=2s, lasts until t=6s (2 Hz).
    np.save(tmp_path / "ach.npy", np.arange(8, dtype=np.float32))
    mean, _, info = align_photometry(
        epochs, tmp_path,
        {"ach_file": "ach.npy", "ach_sampling_rate_hz": 2,
         "photometry_offset_s": 2},
    )
    assert np.isnan(mean[0])
    assert mean[1:].tolist() == pytest.approx([1.5, 5.5])
    assert info["onset_s"] == 2.0


def test_photometry_missing_fs_and_wrong_time_base_are_not_assumed(tmp_path):
    epochs = pd.DataFrame({"t0_s": [0, 1, 2, 3], "t1_s": [1, 2, 3, 4]})
    np.save(tmp_path / "ach.npy", np.arange(10, dtype=np.float32))
    means, _, info = align_photometry(epochs, tmp_path, {"ach_file": "ach.npy"})
    assert np.isnan(means).all()
    assert "sampling rate" in info["status"]
    means, _, info = align_photometry(
        epochs, tmp_path, {"ach_file": "ach.npy", "ach_sampling_rate_hz": 1.0}
    )
    assert np.isnan(means).all()
    assert "time base unverified" in info["status"]


def test_unavailable_photometry_does_not_block_eeg_emg(tmp_path):
    epochs = pd.DataFrame({"t0_s": np.arange(12.), "t1_s": np.arange(12.) + 1})
    pd.DataFrame({
        "t0_s": np.arange(12.), "t1_s": np.arange(12.) + 1,
        "eeg_delta": np.arange(1, 13), "emg_rms": np.arange(1, 13),
        "emg_rms_z": np.arange(12.) / 10,
    }).to_csv(tmp_path / "epoch_features.csv", index=False)
    (tmp_path / "metadata.json").write_text("{}")
    data, info = load_epoch_physiology(tmp_path, epochs)
    assert data["eeg_delta"].notna().all()
    assert data["emg_rms_z"].notna().all()
    assert data["photometry_mean_z"].isna().all()
    assert not info["photometry"]["available"]


def test_event_profile_excludes_events_and_uses_same_state_background():
    n = 30
    epochs = pd.DataFrame({
        "t0_s": np.arange(n, dtype=float),
        "t1_s": np.arange(n, dtype=float) + 1,
        "somnotate_state": ["NREM"] * 20 + ["REM"] * 10,
        "final_state": ["Undefined"] * n,
        "dissociation_index": [0.0] * 2 + [0.9, 0.9] + [0.0] * 26,
        "eeg_delta_z": [0.0] * 2 + [3., 5.] + [0.0] * 26,
        "eeg_theta_delta_ratio_z": [0.1] * n,
        "eeg_sigma_z": [0.0] * n,
        "emg_rms_z": [0.5] * n,
        "emg_abs_p95_z": [0.0] * n,
        "photometry_mean_z": np.arange(n, dtype=float) / 10,
    })
    events = pd.DataFrame([{
        "event_id": "e1", "core_start_s": 2., "core_end_s": 4.,
        "start_s": 0., "end_s": 14., "max_dissociation_index": 0.9,
        "rank": 1,
    }])
    profiles = summarize_event_physiology(epochs, events, threshold=0.2)
    profile = profiles.iloc[0]
    assert profile["n_physiology_epochs"] == 2
    assert profile["reference_source"] == "somnotate_state"
    assert profile["reference_state"] == "NREM"
    assert profile["reference_n_epochs"] == 18
    assert profile["event_eeg_delta_z"] == 4
    assert profile["background_eeg_delta_z"] == 0
    assert profile["delta_eeg_delta_z"] == 4

    overview = physiology_overview_figure(epochs, events)
    contrast = event_contrast_heatmap(profiles, events)
    detail = event_detail_figure(epochs, events.iloc[0])
    assert overview is not None
    assert contrast is not None
    assert detail is not None
    assert any(x["Feature"].startswith("Delta") for x in event_metric_rows(profile))


def test_tiny_same_state_reference_must_not_create_contrast():
    n = 8
    epochs = pd.DataFrame({
        "t0_s": np.arange(n, dtype=float), "t1_s": np.arange(n, dtype=float) + 1,
        "somnotate_state": ["REM"] * n,
        "dissociation_index": [0.8] * 2 + [0.] * 6,
        "emg_rms_z": [3.] * 2 + [0.] * 6,
    })
    events = pd.DataFrame([{
        "event_id": "r1", "core_start_s": 0., "core_end_s": 2.,
    }])
    profile = summarize_event_physiology(epochs, events).iloc[0]
    assert profile["reference_n_epochs"] == 6
    assert np.isnan(profile["delta_emg_rms_z"])


def test_pipeline_outputs_event_physiology_and_provenance(tmp_path):
    rec = tmp_path / "recordings" / "m1"
    (rec / "somnotate").mkdir(parents=True)
    n = 16
    t = np.arange(n, dtype=float)
    pd.DataFrame({
        "t0_s": t, "t1_s": t + 1,
        "layer1_label": ["Sleep"] * n,
    }).to_csv(rec / "layer1_wake_sleep.csv", index=False)
    pd.DataFrame({
        "t0_s": t, "t1_s": t + 1,
        "somnotate_state": ["REM"] * n,
        "somnotate_confidence": [0.4] + [0.98] * (n - 1),
    }).to_csv(rec / "somnotate" / "somnotate_results_timeseries.csv", index=False)
    pd.DataFrame({
        "t0_s": t, "t1_s": t + 1,
        "epoch_id": np.arange(n) + 100,  # deliberately different identifiers
        "eeg_delta": np.arange(1., n + 1),
        "eeg_theta_delta_ratio": np.linspace(1, 3, n),
        "emg_rms": np.arange(1., n + 1),
        "emg_rms_z": [3.] + [0.] * (n - 1),
    }).to_csv(rec / "epoch_features.csv", index=False)
    (rec / "metadata.json").write_text(json.dumps({"sampling_rate_hz": 100}))
    diss.build_analysis(tmp_path, "m1", evidence="combined", threshold=0.2)
    output = rec / "dissociation_analysis"
    assert (output / "dissociation_physiology_info.json").exists()
    p = pd.read_csv(output / "dissociation_event_physiology.csv")
    assert len(p) >= 1
    events = pd.read_csv(output / "dissociation_events.csv")
    assert "event_eeg_delta" in events
    epoch = pd.read_csv(output / "dissociation_epochs.csv")
    assert epoch["eeg_delta"].tolist() == list(np.arange(1., n + 1))
    assert epoch["photometry_mean_raw"].isna().all()
