from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from dash_app.app import (
    normalize_final_scoring_dtypes,
    normalize_state_label,
    recording_dir_from_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("somnotate_layer", ROOT / "pipelines" / "10_somnotate_layer.py")
assert SPEC and SPEC.loader
som_layer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(som_layer)


def test_state_normalization_accepts_current_somnotate_labels():
    assert normalize_state_label("awake") == "Wake"
    assert normalize_state_label("non-REM") == "NREM"
    assert normalize_state_label("REM") == "REM"
    assert som_layer.normalize_state("awake") == "Wake"
    assert som_layer.normalize_state("non-REM") == "NREM"


def test_final_text_columns_are_writable_after_csv_style_inference():
    df = pd.DataFrame(
        {
            "recording_id": ["r1", "r1"],
            "epoch_id": [0, 1],
            "t0_s": [0.0, 1.0],
            "t1_s": [1.0, 2.0],
            "final_state": ["Undefined", "Undefined"],
            "final_source": [float("nan"), float("nan")],
            "review_status": [float("nan"), float("nan")],
            "review_notes": [float("nan"), float("nan")],
        }
    )
    out = normalize_final_scoring_dtypes(df)
    out.loc[:, "review_notes"] = "reviewed on Windows"
    out.loc[:, "final_source"] = "dash_accept_somnotate"
    assert out["review_notes"].tolist() == ["reviewed on Windows", "reviewed on Windows"]


def test_recording_resolution_prefers_current_project_root(tmp_path: Path):
    project = tmp_path / "project"
    canonical = project / "recordings" / "mouse01"
    canonical.mkdir(parents=True)
    pd.DataFrame(
        [{"recording_id": "mouse01", "recording_dir": "/Volumes/T7/old/project/recordings/mouse01"}]
    ).to_csv(project / "recordings_manifest.csv", index=False)
    assert recording_dir_from_manifest(project, "mouse01") == canonical.resolve()


def test_somnotate_configuration_patch_is_two_channel(tmp_path: Path):
    config = tmp_path / "configuration.py"
    config.write_text(
        """
state_annotation_signals = [
    'frontal_eeg_signal_label',
    'occipital_eeg_signal_label',
    'emg_signal_label',
]
state_annotation_signal_labels = [
    'frontal EEG',
    'occipital EEG',
    'EMG',
]
state_annotation_signal_frequency_bands = [
    (0.5, 30.),
    (0.5, 30.),
    (10., 45.),
]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    som_layer.patch_configuration_signals(config)
    text = config.read_text(encoding="utf-8")
    assert "occipital_eeg_signal_label" not in text
    assert "'frontal_eeg_signal_label'" in text
    assert "'emg_signal_label'" in text


def test_lspopt_nperseg_patch_casts_to_int(tmp_path: Path):
    script = tmp_path / "01_preprocess_signals.py"
    script.write_text(
        "frequencies, time, spectrogram = spectrogram_lspopt(raw_signal, nperseg  = sampling_frequency_in_hz * time_resolution_in_sec,)\n",
        encoding="utf-8",
    )
    som_layer.patch_preprocess_integer_nperseg(script)
    text = script.read_text(encoding="utf-8")
    assert "int(round(sampling_frequency_in_hz * time_resolution_in_sec))" in text


def test_somnotate_manifest_paths_are_rebased(tmp_path: Path):
    project = tmp_path / "project"
    som_dir = project / "recordings" / "mouse01" / "somnotate"
    som_dir.mkdir(parents=True)
    row = {
        "file_path_raw_signals": "/Volumes/T7/old/somnotate_input_5s.edf",
        "file_path_preprocessed_signals": "/Volumes/T7/old/preprocessed.npy",
        "sampling_frequency_in_hz": 512.0,
    }
    fixed = som_layer._rebase_somnotate_manifest_row(project, "mouse01", row, 5.0)
    assert Path(fixed["file_path_raw_signals"]) == som_dir / "somnotate_input_5s.edf"
    assert Path(fixed["file_path_preprocessed_signals"]) == som_dir / "somnotate_preprocessed_5s.npy"
