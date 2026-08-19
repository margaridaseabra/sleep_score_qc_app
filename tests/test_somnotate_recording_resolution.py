from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("somnotate_layer_paths", ROOT / "pipelines" / "10_somnotate_layer.py")
assert SPEC and SPEC.loader
som_layer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(som_layer)


def test_pipeline_recording_resolution_falls_back_to_manifest(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external_recordings" / "mouse02"
    external.mkdir(parents=True)
    for name in ("metadata.json", "eeg.npy", "emg.npy"):
        (external / name).write_bytes(b"x")

    pd.DataFrame(
        [{"recording_id": "mouse02", "recording_dir": str(external)}]
    ).to_csv(project / "recordings_manifest.csv", index=False)

    resolved = som_layer.resolve_recording_dir(
        project, "mouse02", required_files=("metadata.json", "eeg.npy", "emg.npy")
    )
    assert resolved == external.resolve()


def test_pipeline_skips_incomplete_canonical_and_uses_valid_manifest_path(tmp_path: Path):
    project = tmp_path / "project"
    incomplete = project / "recordings" / "mouse03"
    incomplete.mkdir(parents=True)
    (incomplete / "metadata.json").write_text("{}", encoding="utf-8")

    external = tmp_path / "external_recordings" / "mouse03"
    external.mkdir(parents=True)
    for name in ("metadata.json", "eeg.npy", "emg.npy"):
        (external / name).write_bytes(b"x")

    pd.DataFrame(
        [{"recording_id": "mouse03", "recording_dir": str(external)}]
    ).to_csv(project / "recordings_manifest.csv", index=False)

    resolved = som_layer.resolve_recording_dir(
        project, "mouse03", required_files=("metadata.json", "eeg.npy", "emg.npy")
    )
    assert resolved == external.resolve()
