from __future__ import annotations

import argparse
import importlib.util
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_FILE = ROOT / "pipelines" / "10_somnotate_layer.py"


def load_integration_module():
    spec = importlib.util.spec_from_file_location("sleep_qc_somnotate_layer", PIPELINE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PIPELINE_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that Sleep Stage QC compatibility patches still apply to a Somnotate checkout."
    )
    parser.add_argument("somnotate_root")
    args = parser.parse_args()

    som_root = Path(args.somnotate_root).expanduser().resolve()
    source = som_root / "example_pipeline"
    required = [source / "configuration.py", source / "01_preprocess_signals.py"]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing Somnotate source files: " + ", ".join(map(str, missing)))

    integration = load_integration_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pipeline = Path(tmp) / "example_pipeline"
        shutil.copytree(source, tmp_pipeline)
        integration.patch_configuration_time_resolution(tmp_pipeline / "configuration.py", 5.0)
        integration.patch_configuration_signals(tmp_pipeline / "configuration.py")
        integration.patch_preprocess_integer_nperseg(tmp_pipeline / "01_preprocess_signals.py")

        config_text = (tmp_pipeline / "configuration.py").read_text(encoding="utf-8")
        preprocess_text = (tmp_pipeline / "01_preprocess_signals.py").read_text(encoding="utf-8")
        if "occipital_eeg_signal_label" in config_text:
            raise RuntimeError("Two-channel compatibility patch did not remove occipital EEG.")
        if "'frontal_eeg_signal_label'" not in config_text or "'emg_signal_label'" not in config_text:
            raise RuntimeError("Two-channel compatibility patch did not retain EEG + EMG.")
        if "time_resolution = 5" not in config_text:
            raise RuntimeError("Epoch/time-resolution patch did not apply.")
        if "int(round(sampling_frequency_in_hz * time_resolution_in_sec))" not in preprocess_text:
            raise RuntimeError("Integer nperseg compatibility patch did not apply.")

    print(f"Somnotate source contract OK: {som_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
