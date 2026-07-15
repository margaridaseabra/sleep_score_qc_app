\
from __future__ import annotations
import importlib
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
print("=== Sleep Stage QC diagnostics ===")
print("Platform:", platform.platform())
print("Python:", sys.executable)
print("Python version:", sys.version.replace("\n", " "))
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "<not set>"))
print("Repository:", ROOT)
print("Working directory:", Path.cwd())

failures = []
for name in ["numpy", "pandas", "scipy", "sklearn", "dash", "plotly", "h5py", "pyedflib"]:
    try:
        module = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(module, '__version__', 'version unavailable')}")
    except Exception as exc:
        failures.append(f"{name}: {exc!r}")
        print(f"[FAIL] {name}: {exc!r}")

for rel in ["dash_app/app.py", "pipelines/01_import_mat_recording.py", "pipelines/02_compute_epoch_features.py", "pipelines/03_layer1_emg_wake_sleep.py"]:
    path = ROOT / rel
    if path.exists():
        print("[OK] file:", path)
    else:
        failures.append(f"Missing file: {path}")
        print("[FAIL] missing:", path)

try:
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        test = Path(tmp) / "write_test.txt"
        test.write_text("ok", encoding="utf-8")
        assert test.read_text(encoding="utf-8") == "ok"
    print("[OK] Repository is writable")
except Exception as exc:
    failures.append(f"Repository write test: {exc!r}")
    print("[FAIL] Repository is not writable:", repr(exc))

ffmpeg = shutil.which("ffmpeg")
print("[OK] ffmpeg:", ffmpeg) if ffmpeg else print("[WARN] ffmpeg not found; AVI conversion will not work")

print("\n=== Result ===")
if failures:
    print(f"FAILED: {len(failures)} blocking problem(s)")
    for item in failures:
        print(" -", item)
    raise SystemExit(1)
print("Environment looks ready.")
