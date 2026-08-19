from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTED_SOMNOTATE_COMMIT = "a20f33de62511d8c172e333896608b7fc166d0f0"
TESTED_SOMNOTATE_VERSION = "0.5.0"
EXPECTED_SOMNOTATE_RUNTIME = {
    "somnotate": "0.5.0",
    "pomegranate": "0.14.4",
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "matplotlib": "3.9.4",
    "scikit-learn": "1.6.1",
    "pandas": "2.3.3",
    "pyedflib": "0.1.42",
    "lspopt": "1.4.0",
}

APP_MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "dash",
    "plotly",
    "h5py",
    "pyedflib",
]

REQUIRED_FILES = [
    "dash_app/app.py",
    "pipelines/01_import_mat_recording.py",
    "pipelines/01_import_edf_recording.py",
    "pipelines/02_compute_epoch_features.py",
    "pipelines/03_layer1_emg_wake_sleep.py",
    "pipelines/10_somnotate_layer.py",
]


def _find_conda_python(env_name: str) -> Path | None:
    conda = shutil.which("conda")
    if conda:
        try:
            result = subprocess.run(
                [conda, "env", "list", "--json"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
            for prefix_text in json.loads(result.stdout).get("envs", []):
                prefix = Path(prefix_text)
                if prefix.name.lower() == env_name.lower():
                    candidate = prefix / ("python.exe" if os.name == "nt" else "bin/python")
                    if candidate.is_file():
                        return candidate
        except Exception:
            pass
    return None


def _somnotate_runtime_info(python_exe: Path) -> tuple[bool, str, dict[str, str]]:
    code = r'''
import json, sys
from importlib import metadata
names = ["somnotate", "pomegranate", "numpy", "scipy", "matplotlib", "scikit-learn", "pandas", "pyedflib", "lspopt"]
import platform
out = {"python": sys.version.split()[0], "machine": platform.machine()}
for name in names:
    try:
        out[name] = metadata.version(name)
    except Exception as exc:
        out[name] = "MISSING: " + repr(exc)
print(json.dumps(out, sort_keys=True))
'''
    p = subprocess.run(
        [str(python_exe), "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        return False, (p.stdout + "\n" + p.stderr).strip(), {}
    try:
        info = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return False, p.stdout.strip(), {}
    missing = [k for k, v in info.items() if isinstance(v, str) and v.startswith("MISSING:")]
    lines = [f"{k}={v}" for k, v in info.items()]
    return not missing, ", ".join(lines), {str(k): str(v) for k, v in info.items()}


def _git_head(repo: Path) -> str | None:
    git = shutil.which("git")
    if not git or not (repo / ".git").exists():
        return None
    try:
        p = subprocess.run(
            [git, "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return p.stdout.strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-platform setup diagnostics for Sleep Stage QC.")
    parser.add_argument("--somnotate-root", default=os.environ.get("SOMNOTATE_ROOT", ""))
    parser.add_argument("--somnotate-env", default="somnotate_env")
    parser.add_argument("--require-somnotate", action="store_true", help="Treat missing/broken Somnotate setup as a blocking failure.")
    args = parser.parse_args()

    print("=== Sleep Stage QC diagnostics ===")
    print("Platform:", platform.platform())
    print("Python:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "<not set>"))
    print("Repository:", ROOT)
    print("Working directory:", Path.cwd())

    failures: list[str] = []
    warnings: list[str] = []

    if sys.version_info[:2] != (3, 11):
        warnings.append(f"App environment is tested with Python 3.11; current Python is {sys.version_info.major}.{sys.version_info.minor}.")

    for name in APP_MODULES:
        try:
            module = importlib.import_module(name)
            print(f"[OK] {name}: {getattr(module, '__version__', 'version unavailable')}")
        except Exception as exc:
            failures.append(f"{name}: {exc!r}")
            print(f"[FAIL] {name}: {exc!r}")

    for rel in REQUIRED_FILES:
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
    if ffmpeg:
        print("[OK] ffmpeg:", ffmpeg)
    else:
        warnings.append("FFmpeg not found; local AVI conversion will not work.")
        print("[WARN] ffmpeg not found; AVI conversion will not work")

    print("\n=== Somnotate (optional) ===")
    som_root_text = str(args.somnotate_root or "").strip()
    if not som_root_text:
        candidate = Path.home() / "somnotate"
        if candidate.exists():
            som_root_text = str(candidate)

    som_failures: list[str] = []
    host_is_apple_silicon = sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    if host_is_apple_silicon:
        print("[INFO] Apple Silicon host detected. Somnotate's legacy pomegranate runtime should target osx-64/Rosetta.")
    if not som_root_text:
        msg = "Somnotate repository not supplied/found. Set SOMNOTATE_ROOT or pass --somnotate-root."
        print("[WARN]", msg)
        som_failures.append(msg)
    else:
        som_root = Path(som_root_text).expanduser().resolve()
        pipeline = som_root / "example_pipeline" / "01_preprocess_signals.py"
        if not pipeline.exists():
            msg = f"Somnotate example pipeline not found under: {som_root}"
            print("[FAIL]", msg)
            som_failures.append(msg)
        else:
            print("[OK] Somnotate repository:", som_root)
            head = _git_head(som_root)
            if head:
                if head == TESTED_SOMNOTATE_COMMIT:
                    print("[OK] Somnotate git commit:", head)
                else:
                    warnings.append(
                        f"Somnotate checkout is {head}; tested commit is {TESTED_SOMNOTATE_COMMIT}."
                    )
                    print("[WARN] Somnotate git commit:", head)
                    print("       Tested commit:", TESTED_SOMNOTATE_COMMIT)

        som_py = _find_conda_python(str(args.somnotate_env))
        if som_py is None:
            msg = f"Could not find Conda environment '{args.somnotate_env}'."
            print("[FAIL]", msg)
            som_failures.append(msg)
        else:
            print("[OK] Somnotate Python:", som_py)
            ok, details, runtime_info = _somnotate_runtime_info(som_py)
            print("[OK] Somnotate runtime:" if ok else "[FAIL] Somnotate runtime:", details)
            if not ok:
                som_failures.append(details)
            else:
                runtime_mismatches = []
                python_mm = ".".join(runtime_info.get("python", "").split(".")[:2])
                if python_mm != "3.9":
                    runtime_mismatches.append(f"python expected 3.9.x, found {runtime_info.get('python')}")
                for package, expected in EXPECTED_SOMNOTATE_RUNTIME.items():
                    found = runtime_info.get(package)
                    if found != expected:
                        runtime_mismatches.append(f"{package} expected {expected}, found {found}")
                if runtime_mismatches:
                    message = "Somnotate runtime differs from environment_somnotate.yml: " + "; ".join(runtime_mismatches)
                    print("[WARN]", message)
                    if args.require_somnotate:
                        som_failures.append(message)
                    else:
                        warnings.append(message)
                if host_is_apple_silicon and runtime_info.get("machine", "").lower() != "x86_64":
                    message = (
                        "Apple Silicon host detected but Somnotate runtime does not report x86_64. "
                        "Create it with: conda env create --platform osx-64 -f environment_somnotate.yml"
                    )
                    warnings.append(message)
                    print("[WARN]", message)

    if args.require_somnotate and som_failures:
        failures.extend(som_failures)

    print("\n=== Result ===")
    if warnings:
        print(f"WARNINGS: {len(warnings)}")
        for item in warnings:
            print(" -", item)
    if failures:
        print(f"FAILED: {len(failures)} blocking problem(s)")
        for item in failures:
            print(" -", item)
        return 1
    print("Environment looks ready.")
    if som_failures and not args.require_somnotate:
        print("Core app is ready; Somnotate is not fully configured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
