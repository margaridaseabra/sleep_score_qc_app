# Sleep Stage QC

**Version 0.9.0-rc1**

Cross-platform Dash app for semi-automated EEG/EMG sleep scoring, quality control, manual review, Somnotate integration, dissociation analysis, synchronized video review, and export.

> Current app: **Dash** (`http://127.0.0.1:8050`)
> Legacy Streamlit code is kept only for reference and is not part of the supported setup.

## What the app does

1. Import MATLAB (`.mat`) or EDF/BDF recordings.
2. Compute epoch features and Layer 1 Wake/Sleep scoring.
3. Run/import Somnotate Wake/NREM/REM scoring.
4. Review EEG, EMG, spectrogram, optional photometry, probabilities, and video.
5. Apply manual or source labels to selected intervals or the visible window.
6. Review dissociation events and disagreements between scoring layers.
7. Export reviewed scoring to CSV, MATLAB, and EDF+.

The main scoring layers are:

| Layer | Meaning |
|---|---|
| Layer 1 | Automatic Wake/Sleep |
| Somnotate | Automatic Wake/NREM/REM |
| Manual | Imported manual scoring, if available |
| Final | Reviewed scoring created by the user |

**Final starts as `Undefined` and is filled only when the user explicitly accepts or edits labels.**

---

# 1. Install the main app

The Dash app and Somnotate intentionally use **separate Conda environments**.

```bash
git clone https://github.com/margaridaseabra/sleep_score_qc_app.git
cd sleep_score_qc_app
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
```

If the app environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate sleep_stage_qc_v2
```

The supported app runtime is Python 3.11. `environment.yml` constrains major package versions to avoid accidental breaking upgrades. The separate Somnotate environment is pinned more tightly because preprocessing/model persistence is scientifically version-sensitive.

## Windows

Use **Anaconda Prompt**:

```bat
cd C:\path\to\sleep_score_qc_app
python check_setup.py
run_app_windows.bat
```

or:

```bat
python -m dash_app.app
```

## macOS

Use Terminal:

```bash
cd /path/to/sleep_score_qc_app
python check_setup.py
./run_app.sh
```

or:

```bash
python -m dash_app.app
```

Open:

```text
http://127.0.0.1:8050
```

---

# 2. Install Somnotate

Somnotate is an external project and is not installed into the Dash environment.

This app version is tested against:

```text
Somnotate 0.5.0
commit a20f33de62511d8c172e333896608b7fc166d0f0
```

Clone and pin that checkout:

```bash
git clone https://github.com/paulbrodersen/somnotate.git
cd somnotate
git checkout a20f33de62511d8c172e333896608b7fc166d0f0
```

Create the separate Somnotate environment from the app repository:

```bash
conda env create -f environment_somnotate.yml
conda activate somnotate_env
```

Then install the cloned Somnotate code **without allowing pip to replace the Conda-pinned dependencies**:

```bash
cd /path/to/somnotate
python -m pip install -e . --no-deps
```

Run the full setup check:

```bash
cd /path/to/sleep_score_qc_app
conda activate sleep_stage_qc_v2
python check_setup.py --require-somnotate --somnotate-root /path/to/somnotate
```

On Windows, for example:

```bat
python check_setup.py --require-somnotate --somnotate-root C:\Users\YOUR_USER\somnotate
```

### Apple Silicon Macs (M1/M2/M3/M4/...)

The legacy `pomegranate 0.14.x` Conda packages used by Somnotate are not available natively for `osx-arm64`. Create the Somnotate environment as an Intel (`osx-64`) environment and let macOS run it through Rosetta:

```bash
conda env create --platform osx-64 -f environment_somnotate.yml
conda activate somnotate_env
python -m pip install -e /path/to/somnotate --no-deps
```

The main Dash environment should remain native to the Mac; only the separate Somnotate subprocess environment needs this workaround. `python check_setup.py --require-somnotate --somnotate-root /path/to/somnotate` reports the detected runtimes.

## Why Somnotate has its own environment

Somnotate 0.5.0 uses the legacy dependency `pomegranate=0.14.4`. The working Windows build requires Python 3.9, while the Dash app uses Python 3.11. `environment_somnotate.yml` pins the validated scientific runtime rather than allowing package upgrades to drift silently. Keeping the two runtimes separate prevents dependency conflicts.

The app calls the Somnotate Python executable as a subprocess; it does not import Somnotate into the Dash process.

---

# 3. Somnotate compatibility layer

The app uses a **temporary copy** of Somnotate's `example_pipeline` for each run. The original Somnotate checkout is never modified.

The temporary copy is configured to:

- use this app's two-channel **EEG + EMG** workflow;
- use the selected 1 s, 2 s, or 5 s Somnotate epoch length;
- normalize current Somnotate labels such as `awake` and `non-REM` to the app labels `Wake` and `NREM`;
- apply a compatibility fix that ensures the spectrogram window length passed to `lspopt` is an integer;
- repair generated Somnotate paths when a project is moved between macOS and Windows.

If upstream Somnotate changes, use the tested commit above until this integration has been revalidated.

---

# 4. Models and reproducibility

Somnotate models are epoch-specific. A 1 s model must be used with 1 s Somnotate preprocessing, a 2 s model with 2 s preprocessing, and a 5 s model with 5 s preprocessing.

Models trained from the app now receive a neighboring `.metadata.json` file recording:

- app version;
- Somnotate epoch length;
- target sampling rate;
- EEG/EMG signal configuration;
- training/test recording IDs;
- Somnotate Git commit;
- Python, Somnotate, scikit-learn, pomegranate, NumPy, SciPy, pandas, pyEDFlib and lspopt versions.

The pipeline blocks known runtime mismatches for newly trained models unless the advanced `--allow-version-mismatch` flag is explicitly used.

### Bundled legacy models

The two historical model files currently bundled with the repository are identical legacy copies and were observed to contain a scikit-learn estimator serialized with **scikit-learn 1.7.2**. The standardized Somnotate environment uses scikit-learn 1.6.1 because of the Python 3.9/pomegranate constraint.

They are retained for backward compatibility and app testing, but a **new model trained/validated in the standardized release environment should replace them before a final scientific v1.0 release**. See `COMPATIBILITY.md`.

> **Model security:** Somnotate models are Python pickle files. Only load model files from a trusted source; unpickling an untrusted file can execute arbitrary code.

---

# 5. Project folder structure

```text
project_root/
├── recordings_manifest.csv
├── somnotate_runs/
└── recordings/
    └── recording_id/
        ├── metadata.json
        ├── eeg.npy
        ├── emg.npy
        ├── ach.npy                         # optional
        ├── epoch_features.csv
        ├── layer1_wake_sleep.csv
        ├── manual_scoring_aligned.csv      # optional
        ├── final_scoring.csv
        └── somnotate/
            ├── somnotate_input_5s.edf
            ├── somnotate_preprocessed_5s.npy
            ├── somnotate_automated_5s.tsv
            ├── somnotate_state_probabilities_5s.npz
            └── somnotate_results_timeseries.csv
```

Recording folders are resolved relative to the loaded project root. This allows a project copied from `/Volumes/...` on macOS to `E:\...` on Windows to continue working without manually editing `recordings_manifest.csv`.

Raw source paths are kept only as provenance. Large recordings, generated arrays, videos, logs, and local project data are excluded from Git.

---

# 6. Typical workflow

## Import + Layer 1

1. Load/create a project root.
2. Open **1. Import .mat / EDF + Layer 1**.
3. Enter the recording path and ID.
4. Import the recording.
5. Compute epoch features.
6. Run Layer 1.

## Somnotate

1. Open **3. Somnotate**.
2. Set the Somnotate repository path, e.g. `C:\Users\name\somnotate` or `/Users/name/somnotate`.
3. Keep the Conda environment as `somnotate_env`, or provide the full Somnotate Python executable.
4. Select the epoch length that matches the model.
5. Select an existing model or train a new one.
6. Run the existing-model workflow with `prepare`, `preprocess`, `score`, `probabilities`, and `import-results` selected.

The app validates the repository/model paths before starting and performs a runtime preflight before regenerating EDFs.

## QC / Review

Keyboard shortcuts:

| Key | Action |
|---|---|
| `P` | Pan |
| `S` | Select scoring interval |
| `Z` | Zoom |
| `1` | Wake |
| `2` | NREM |
| `3` | REM |
| `A` | Apply Somnotate |
| `L` | Apply Layer 1 |
| `M` | Apply Manual |

You can also apply Somnotate, Layer 1, or Manual scoring to the entire visible window. Status/errors are displayed directly below those buttons.

Scoring writes immediately to:

```text
project_root/recordings/<recording_id>/final_scoring.csv
```

---

# 7. Video QC

Recommended browser format:

```text
MP4 / H.264
```

AVI can be converted locally using the app. FFmpeg is installed in the main Conda environment.

Manual equivalent command:

```text
ffmpeg -i "videoname.avi" -map 0:v:0 -an -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 -movflags +faststart "videoname.mp4"
```

Video synchronization uses:

```text
video_time_s = recording_time_s - video_offset_s
```

---

# 8. Diagnostics and troubleshooting

Core app:

```bash
python check_setup.py
```

Core app + Somnotate:

```bash
python check_setup.py --require-somnotate --somnotate-root /path/to/somnotate
```

Pipeline command logs are written to:

```text
logs/
```

When reporting a GitHub issue, include the app version, operating system, `check_setup.py` output, the shortest reproduction steps, and the relevant diagnostic log. Remove sensitive or identifying data before attaching screenshots/logs. The repository includes a bug-report template that prompts for these items.

### Somnotate result appears empty/gray

Current Somnotate versions can emit `awake` and `non-REM`. The app normalizes these to `Wake` and `NREM`. Re-import old Somnotate output if needed:

```bash
python pipelines/10_somnotate_layer.py import-results --project-root /path/to/project --recording-ids RECORDING_ID --epoch-sec 5
```

### Project moved from Mac to Windows

Load the new project root. The app prefers:

```text
<project_root>/recordings/<recording_id>
```

and repairs generated Somnotate manifest paths during the next run.

### `InconsistentVersionWarning` when loading an old model

Do not treat this as a harmless release warning. scikit-learn does not support loading pickled estimators with a different scikit-learn version. Legacy models are marked in the model dropdown and should be validated/retrained before final scientific use.

---

# 9. Release status

`0.9.0-rc1` is a release candidate for cross-platform hardening. Before tagging `v1.0.0`, complete `RELEASE_CHECKLIST.md`, including an end-to-end Windows test and a macOS regression test.

See:

- `START_HERE.md` — short lab-user setup guide
- `COMPATIBILITY.md` — supported/tested versions and known constraints
- `RELEASE_CHECKLIST.md` — release validation matrix
- `CHANGELOG.md` — release history
