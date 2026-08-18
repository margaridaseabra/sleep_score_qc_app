# Sleep Stage QC

**Sleep Stage QC** is a Dash-based application for semi-automated EEG/EMG sleep scoring quality control.

It supports recording import, Layer 1 Wake/Sleep scoring, Somnotate Wake/NREM/REM scoring, manual review, synchronized video QC, dissociation analysis, and export of final reviewed scoring.

> **Current release candidate:** `1.1.0-rc1`

## Windows: easiest installation

For normal Windows users, you do **not** need to create Python environments or install Somnotate manually.

### Prerequisite

Install one Conda distribution first:

- Miniconda
- Anaconda
- Miniforge

Then:

1. Open this repository on GitHub.
2. Choose **Code → Download ZIP**.
3. Extract the ZIP to a normal folder.
4. Double-click **`INSTALL_WINDOWS.bat`**.
5. Wait until it reports **Installation successful**.
6. Double-click **`RUN_APP.bat`** whenever you want to use the app.

`RUN_APP.bat` starts the Dash server and opens:

```text
http://127.0.0.1:8050
```

### What the Windows installer does

`INSTALL_WINDOWS.bat` automatically:

- creates or updates the `sleep_stage_qc_v2` Conda environment;
- creates or updates the separate `somnotate_env` environment;
- downloads Somnotate from the official `paulbrodersen/somnotate` GitHub repository;
- pins Somnotate to the tested commit:

```text
a20f33de62511d8c172e333896608b7fc166d0f0
```

- installs Somnotate into its dedicated environment;
- checks that the Somnotate source matches the app's supported pipeline contract;
- runs the app setup diagnostics.

The supported Somnotate source is stored automatically under the user's local application-data directory, approximately:

```text
C:\Users\<USER>\AppData\Local\SleepStageQC\somnotate_a20f33de
```

Users normally do **not** need to enter or manage that path manually.

The installer can be run again later to repair or update the environments.

## Main workflow

The app supports:

1. Import `.mat` or EDF recordings.
2. Compute epoch features.
3. Run Layer 1 Wake/Sleep scoring.
4. Run Somnotate Wake/NREM/REM scoring with an existing model, or train/evaluate a model.
5. Review EEG, EMG, probabilities and scoring layers.
6. Correct Final scoring manually or copy labels from Somnotate, Layer 1 or Manual scoring.
7. Review dissociation/disagreement events.
8. Use synchronized video QC when video is available.
9. Export final scoring.

The scoring layers are:

| Layer | Meaning |
|---|---|
| Layer 1 | Automatic Wake/Sleep |
| Somnotate | Automatic Wake/NREM/REM |
| Manual | Imported manual scoring |
| Final | Reviewed scoring created by the user |

**Final scoring starts empty/Undefined by default** and is filled only when the user explicitly applies or accepts labels.

## Project folder structure

A prepared project typically looks like:

```text
project_root/
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
            └── somnotate_results_timeseries.csv
```

Keep real recording data outside this Git repository.

## QC / Review

The QC viewer can display:

- scoring rows;
- EEG;
- EEG spectrogram;
- EMG;
- optional ACh/fiber-photometry trace;
- Somnotate state probabilities;
- Final scoring;
- dissociation-review events.

Useful keyboard shortcuts:

| Key | Action |
|---|---|
| `P` | Pan |
| `S` | Select interval |
| `Z` | Zoom |
| `1` | Wake |
| `2` | NREM |
| `3` | REM |
| `A` | Apply Somnotate |
| `L` | Apply Layer 1 |
| `M` | Apply Manual |

Final scoring is written to:

```text
project_root/recordings/<recording_id>/final_scoring.csv
```

## Synchronized video QC

A recording can be linked to an `.mp4`, `.mov`, or `.avi` video.

For reliable browser playback, H.264 MP4 is recommended.

Long source videos do not need to be converted in full. During synchronized review, the app creates short optimized **local QC clips** and reuses them for nearby selections. This avoids repeatedly seeking through multi-hour videos.

The original video is not modified.

The synchronized review panel provides:

- EEG and EMG for the selected interval;
- synchronized video;
- Play selection;
- previous/next epoch controls;
- replay current epoch;
- a browser-side red playhead showing the current video position on EEG/EMG;
- a local review cache that can be cleared from the app.

Synchronization uses:

```text
video_time_s = recording_time_s - video_offset_s
```

## Somnotate

Somnotate runs in a separate environment because the supported Somnotate 0.5.0 stack uses older scientific dependencies than the main Dash app.

### Normal Windows users

Do not install Somnotate manually.

Run:

```text
INSTALL_WINDOWS.bat
```

The installer downloads and configures the supported Somnotate version automatically.

### Advanced/manual Somnotate setup

For development, troubleshooting, macOS setup, or manual installation:

```bash
git clone https://github.com/paulbrodersen/somnotate.git
cd somnotate
git checkout a20f33de62511d8c172e333896608b7fc166d0f0
```

From the Sleep Stage QC repository:

```bash
conda env create -f environment_somnotate.yml
conda activate somnotate_env
python -m pip install -e /path/to/somnotate --no-deps
```

The tested Windows Somnotate runtime uses:

```text
Python          3.9
Somnotate       0.5.0
pomegranate     0.14.4
numpy           1.26.4
pandas          2.3.3
scikit-learn    1.6.1
scipy           1.13.1
matplotlib      3.9.4
pyedflib        0.1.42
lspopt          1.4.0
```

The app runs Somnotate using a temporary compatibility copy of its example pipeline. The upstream Somnotate checkout itself is not modified.

## Bundled models

The repository can contain Somnotate model files under:

```text
somnotate_models/
```

Historical models may have been serialized under a different scikit-learn version. Model metadata and runtime warnings should therefore be reviewed before using a model for final scientific analyses.

For reproducible final work, prefer a model trained and saved with the standardized supported environment.

## macOS

The current Windows installer is the supported easy-install path for v1.1.

For macOS, use the manual setup described in `START_HERE.md` and `COMPATIBILITY.md` until the dedicated Mac installer has completed regression testing.

Apple Silicon may require the Somnotate environment to run as `osx-64` under Rosetta because of the legacy `pomegranate 0.14.4` dependency.

## Diagnostics and tests

Check the current setup with:

```bash
python check_setup.py
```

Run the test suite with:

```bash
python -m pytest -q
```

The repository CI runs the supported tests on Windows and macOS.

## Repository structure

```text
sleep_score_qc_app/
├── INSTALL_WINDOWS.bat
├── RUN_APP.bat
├── README.md
├── START_HERE.md
├── COMPATIBILITY.md
├── CHANGELOG.md
├── RELEASE_CHECKLIST.md
├── VERSION
├── environment.yml
├── environment_somnotate.yml
├── check_setup.py
├── check_windows_setup.py
├── run_app.sh
├── dash_app/
├── pipelines/
├── somnotate_models/
├── tests/
├── tools/
└── .github/
```

## Troubleshooting

### Conda is not found

Install Miniconda, Anaconda, or Miniforge, then run `INSTALL_WINDOWS.bat` again.

### App environment is missing or damaged

Run `INSTALL_WINDOWS.bat` again. It is designed to update/repair the environments.

### Somnotate is missing

Run `INSTALL_WINDOWS.bat` again. The installer automatically restores the supported Somnotate source and environment.

### App does not open automatically

Open:

```text
http://127.0.0.1:8050
```

while `RUN_APP.bat` is still running.

### No recordings appear

Confirm that the selected project root contains a `recordings/` folder and prepared recording files.

### Video review is slow

Use the synchronized review panel. It creates short local QC clips specifically to avoid repeated random seeking through long source videos.

## Development / release status

`1.1.0-rc1` is a release candidate.

Before final `1.1.0`:

- complete fresh-install Windows validation;
- complete full macOS regression;
- verify the same Somnotate model/recording produces equivalent outputs across supported platforms;
- then merge the release branch and tag the final release.
