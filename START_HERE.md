# Start here

This is the short setup guide for **Sleep Stage QC 1.1.0-rc1**.

## Windows — recommended

### 1. Install Conda once

Install one of:

- Miniconda
- Anaconda
- Miniforge

You do not need to create any environments manually.

### 2. Download Sleep Stage QC

On GitHub choose:

```text
Code → Download ZIP
```

Extract the ZIP to a normal folder.

### 3. Install everything

Double-click:

```text
INSTALL_WINDOWS.bat
```

Wait for:

```text
Installation successful.
```

This automatically creates the app environment, creates the Somnotate environment, downloads the supported Somnotate source, installs it, and runs diagnostics.

### 4. Run the app

Double-click:

```text
RUN_APP.bat
```

The browser should open automatically at:

```text
http://127.0.0.1:8050
```

Keep the launcher window open while using the app.

That is the normal Windows installation. No manual Somnotate setup is required.

## First workflow

1. Set **Project root**.
2. Click **Load project**.
3. Open **1. Import .mat / EDF + Layer 1** if the recording is not already prepared.
4. Import the recording.
5. Compute epoch features.
6. Run Layer 1.
7. Open **3. Somnotate** for Wake/NREM/REM scoring.
8. Open **2. QC / Review** to inspect and correct scoring.
9. Use **4. Dissociation** if disagreement/event review is needed.
10. Export Final scoring when review is complete.

Final scoring saves automatically to:

```text
project_root/recordings/<recording_id>/final_scoring.csv
```

## Video review

In **QC / Review**:

1. Link the recording video.
2. Set the video offset if needed.
3. Select an interval in the main QC plot.
4. Use the synchronized review panel.

The app creates short local QC clips automatically for fast review of long videos.

The panel shows synchronized EEG, EMG and video with a red moving playhead.

## Somnotate on Windows

For normal Windows use, Somnotate is installed automatically by `INSTALL_WINDOWS.bat`.

The installer uses the supported Somnotate source commit:

```text
a20f33de62511d8c172e333896608b7fc166d0f0
```

Users normally do not need to clone Somnotate or enter its path manually.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `P` | Pan |
| `S` | Select interval |
| `Z` | Zoom |
| `1` | Wake |
| `2` | NREM |
| `3` | REM |
| `A` | Somnotate |
| `L` | Layer 1 |
| `M` | Manual |

## macOS

The dedicated easy installer is currently Windows-only.

For macOS, create the app environment manually:

```bash
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
./run_app.sh
```

Somnotate uses its own environment. Follow `COMPATIBILITY.md` for the current tested Mac requirements.

## If something is broken on Windows

First run:

```text
INSTALL_WINDOWS.bat
```

again.

It is designed to update or repair the environments and supported Somnotate installation.

For detailed diagnostics:

```bat
conda activate sleep_stage_qc_v2
python check_setup.py
```

For more details, read `README.md` and `COMPATIBILITY.md`.
