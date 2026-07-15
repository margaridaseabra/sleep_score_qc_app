# Sleep Stage QC

A cross-platform Dash application for semi-automated EEG/EMG sleep scoring, quality control, and manual review.

The app supports importing recordings, computing initial Wake/Sleep scoring, running or importing Somnotate scoring, reviewing EEG/EMG/ACh signals, identifying disagreements between scoring layers, manually correcting sleep states, linking local video, and exporting reviewed scoring.

> **Recommended application:** Dash  
> **Legacy application:** The older Streamlit version is retained only for reference.

---

## Features

- MATLAB (`.mat`) and EDF (`.edf`) import
- Automatic Layer 1 Wake/Sleep scoring
- Somnotate Wake/NREM/REM integration
- Interactive EEG, EMG, ACh and spectrogram review
- Manual interval scoring and keyboard shortcuts
- Dissociation analysis between scoring layers
- Optional local video synchronization
- Local AVI-to-MP4 conversion for browser playback
- CSV, MATLAB and EDF export
- 1-second and 2-second final scoring epochs
- 1-second, 2-second and legacy 5-second Somnotate model support
- macOS and Windows support

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/margaridaseabra/sleep_score_qc_app.git
cd sleep_score_qc_app
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate sleep_stage_qc_v2
```

The environment installs Dash and FFmpeg. FFmpeg is used only for local AVI conversion; it is not bundled in the repository.

---

## Run on macOS

Open Terminal in the repository folder:

```bash
conda activate sleep_stage_qc_v2
bash run_app.sh
```

Alternatively:

```bash
python dash_app/app.py
```

## Run on Windows

Open **Anaconda Prompt** in the repository folder:

```bat
conda activate sleep_stage_qc_v2
run_app_windows.bat
```

Alternatively:

```bat
python dash_app\app.py
```

The app normally opens at:

```text
http://127.0.0.1:8050
```

### Windows diagnostics

```bat
python check_windows_setup.py
```

The diagnostic script checks Python, the active Conda environment, required packages, pipeline scripts, write permissions and FFmpeg availability.

---

## Typical workflow

1. Import a MATLAB or EDF recording.
2. Compute epoch features.
3. Run Layer 1 Wake/Sleep scoring.
4. Run or import Somnotate scoring, if required.
5. Review and correct scoring in **QC / Review**.
6. Run dissociation analysis, if required.
7. Export the final reviewed scoring.

---

## Project structure

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

Raw recordings and videos should remain outside the Git repository, for example on a local disk, external drive, OneDrive or shared data drive.

---

## Import and Layer 1

Supported recording formats:

| Format | Support |
|---|---|
| MATLAB (`.mat`) | Yes |
| EDF (`.edf`) | Yes |

Supported signals include EEG, EMG and optional ACh/fiber photometry.

After import, run **Compute epoch features** and then **Run Layer 1**. The main generated files are:

```text
epoch_features.csv
layer1_wake_sleep.csv
```

---

## Somnotate

The app supports existing or newly trained Somnotate models using:

- 1-second epochs
- 2-second epochs
- legacy 5-second models

The Somnotate scoring epoch length must match the epoch length used during model training.

Somnotate is installed separately from its upstream repository:

```text
https://github.com/paulbrodersen/somnotate
```

---

## QC / Review

The QC viewer can display:

- Final, Somnotate, Layer 1 and Manual scoring rows
- Raw EEG
- EEG spectrogram
- Raw EMG
- Optional ACh/fiber photometry
- State probabilities
- Dissociation review events
- Optional synchronized video

The coloured overlays over the traces represent the current **Final** scoring.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `P` | Pan |
| `S` | Select an interval |
| `Z` | Zoom |
| `1` | Apply Wake |
| `2` | Apply NREM |
| `3` | Apply REM |
| `A` | Apply Somnotate |
| `L` | Apply Layer 1 |
| `M` | Apply Manual scoring |

After an interval is scored, the selection is cleared to reduce accidental rescoring of an older selection.

---

## Local video synchronization

The app stores only the path to a video on the user's computer. Videos are not uploaded to GitHub or copied into the application.

Recommended browser format:

```text
MP4 with H.264 video
```

### AVI conversion

Most browsers cannot play AVI directly. When an AVI is selected, use **Convert AVI locally to browser MP4**.

The app then:

1. Keeps the original AVI unchanged.
2. Runs FFmpeg locally on the user's computer.
3. Saves the converted file beside the AVI as:

```text
original_name_browser.mp4
```

4. Reuses that MP4 on future runs when it is already up to date.
5. Stores the original AVI path and browser MP4 path in the recording's `metadata.json`.

Example local files:

```text
D:\sleep_data\mouse12\mouse12.avi
D:\sleep_data\mouse12\mouse12_browser.mp4
```

or on macOS:

```text
/Volumes/T7/sleep_data/mouse12/mouse12.avi
/Volumes/T7/sleep_data/mouse12/mouse12_browser.mp4
```

Video synchronization follows:

```text
video_time_s = recording_time_s - video_offset_s
```

The converted video remains part of the user's local dataset and is ignored by Git.

---

## Dissociation analysis

The Dissociation tab highlights periods where scoring layers disagree, including:

- Layer 1 versus Somnotate
- Final versus Somnotate
- Low-confidence periods
- Frequently corrected scoring patterns

After running the analysis, use the dissociation queue in **QC / Review** to inspect the flagged intervals.

---

## Final scoring and export

Final scoring starts as `Undefined` and changes only when the reviewer applies labels or fills empty epochs from another scoring source.

Final scoring is continuously saved to:

```text
project_root/recordings/<recording_id>/final_scoring.csv
```

Export formats:

- CSV
- MATLAB (`.mat`)
- EDF (`.edf`)

---

## Troubleshooting

### No recordings appear

Confirm that the selected project root contains:

```text
recordings/
```

and at least one imported recording folder.

### Feature extraction fails

Update and reactivate the environment:

```bash
conda env update -f environment.yml --prune
conda activate sleep_stage_qc_v2
```

Check the pipeline log in the repository's `logs/` folder.

### Layer 1 does not run

Confirm that `epoch_features.csv` exists in the recording folder. Layer 1 depends on that file.

### AVI conversion is unavailable

Confirm FFmpeg is available in the active environment:

```bash
ffmpeg -version
```

If it is missing:

```bash
conda env update -f environment.yml --prune
conda activate sleep_stage_qc_v2
```

### Somnotate feature mismatch

Use a Somnotate model trained with the same epoch duration as the scoring run.

### Windows setup problems

Run:

```bat
python check_windows_setup.py
```

and share the generated output and relevant file from `logs/` when reporting a problem.

---

## Legacy Streamlit application

The previous Streamlit application is retained only for reference. New scoring and development should use the Dash application.
