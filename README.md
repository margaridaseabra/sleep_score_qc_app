# Sleep Stage QC

A cross-platform Dash application for semi-automated EEG/EMG sleep scoring, quality control, and manual review.

The application provides an end-to-end workflow for importing recordings, computing automatic scoring, reviewing EEG/EMG/ACh signals, comparing multiple scoring layers, manually correcting sleep states, and exporting the final reviewed scoring.

> **Current version:** Dash application  
> **Legacy version:** The previous Streamlit implementation is kept only for reference and is no longer recommended.

---

# Features

- Import recordings from:
  - MATLAB (`.mat`)
  - EDF (`.edf`)
- Export reviewed recordings to:
  - CSV
  - MATLAB (`.mat`)
  - EDF (`.edf`)
- Automatic Layer 1 Wake/Sleep scoring
- Somnotate integration
- Manual scoring review
- Dissociation analysis
- Interactive EEG/EMG/ACh viewer
- Interactive spectrogram
- Optional synchronized video review
- Support for:
  - 1-second epochs
  - 2-second epochs
  - Legacy 5-second Somnotate models
- Cross-platform support
  - macOS
  - Windows

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/margaridaseabra/sleep_score_qc_app.git
cd sleep_score_qc_app
```

---

## 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate sleep_stage_qc_v2
```

---

# Running the application

## macOS

```bash
bash run_app.sh
```

or

```bash
python dash_app/app.py
```

---

## Windows

Open **Anaconda Prompt** and run

```bat
run_app_windows.bat
```

or

```bat
python dash_app\app.py
```

The application will normally open at

```
http://127.0.0.1:8050
```

---

# Windows diagnostics

Before running the application on Windows, it is recommended to verify the installation:

```bash
python check_windows_setup.py
```

The diagnostics script checks:

- Python installation
- Conda environment
- Dash
- NumPy
- pandas
- SciPy
- scikit-learn
- pyEDFlib
- required pipeline scripts
- repository write permissions

If FFmpeg is not found, only AVI video conversion will be unavailable. The rest of the application will continue to function normally.

---

# Typical workflow

1. Import recording
2. Compute epoch features
3. Run Layer 1 Wake/Sleep scoring
4. Run Somnotate (optional)
5. Review scoring in the QC viewer
6. Export the final reviewed scoring

---

# Importing recordings

Supported formats

| Format | Supported |
|---------|-----------|
| MATLAB (.mat) | ✓ |
| EDF (.edf) | ✓ |

Supported channels include

- EEG
- EMG
- ACh / Fiber photometry (optional)

During import, the application creates the processed recording folder used throughout the remainder of the workflow.

---

# Layer 1 Wake/Sleep scoring

Layer 1 computes:

- EMG RMS
- EEG spectral features
- Wake/Sleep classification

The generated files include

```
epoch_features.csv
layer1_wake_sleep.csv
```

---

# Somnotate

The application supports:

- Running existing Somnotate models
- Training new Somnotate models
- 1-second epochs
- 2-second epochs
- Legacy 5-second models

**Important**

The epoch length used during scoring must match the epoch length used when the model was trained.

Legacy models without stored epoch metadata will generate a warning before scoring.

Somnotate itself is **not bundled** with this repository and must be installed separately:

https://github.com/paulbrodersen/somnotate

---

# QC / Review

The QC viewer allows simultaneous visualization of:

- EEG
- EMG
- ACh / Fiber photometry
- EEG spectrogram
- State probabilities
- Manual scoring
- Layer 1 scoring
- Somnotate scoring
- Final reviewed scoring

Features include

- Interactive Plotly viewer
- Scroll-wheel zoom
- Adaptive spectrogram resolution
- Dissociation review queue
- Manual interval scoring
- Keyboard shortcuts
- Video synchronization

The coloured overlays correspond to the current **Final** scoring.

---

# Video synchronization

The QC viewer supports linking a local video file to each recording.

Recommended formats:

- MP4 (H.264)
- MOV

AVI files may not play directly in some browsers.

If required, convert AVI to MP4 using:

```bash
ffmpeg -i "video.avi" \
-map 0:v:0 \
-an \
-c:v libx264 \
-pix_fmt yuv420p \
-preset fast \
-crf 23 \
-movflags +faststart \
"video.mp4"
```

Video synchronization is performed using

```
video_time = recording_time − video_offset
```

The viewer includes controls for

- Jumping to the selected QC window
- Playing only the selected interval
- Automatic stopping at the end of the selected interval

---

# Dissociation analysis

The Dissociation module highlights periods where scoring methods disagree.

Examples include

- Layer 1 vs Somnotate
- Final vs Somnotate
- Low-confidence Somnotate predictions

Biological summaries include information such as

- Common reviewer corrections
- Distribution of corrected sleep states
- Frequently reviewed transitions
- Percentage of Somnotate REM corrected by the reviewer

---

# Export

The reviewed Final scoring can be exported as

- CSV
- MATLAB (.mat)
- EDF (.edf)

---

# Project structure

```
project_root/
└── recordings/
    └── recording_id/
        ├── metadata.json
        ├── eeg.npy
        ├── emg.npy
        ├── ach.npy                    # optional
        ├── epoch_features.csv
        ├── layer1_wake_sleep.csv
        ├── final_scoring.csv
        ├── manual_scoring_aligned.csv # optional
        └── somnotate/
            └── somnotate_results_timeseries.csv
```

Raw recordings should generally remain outside the repository and be stored on a shared drive, OneDrive, or local project folder.

---

# Keyboard shortcuts

| Key | Action |
|------|--------|
| P | Pan |
| S | Select interval |
| Z | Zoom |
| 1 | Apply Wake |
| 2 | Apply NREM |
| 3 | Apply REM |
| A | Apply Somnotate |
| L | Apply Layer 1 |
| M | Apply Manual scoring |

---

# Troubleshooting

## No recordings appear

Verify that the selected project contains

```
recordings/
```

with one or more imported recording folders.

---

## Feature extraction fails

Update the Conda environment

```bash
conda env update -f environment.yml --prune
```

---

## Video does not play

Some browsers cannot decode AVI files.

Convert the video to MP4 using the FFmpeg command shown above.

---

## Windows installation

Run

```bash
python check_windows_setup.py
```

to verify the installation.

---

## Somnotate

Always use a Somnotate model trained with the same epoch length as the recordings being scored.

---

# Legacy Streamlit version

The previous Streamlit application is retained only for reference.

All new development is performed in the Dash application.
