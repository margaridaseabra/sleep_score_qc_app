# Semi-automated sleep scoring QC app

This repository contains a Dash-based app for semi-automated EEG/EMG sleep scoring quality control.

The app is designed to help users import recordings, run initial Wake/Sleep scoring, inspect EEG/EMG/ACh traces, compare automatic scoring layers, review dissociation events, manually correct scoring, and export final reviewed sleep scoring.

> **Current recommended version:** Dash app  
> **Legacy version:** the older Streamlit app is kept only for reference and should not be used for new scoring.

---

## What the app does

The app supports the following workflow:

1. Import `.mat` recordings containing EEG, EMG, optional ACh/fiber photometry, and optional manual scoring.
2. Run Layer 1 Wake/Sleep scoring.
3. Run or import Somnotate Wake/NREM/REM scoring.
4. Inspect EEG, EMG, ACh, scoring rows, probabilities, and dissociation events.
5. Select windows manually and apply reviewed labels.
6. Save/export final scoring.

The main scoring layers are:

| Layer | Meaning |
|---|---|
| Layer 1 | Automatic Wake/Sleep scoring |
| Somnotate | Automatic Wake/NREM/REM scoring |
| Manual | Imported manual scoring, if available |
| Final | Reviewed scoring created by the user |

Important: **Final scoring starts empty/Undefined by default.** It is only filled when the user explicitly applies labels or accepts a source such as Somnotate, Layer 1, or Manual.

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/margaridaseabra/sleep_score_qc_app.git
cd sleep_score_qc_app
```

### 2. Create the Python environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
```

If your local environment already exists under another name, for example `sleep_app`, activate that instead:

```bash
conda activate sleep_app
```

### 3. Launch the Dash app

```bash
bash run_app.sh
```

Then open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:8050
```

---

## Expected project folder structure

The app expects a project folder containing one or more prepared recordings:

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
        ├── manual_scoring_aligned.csv       # optional
        ├── final_scoring.csv
        ├── metadata.json                  # may include video_file and video_offset_s
        └── somnotate/
            └── somnotate_results_timeseries.csv
```

Raw `.mat`, `.npy`, `.h5`, `.hdf5`, and large recording files should generally **not** be committed to GitHub. Keep data on a shared drive, OneDrive, or a local project folder.

---

## Typical workflow inside the app

### 1. Import `.mat` + Layer 1

Use this tab to import a recording and generate the prepared files used by the app.

Typical variables:

| Field | Example |
|---|---|
| EEG variable | `eeg` |
| EMG variable | `emg` |
| ACh / photometry variable | `ne` |
| EEG sampling frequency variable | `eeg_frequency` |
| ACh sampling frequency variable | `ne_frequency` |

After import, run Layer 1 Wake/Sleep scoring.

### 2. QC / Review

Use this tab to inspect the recording and edit the final scoring.

The viewer can show:

- scoring rows
- raw EEG
- EEG spectrogram
- raw EMG
- optional ACh/fiber photometry
- state probabilities
- dissociation review queue

The faint colours over the raw traces correspond to the current **Final** scoring.


### Optional video QC

The QC / Review tab includes an optional **Video QC** panel. It can link a local `.mp4`, `.mov`, or `.avi` video to each recording. The video path and synchronization offset are saved in the recording `metadata.json` file.

Recommended video format:

```text
.mp4 encoded with H.264
```

AVI files can be selected and saved, but many browsers cannot play `.avi` directly. If the video player is blank, convert the file to MP4 and save the MP4 path instead:

```bash
ffmpeg -i input_video.avi -c:v libx264 -crf 23 -preset fast -c:a aac output_video.mp4
```

Video synchronization uses:

```text
video_time_s = recording_time_s - video_offset_s
```

Examples:

| Situation | `video_offset_s` |
|---|---:|
| Video and EEG start together | `0` |
| Video starts 10 s after EEG | `10` |
| Video starts 5 s before EEG | `-5` |

The video panel has buttons to jump the video to the current QC window start or to play only the selected scoring interval. When playing a selected interval, the video automatically pauses at the end of that selected period.

### 3. Somnotate

Use this tab to run or import Somnotate scoring.

Somnotate is not bundled inside this repository. It must be installed separately from:

```text
https://github.com/paulbrodersen/somnotate
```

The app can use an existing trained model or help prepare recordings for a Somnotate workflow.

### 4. Dissociation

Use this tab to detect and review disagreement between scoring layers, for example:

- Layer 1 vs Somnotate
- Final vs Somnotate
- low-confidence periods
- suspicious dissociation events

After running dissociation analysis, return to the QC / Review tab and use the dissociation review queue to jump through interesting parts of the recording.

---

## QC keyboard shortcuts

| Key | Action |
|---|---|
| `P` | Pan / move through the recording |
| `S` | Select window for scoring |
| `1` | Apply Wake |
| `2` | Apply NREM |
| `3` | Apply REM |
| `A` | Apply automatic / Somnotate scoring |
| `L` | Apply Layer 1 scoring |
| `M` | Apply Manual scoring |
| `Z` | Zoom mode |

---

## Saving and interrupting scoring

Scoring is saved to:

```text
project_root/recordings/<recording_id>/final_scoring.csv
```

Each time the user applies a label, the app writes the updated final scoring to this file.

If you need to stop in the middle of scoring:

1. Finish the current scoring action.
2. Wait for the app feedback/status to update.
3. Optionally note the approximate time where you stopped.
4. Stop the app with `Ctrl+C` in the terminal.
5. Restart later with `bash run_app.sh`.
6. Load the same project and recording.

You should not lose labels already written to `final_scoring.csv`.

For extra safety, you can manually back up final scoring files:

```bash
PROJECT="/path/to/project_root"
STAMP=$(date +%Y%m%d_%H%M)

find "$PROJECT/recordings" -name "final_scoring.csv" -exec sh -c '
  for f do
    cp "$f" "${f%.csv}_backup_'$STAMP'.csv"
    echo "Backed up $f"
  done
' sh {} +
```

---

## Legacy Streamlit version

The previous Streamlit version is kept only for reference. New users should use the Dash app.

If the repository still contains `sleep_stage_qc_v2_app.py`, treat it as legacy code.

---

## Troubleshooting

### The app opens but no recordings appear

Check that the project root points to a folder containing:

```text
recordings/
recordings_manifest.csv
```

or at least one prepared recording folder inside `recordings/`.

### The final score looks already filled

Existing recordings may have an old `final_scoring.csv` created before the empty-final workflow. Reset it from the QC viewer, or manually back it up and set `final_state` to `Undefined`.

### The Plotly selection does not work

Press `S` to activate select mode, then drag horizontally over the QC plot.

Press `P` to return to pan mode.

### The app still looks like the old Streamlit version

Make sure you launched the Dash app with:

```bash
bash run_app.sh
```

and that the terminal says the app is running on:

```text
http://127.0.0.1:8050
```
