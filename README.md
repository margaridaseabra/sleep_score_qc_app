# Semi-automated sleep scoring QC app

This repository contains a **Dash-based app** for semi-automated EEG/EMG sleep scoring quality control.

The app is designed to help users import sleep recordings, run automatic scoring steps, inspect EEG/EMG/ACh signals, compare scoring layers, review suspicious/dissociated periods, manually correct labels, and save a final reviewed sleep scoring file.

> **Current recommended version:** Dash app
> **Legacy version:** the older Streamlit app is kept only for reference and should not be used for new scoring.

---

## What this app is for

This app was developed for sleep scoring quality control in mouse EEG/EMG recordings.

It supports:

* importing `.mat` recordings;
* extracting EEG, EMG, optional ACh/fiber photometry, and optional manual scoring;
* running a first automatic Wake/Sleep layer;
* using Somnotate scoring when available;
* comparing scoring layers;
* reviewing dissociation/disagreement events;
* manually correcting scoring;
* saving a final reviewed scoring file.

The main idea is that automatic models help guide the review, but the user keeps control over the final scoring.

---

## Main scoring layers

The app keeps scoring layers separate:

| Layer         | Meaning                              |
| ------------- | ------------------------------------ |
| **Layer 1**   | Automatic Wake/Sleep scoring         |
| **Somnotate** | Automatic Wake/NREM/REM scoring      |
| **Manual**    | Imported manual labels, if available |
| **Final**     | Reviewed labels created by the user  |

Important:

> **Final scoring starts empty/Undefined by default.**
> It is only filled when the user explicitly applies labels or accepts one of the scoring sources.

This means the app should not automatically copy Layer 1 or Somnotate into the final score unless the user chooses to do so.

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/margaridaseabra/sleep_score_qc_app.git
cd sleep_score_qc_app
```

### 2. Create the Python environment

If using Conda:

```bash
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
```

### 3. Launch the Dash app

```bash
bash run_app.sh
```

Then open the URL printed in the terminal, usually:

```text
http://127.0.0.1:8050
```

---

## App tabs

The app is organized into four main tabs.

### 1. Import `.mat` + Layer 1

Use this tab to import a recording and prepare it for review.

Typical variable names used in the `.mat` file:

| Field                           | Example         |
| ------------------------------- | --------------- |
| EEG variable                    | `eeg`           |
| EMG variable                    | `emg`           |
| ACh / photometry variable       | `ne`            |
| EEG sampling frequency variable | `eeg_frequency` |
| ACh sampling frequency variable | `ne_frequency`  |

After importing the recording, this tab can also be used to run Layer 1 Wake/Sleep scoring.

---

### 2. QC / Review

Use this tab for the main scoring review.

The QC viewer can show:

* scoring rows;
* raw EEG;
* raw EMG;
* optional ACh/fiber photometry;
* scoring probabilities;
* dissociation review queue;
* final reviewed scoring.

The faint colours over the raw traces correspond to the current **Final** scoring.

---

### 3. Somnotate

Use this tab to run or import Somnotate scoring.

Somnotate is not bundled inside this repository. It must be installed separately from:

```text
https://github.com/paulbrodersen/somnotate
```

The app can use Somnotate outputs as an automatic Wake/NREM/REM scoring layer.

---

### 4. Dissociation

Use this tab to identify periods where scoring layers disagree or where model confidence is low.

Examples of useful comparisons:

* Layer 1 vs Somnotate;
* Final vs Somnotate;
* Wake/Sleep disagreement;
* low-confidence automatic scoring;
* suspicious periods for manual review.

After running dissociation analysis, return to the **QC / Review** tab and use the dissociation review queue to jump through interesting events.

---

## Expected project folder structure

The app expects a local project folder containing prepared recordings.

A typical project structure is:

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
        └── somnotate/
            └── somnotate_results_timeseries.csv
```

The app reads and writes files inside the selected project folder.

Large data files should not be committed to GitHub. Keep raw recordings and prepared data on a local folder, shared drive, or OneDrive folder.

---

## Important files created by the app

### `final_scoring.csv`

This is the main reviewed output file.

It is saved inside each recording folder:

```text
project_root/recordings/<recording_id>/final_scoring.csv
```

This file contains the user-reviewed scoring.

### `layer1_wake_sleep.csv`

This contains the Layer 1 automatic Wake/Sleep scoring.

### `manual_scoring_aligned.csv`

This contains imported manual labels, if available.

### Somnotate output files

Somnotate outputs are stored inside the recording’s `somnotate/` folder when available.

---

## QC keyboard shortcuts

| Key | Action                              |
| --- | ----------------------------------- |
| `P` | Pan / move through the recording    |
| `S` | Select window for scoring           |
| `1` | Apply Wake                          |
| `2` | Apply NREM                          |
| `3` | Apply REM                           |
| `A` | Apply automatic / Somnotate scoring |
| `L` | Apply Layer 1 scoring               |
| `M` | Apply Manual scoring                |
| `Z` | Zoom mode                           |

Typical use:

```text
P = move through the recording
S = select a scoring window
1 / 2 / 3 / A / L / M = apply label/source
```

---

## How scoring is saved

Scoring is saved automatically when labels are applied.

The reviewed scoring is written to:

```text
project_root/recordings/<recording_id>/final_scoring.csv
```

If you close the app, labels already written to this file should remain.

What may be lost when closing the app:

* current zoom/pan view;
* currently selected interval;
* active Plotly mouse mode;
* temporary interface state.

What should not be lost:

* labels already applied to `final_scoring.csv`.

---

## Safe stopping procedure

If you need to stop while scoring:

1. Finish the current scoring action.
2. Wait for the app feedback/status to update.
3. Note the approximate time/minute where you stopped, if useful.
4. Stop the app in the terminal with `Ctrl+C`.
5. Restart later with:

```bash
bash run_app.sh
```

6. Load the same project and recording.
7. Continue scoring from the saved `final_scoring.csv`.

---

## Optional backup of final scoring files

Before a long scoring session or before stopping for the day, you can back up all `final_scoring.csv` files:

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

The previous Streamlit version is kept only for reference.

New users should use the Dash app.

If the repository still contains a file such as:

```text
sleep_stage_qc_v2_app.py
```

treat it as legacy code.

The current recommended launch command is:

```bash
bash run_app.sh
```

not:

```bash
streamlit run ...
```

---

## Troubleshooting

### The app opens but no recordings appear

Check that the selected project root contains a `recordings/` folder.

Expected structure:

```text
project_root/
└── recordings/
    └── recording_id/
```

### The app opens on the wrong port

The Dash app usually opens at:

```text
http://127.0.0.1:8050
```

The old Streamlit app used port `8501`. That is not the current recommended app.

### The final score is already filled

This can happen if the recording already had an older `final_scoring.csv` created before the empty-final workflow.

To fix this, reset the Final scoring in the app or manually back up and reset `final_scoring.csv` to `Undefined`.

### I cannot select a scoring window

Press:

```text
S
```

to activate select mode, then drag horizontally over the QC plot.

Press:

```text
P
```

to return to pan mode.

### Somnotate is not found

Somnotate must be installed separately.

Check:

* the local Somnotate repository path;
* the Python/Conda environment used for Somnotate;
* whether the trained model file exists.

---

## What to tell new users

1. Clone the repository.
2. Create or activate the Python environment.
3. Run:

```bash
bash run_app.sh
```

4. Open:

```text
http://127.0.0.1:8050
```

5. Choose the project root.
6. Follow the app tabs from left to right.
7. Use the QC / Review tab for final scoring.
8. Final scoring is saved in `final_scoring.csv`.
