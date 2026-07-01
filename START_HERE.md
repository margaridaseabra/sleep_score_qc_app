# Start here

Use this file when setting up the app for the first time.

This repository now uses the **Dash version** of the semi-automated sleep scoring QC app.

The older Streamlit app is legacy and should not be used for new scoring.

---

## 1. Install / clone

Clone the repository:

```bash
git clone https://github.com/margaridaseabra/sleep_score_qc_app.git
cd sleep_score_qc_app
```

---

## 2. Create the environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
```

If your local environment already exists under another name, for example `sleep_app`, activate that instead:

```bash
conda activate sleep_app
```

---

## 3. Launch the Dash app

Run:

```bash
bash run_app.sh
```

Open the local URL printed in the terminal, usually:

```text
http://127.0.0.1:8050
```

---

## 4. Prepare your data

You need a local project folder. The app will read/write files inside that folder.

Expected structure:

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
```

Do not put large real data files directly in GitHub.

Use a local folder, shared drive, or OneDrive folder for recordings.

---

## 5. First time inside the app

### Step 1 — Load project

1. Paste/select the project root folder.
2. Click **Load project**.
3. Confirm that recordings appear in the dropdowns.

### Step 2 — Import `.mat` + Layer 1

Open:

```text
1. Import .mat + Layer 1
```

Typical import fields:

| Field | Example |
|---|---|
| EEG variable | `eeg` |
| EMG variable | `emg` |
| ACh / photometry variable | `ne` |
| EEG sampling frequency variable | `eeg_frequency` |
| ACh sampling frequency variable | `ne_frequency` |

Run the import/processing steps in order.

### Step 3 — QC / Review

Open:

```text
2. QC / Review
```

Use this page to inspect:

- scoring rows
- EEG
- EMG
- ACh/fiber photometry if available
- probabilities
- dissociation events
- final scoring


### Optional video QC

In the QC / Review tab, you can link a local video to the recording.

1. Paste the full path to the `.mp4`, `.mov`, or `.avi` file.
2. Set the video offset in seconds. Use `0` if the video and EEG start together.
3. Click **Save video**.
4. Use **Jump video to window start** or **Jump video to selected interval** during review.

MP4 is the most reliable browser format. AVI paths are accepted, but if the video does not play, convert it to MP4:

```bash
ffmpeg -i input_video.avi -c:v libx264 -crf 23 -preset fast -c:a aac output_video.mp4
```

Synchronization rule:

```text
video_time_s = recording_time_s - video_offset_s
```

### Step 4 — Somnotate

Open:

```text
3. Somnotate
```

Use this if you want to run automatic Wake/NREM/REM scoring.

You need:

- local Somnotate repository path
- Somnotate Python executable or Conda environment
- trained model file if using an existing model

### Step 5 — Dissociation

Open:

```text
4. Dissociation
```

Run dissociation analysis to find suspicious periods or disagreement between scoring layers.

Then return to QC / Review and use the dissociation review queue to jump through interesting events.

---

## 6. Scoring workflow

The app keeps different scoring layers separate:

| Layer | Meaning |
|---|---|
| Layer 1 | Automatic Wake/Sleep |
| Somnotate | Automatic Wake/NREM/REM |
| Manual | Imported manual labels |
| Final | Reviewed labels created by the user |

Important:

```text
Final scoring starts empty/Undefined by default.
```

The user must explicitly add labels to the Final score.

You can apply labels to a selected interval using:

- Wake
- NREM
- REM
- Somnotate / automatic
- Layer 1
- Manual

You can also apply Somnotate, Layer 1, or Manual to the whole visible window.

---

## 7. QC keyboard shortcuts

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

Typical use:

```text
P = move through recording
S = select a window
1/2/3/A/L/M = apply scoring
```

---

---

## Safe keyboard scoring and final utilities

The QC / Review tab is designed so keyboard scoring uses the current confirmed selected interval. After a selected interval is scored, the selection is cleared, so a new interval must be selected before the next keyboard scoring action. This reduces the risk of accidentally scoring an older selection.

The bottom of the QC / Review page includes **Final scoring utilities**:

- **Fill empty Final with Somnotate**: fills only epochs where Final is still empty/Undefined; existing reviewed labels are preserved.
- **Fill empty Final with Somnotate + export**: fills empty epochs from Somnotate, then exports the final scoring.
- **Export final scoring**: exports the current Final scoring without changing labels.

The QC page layout is ordered for review: dissociation review queue first, recording-position controls directly above the plot, then scoring/video/export tools below.

## 8. Does scoring save automatically?

Yes.

The app writes final scoring to:

```text
project_root/recordings/<recording_id>/final_scoring.csv
```

If you close the app, labels already written to this file should remain.

To stop safely:

1. Finish the current scoring action.
2. Wait for feedback/status to update.
3. Stop the server with `Ctrl+C`.
4. Restart later with `bash run_app.sh`.
5. Load the same project and recording.

---

## 9. Quick safety backup

Before a long scoring session or before stopping for the day, you can back up final scoring files:

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

## 10. Common problems

### I cannot see my recording

Check that the project root is correct and that it contains a `recordings/` folder.

### The app opens on the wrong port

Dash usually opens on:

```text
http://127.0.0.1:8050
```

The old Streamlit app used port `8501`; that is not the current recommended version.

### The final score is already filled

That recording may have an old `final_scoring.csv`.

Use the reset option in the app, or back up the file and reset the Final score to `Undefined`.

### I cannot select a scoring window

Press `S` to activate select mode, then drag horizontally over the QC plot.

Press `P` to go back to pan mode.

---

## 11. What to tell colleagues

Use the Dash app.

Run:

```bash
bash run_app.sh
```

Open:

```text
http://127.0.0.1:8050
```

Then follow the tabs from left to right.
