# Start here

This is the short setup guide for lab users of **Sleep Stage QC 1.1.0-rc1**.

The supported app is the **Dash** interface at `http://127.0.0.1:8050`. The old Streamlit files are kept only as legacy reference.

## A. Install the app

Clone the repository and create the main environment:

```bash
git clone https://github.com/margaridaseabra/sleep_score_qc_app.git
cd sleep_score_qc_app
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
```

If the environment already exists, update it instead:

```bash
conda env update -f environment.yml --prune
conda activate sleep_stage_qc_v2
```

### Windows

Open **Anaconda Prompt** in the repository:

```bat
python check_setup.py
run_app_windows.bat
```

### macOS

Open Terminal in the repository:

```bash
python check_setup.py
./run_app.sh
```

Open `http://127.0.0.1:8050`.

## B. First workflow

1. Set the **Project root** and click **Load project**.
2. Open **1. Import .mat / EDF + Layer 1**.
3. Import a recording.
4. Compute epoch features.
5. Run Layer 1.
6. Open **3. Somnotate** if automatic Wake/NREM/REM scoring is needed.
7. Open **2. QC / Review** to inspect and correct scoring.
8. Run **4. Dissociation** if disagreement/event review is needed.
9. Export Final scoring when review is complete.

Final scoring saves automatically to:

```text
project_root/recordings/<recording_id>/final_scoring.csv
```

## C. QC shortcuts

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

## D. Video review

1. Link the recording's video in the Video QC section and set/save the video offset if needed.
2. Select an interval in the main QC plot.
3. The app prepares or reuses a short local QC clip for that interval; the original video is not modified.
4. Use **Play selection** for continuous synchronized playback. A red tracer shows the current video position on the EEG/EMG panel.
5. Use **◀ Epoch**, **Epoch ▶**, or **Replay epoch** for epoch-by-epoch review.
6. Use **Clear local review cache** if you want to remove cached clips for the recording.

For long videos, the original may remain on a lab/network drive. The interactive review clip is stored locally under the user's OS cache directory, for example `%LOCALAPPDATA%\SleepStageQC\video_cache` on Windows. The cache is automatically limited to about 10 GB.

MP4/H.264 is the recommended source format. FFmpeg is included in the main environment for conversion and local QC-clip preparation.

## E. Install Somnotate if needed

Somnotate uses a **separate environment** from the Dash app. The tested upstream version is Somnotate 0.5.0 at commit:

```text
a20f33de62511d8c172e333896608b7fc166d0f0
```

From the app repository:

```bash
conda env create -f environment_somnotate.yml
```

Clone and pin Somnotate:

```bash
git clone https://github.com/paulbrodersen/somnotate.git
cd somnotate
git checkout a20f33de62511d8c172e333896608b7fc166d0f0
conda activate somnotate_env
python -m pip install -e . --no-deps
```

Then verify from the app repository:

```bash
conda activate sleep_stage_qc_v2
python check_setup.py --require-somnotate --somnotate-root /path/to/somnotate
```

Typical paths:

```text
Windows: C:\Users\YOUR_USER\somnotate
macOS:   /Users/YOUR_USER/somnotate
```

On Apple Silicon, create only the Somnotate environment as Intel/Rosetta:

```bash
conda env create --platform osx-64 -f environment_somnotate.yml
```

The main Dash environment should remain native.

## F. Somnotate existing-model setup

In the Somnotate tab:

- **Somnotate repository path**: your local Somnotate clone
- **Somnotate conda env**: `somnotate_env`
- **Optional Somnotate Python executable**: normally leave blank
- **Existing model**: choose the intended `.pickle`
- **Somnotate epoch sec**: must match the model (1 s, 2 s, or 5 s)

For a complete new scoring run, keep these steps selected:

```text
prepare
preprocess
score
probabilities
import-results
```

The app uses a temporary Somnotate pipeline copy and does not modify the Somnotate repository itself.

## G. Important model note

Models trained from the app save environment/version metadata automatically.

The two historical bundled models are marked **LEGACY** because they were serialized with scikit-learn 1.7.2 while the standardized Somnotate environment uses 1.6.1. They are retained for compatibility/testing, but a version-matched and scientifically validated model should be used for final analysis.

For details and troubleshooting, read `README.md` and `COMPATIBILITY.md`.
