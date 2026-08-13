# Start here

This is the short setup guide for lab users of **Sleep Stage QC 0.9.0-rc1**.

## A. Install the app

```bash
git clone https://github.com/margaridaseabra/sleep_score_qc_app.git
cd sleep_score_qc_app
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
```

### Windows

Open Anaconda Prompt:

```bat
python check_setup.py
run_app_windows.bat
```

### macOS

Open Terminal:

```bash
python check_setup.py
./run_app.sh
```

Open `http://127.0.0.1:8050`.

## B. Install Somnotate if you need automatic Wake/NREM/REM scoring

Somnotate uses a separate environment.

```bash
conda env create -f environment_somnotate.yml
```

Clone the tested Somnotate checkout:

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

On Windows, a typical Somnotate path is:

```text
C:\Users\YOUR_USER\somnotate
```

On macOS:

```text
/Users/YOUR_USER/somnotate
```

If the Mac uses Apple Silicon, create the Somnotate environment for Intel/Rosetta instead of the native ARM platform:

```bash
conda env create --platform osx-64 -f environment_somnotate.yml
```

The normal app environment stays native.

## C. First workflow

1. Set the **Project root** and click **Load project**.
2. Open **1. Import .mat / EDF + Layer 1**.
3. Import a recording.
4. Compute epoch features.
5. Run Layer 1.
6. Open **3. Somnotate** if automatic Wake/NREM/REM scoring is needed.
7. Open **2. QC / Review** to inspect and correct scoring.
8. Run **4. Dissociation** if you want disagreement/event review.
9. Export Final scoring when review is complete.

## D. Somnotate existing-model setup

In the Somnotate tab:

- **Somnotate repository path**: your local Somnotate clone
- **Somnotate conda env**: `somnotate_env`
- **Optional Somnotate Python executable**: normally leave blank
- **Existing model**: choose the intended `.pickle`
- **Somnotate epoch sec**: must match the model (1 s, 2 s, or 5 s)

Keep these steps selected for a full new scoring run:

```text
prepare
preprocess
score
probabilities
import-results
```

The app uses a temporary Somnotate pipeline copy and does not modify the Somnotate repository itself.

## E. QC shortcuts

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

Final scoring saves automatically to:

```text
project_root/recordings/<recording_id>/final_scoring.csv
```

## F. Important model note

Models trained from the app now save environment/version metadata automatically.

The two historical bundled models are marked **LEGACY** because they were serialized with scikit-learn 1.7.2 while the standardized Somnotate environment uses 1.6.1. They can be used for compatibility testing, but a version-matched/retrained model should be used for final scientific release work.

For details, read `COMPATIBILITY.md`.
