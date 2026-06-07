# Start Here

Use this when sharing the project with lab colleagues.

## What they need

- this repository
- a Conda or Python environment created from `environment.yml`
- a local Somnotate checkout
- a Somnotate Python executable or Conda environment
- the trained Somnotate model file if they are scoring with an existing model
- a recording project folder containing imported data

## Local folders in the repository

The repo includes two placeholder folders for local use:

- `test_data/` for example recordings or a small project to test the app
- `somnotate_models/` for pretrained Somnotate `.pickle` files

If you want someone to test the app right away, put one small project inside `test_data/` with one recording inside it.

The app still expects the usual project files inside that test project:

- `recordings_manifest.csv`
- `metadata.json`
- `eeg.npy`
- `emg.npy`
- `epoch_features.csv`
- `layer1_wake_sleep.csv`
- `manual_scoring_aligned.csv` if manual scoring exists
- `final_scoring.csv` if you want a reviewed scoring example
- `somnotate/somnotate_results_timeseries.csv` if Somnotate has been run
- `somnotate/somnotate_automated.tsv` depending on the Somnotate workflow

## Fastest setup

1. Clone the private GitHub repository.
2. Create the app environment:

```bash
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
```

3. Clone Somnotate:

```bash
git clone https://github.com/paulbrodersen/somnotate.git
```

4. Launch the app:

```bash
bash run_app.sh
```

5. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## First time inside the app

1. Open the `1. Import .mat + Layer 1` tab.
2. Enter your project root folder.
3. Paste the path to the preprocessed `.mat` file.
4. Run `1. Import .mat`, then `2. Compute epoch features`, then `3. Run Layer 1`.
5. Move to the `2. QC viewer` tab and confirm the recording appears.
6. Open the `3. Somnotate` tab only after the app environment is working.

## How to fill the Somnotate tab

### If you already have a trained model

1. Paste the Somnotate repository folder path.
2. Paste the Somnotate Python executable path, or leave it as the Conda environment name if that works on your machine.
3. Paste the trained model file path.
4. Select the recording you want to score.
5. Click `Run Somnotate using existing model`.

### If you want to train a new model

1. Paste the Somnotate repository folder path.
2. Paste the Somnotate Python executable path, or use the Conda environment name.
3. Choose recordings that already have manual scoring.
4. Optionally choose test recordings.
5. Enter a new model name.
6. Click `Train Somnotate model`.

## What to do next

After Layer 1 or Somnotate is ready:

1. Go to `4. Review / Edit scoring`.
2. Inspect suspicious periods.
3. Accept Somnotate, manual scoring, or Layer 1 when appropriate.
4. Use manual labels when needed.
5. Export the final scoring.

## Somnotate choices

### Use an existing trained model

Use this when a reliable `.pickle` model already exists.

You will need:

- Somnotate repository path
- Somnotate Python environment or executable
- trained model file
- recordings to score

### Train a new model

Use this when the lab wants its own model trained from manual scoring.

You will need:

- Somnotate repository path
- Somnotate Python environment or executable
- training recordings with manual scoring
- optional test recordings
- model name

## How to verify it works

1. Open the app.
2. Import one test recording.
3. Run Layer 1.
4. Run Somnotate or attach existing Somnotate results.
5. Confirm that the QC viewer shows EEG, EMG, scoring rows, and probabilities.

If the test run succeeds, the colleague can use their own project folder next.
