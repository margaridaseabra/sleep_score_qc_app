# Sleep Scoring QC app
Sleep Stage QC v2 is a Streamlit app for inspecting, comparing, correcting, and exporting EEG/EMG sleep-stage scoring.

If you are setting this up for the first time, start with [START_HERE.md](START_HERE.md).

It includes:

- import of preprocessed `.mat` recordings
- Layer 1 Wake/Sleep scoring
- QC visualisation of EEG, EMG, scoring rows, and probabilities
- Somnotate integration for Wake/NREM/REM scoring
- review, undo, dissociation analysis, and export

## Quick Start

### 1. Clone the repository

```bash
git clone <https://github.com/margaridaseabra/sleep_score_qc_app>
cd sleep_stage_qc_v2
```

### 2. Create the Python environment

If you use Conda:

```bash
conda env create -f environment.yml
conda activate sleep_stage_qc_v2
```

If you prefer pip, install the same packages listed in `environment.yml` into your own environment.

### 3. Launch the app

```bash
streamlit run sleep_stage_qc_v2_app.py
```

If you use Conda, you can also run the helper script:

```bash
bash run_app.sh
```

When the app opens in the browser, the typical workflow is:

1. Open `1. Import .mat + Layer 1`.
2. Set the project root.
3. Select the `.mat` file and run the import steps in order.
4. Open `2. QC viewer` to confirm the recording and Layer 1 output.
5. Open `3. Somnotate` to score with an existing model or train a new one.
6. Use `4. Review / Edit scoring` to inspect and correct the scoring.
7. Export the final results when you are done.

## What you need to have

To use the app, colleagues need:

- this repository
- a Python environment with the packages in `environment.yml`
- their recording project folder with imported data
- a preprocessed `.mat` file, or existing recordings already prepared by the import pipeline

## Local folders for test data and Somnotate models

The repository includes two tracked placeholder folders:

- `test_data/` for example recordings or a small project used to test the app
- `somnotate_models/` for pretrained Somnotate `.pickle` files

You can place your own local test project files inside `test_data/` and keep pretrained models inside `somnotate_models/`.

For a test project, the app still expects the usual recording structure inside the project folder:

- `recordings_manifest.csv`
- `metadata.json`
- `eeg.npy`
- `emg.npy`
- `epoch_features.csv`
- `layer1_wake_sleep.csv`

Optional files can also live in the same test project if you want to try more features:

- `manual_scoring_aligned.csv` if manual scoring exists
- `final_scoring.csv` if you want a reviewed scoring example
- `somnotate/somnotate_results_timeseries.csv` if Somnotate has been run
- `somnotate/somnotate_automated.tsv` depending on the Somnotate workflow

## Project workflow

1. Import the `.mat` file and run Layer 1.
2. Open the QC viewer and inspect EEG, EMG, and scoring.
3. Run Somnotate if available.
4. Use the Review/Edit tab to inspect suspicious periods and correct labels.
5. Export the final scoring as CSV and MAT.

## First-run checklist

Before colleagues start using the app, make sure they have:

- the repository cloned locally
- the Conda environment created from `environment.yml`
- the app launched with `streamlit run sleep_stage_qc_v2_app.py` or `bash run_app.sh`
- the Somnotate repository cloned locally
- a Somnotate Python environment or executable path
- either an existing trained Somnotate model or manual scoring files for training
- a test recording or sample project to confirm everything works end to end

## Somnotate setup

Somnotate is not bundled as a Python dependency of this app. It must be installed from the official Somnotate repository and run in its own Python environment.

### 1. Download Somnotate

```bash
git clone https://github.com/paulbrodersen/somnotate.git
```

The app expects a local Somnotate checkout so it can call the external example pipeline scripts.

### 2. Create a Somnotate Python environment

Follow the installation instructions in the Somnotate repository. The app can use either:

- the full path to the Somnotate Python executable, or
- the name of a Conda environment such as `somnotate_env`

### 3. Prepare or train a model

Use Somnotate in one of two ways:

- existing trained model: provide the trained `.pickle` model file
- new model: provide manually scored training recordings and optionally test recordings

### 4. Point the app to Somnotate

In the Somnotate tab of the app, provide:

- Somnotate repository folder
- Somnotate Python executable or Conda environment
- trained model file if using an existing model
- training and optional test recordings if training a new model

### 5. Run Somnotate from the app

The app prepares the recordings, launches the external Somnotate pipeline, imports the results, and displays them together with the raw signals and other scoring layers.

## Somnotate workflows

### Use an existing trained model

Use this if a reliable model already exists. You will need:

- Somnotate repository path
- Somnotate Python environment
- trained model file
- one or more recordings to score

This is the fastest way to score new recordings.

### Train a new Somnotate model

Use this if you want to train your own model from manually scored recordings. You will need:

- Somnotate repository folder
- Somnotate Python executable or environment
- training recordings with manual scoring
- optional test recordings
- a model name

After training, the model is saved in the project’s `somnotate_models` folder and can be reused later.

## Expected output folders

After import and scoring, a recording folder typically contains:

- `eeg.npy`
- `emg.npy`
- `metadata.json`
- `epoch_features.csv`
- `layer1_wake_sleep.csv`
- `manual_scoring_aligned.csv` if manual scoring exists
- `somnotate/` for Somnotate outputs
- `final_scoring.csv` for review/export

## Troubleshooting

- If the app cannot find Somnotate, check the Somnotate repository path and Python executable.
- If Somnotate training fails, confirm that the training recordings contain manual scoring.
- If EDF export fails, install `pyedflib` in the app environment.
