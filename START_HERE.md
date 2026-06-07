# Start Here

Use this when sharing the project with lab colleagues.

## What they need

- this repository
- a Conda or Python environment created from `environment.yml`
- a local Somnotate checkout
- a Somnotate Python executable or Conda environment
- the trained Somnotate model file if they are scoring with an existing model
- a recording project folder containing imported data

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
