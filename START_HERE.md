# Start Here

Use this when sharing the project with lab colleagues.

## What you need

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

## Adapting Somnotate for one EEG + one EMG

Some Somnotate examples may assume a recording setup with more than one EEG channel, for example two EEG channels plus one EMG channel. In our app, the expected format is simpler:

* one EEG signal
* one EMG signal

This is enough to run Somnotate, as long as the Somnotate configuration and the trained model match this signal setup.

### Where Somnotate defines the input signals

In Somnotate, the preprocessing script reads the list of signals from `configuration.py`, specifically from:

```python
state_annotation_signals
```

The preprocessing script then checks that the spreadsheet contains one column for each signal listed in `state_annotation_signals`.

For example, if `configuration.py` contains something like:

```python
state_annotation_signals = [
    "frontal_eeg_signal_label",
    "parietal_eeg_signal_label",
    "emg_signal_label",
]
```

then the spreadsheet must contain all three columns:

```text
frontal_eeg_signal_label
parietal_eeg_signal_label
emg_signal_label
```

This is the setup for two EEG channels plus one EMG channel.

For our usual recordings, with one EEG and one EMG, this should instead be changed to:

```python
state_annotation_signals = [
    "frontal_eeg_signal_label",
    "emg_signal_label",
]
```

or to the equivalent EEG/EMG label names used in your local Somnotate configuration.

### What the Somnotate spreadsheet should contain

For one EEG + one EMG, the Somnotate spreadsheet should contain the standard file/path columns:

```text
file_path_raw_signals
sampling_frequency_in_hz
file_path_preprocessed_signals
```

and the two signal-label columns:

```text
frontal_eeg_signal_label
emg_signal_label
```

For example:

```text
file_path_raw_signals,sampling_frequency_in_hz,file_path_preprocessed_signals,frontal_eeg_signal_label,emg_signal_label
/path/to/recording.edf,512,/path/to/preprocessed.npy,EEG,EMG
```

The exact signal names, for example `EEG` and `EMG`, must match the channel labels inside the EDF file.

### Why this works

The Somnotate preprocessing script loads only the signals listed in `state_annotation_signals`:

```python
signal_labels = [dataset[column_name] for column_name in state_annotation_signals]
raw_signals = load_raw_signals(dataset["file_path_raw_signals"], signal_labels)
```

It then computes a spectrogram for each signal and concatenates the features. Therefore, if `state_annotation_signals` contains only one EEG and one EMG, Somnotate will preprocess only those two signals.

### Important model compatibility note

The Somnotate model must be trained with the same signal setup that is used later for scoring.

This means:

* a model trained with one EEG + one EMG should be used with one EEG + one EMG;
* a model trained with two EEGs + one EMG should not be used directly with one EEG + one EMG unless the configuration and feature structure are adapted;
* if you change the number or type of signals, it is usually best to train a new model with that same configuration.

The pretrained Somnotate models included with this app were trained using the signal structure expected by the app.

### Do not add fake channels

Do not create fake EEG or EMG channels just to satisfy a Somnotate configuration. A dummy or duplicated signal changes the input feature space and can make the model unreliable.

The preferred solution is to adapt the Somnotate configuration so that it uses the real available signals:

```text
one EEG + one EMG
```

python sleep_scoring_qc_app/pipelines/20_prepare_somnotate_recording.py --project-root . --recording-id <RECORDING_ID>
```

If you prefer to run Somnotate using only EEG and no EMG, you would need to adapt the Somnotate `configuration.py` and pipeline scripts to remove EMG from preprocessing. That is more invasive — creating a dummy `emg.npy` is the simplest and safest approach.
