# Compatibility and reproducibility

## Release candidate

```text
Sleep Stage QC: 1.1.0-rc1
```

The goal of this release candidate is to make Windows a first-class target while preserving macOS behavior.

## Supported architecture

The app deliberately separates two runtimes:

| Runtime | Environment | Python | Purpose |
|---|---|---:|---|
| Dash app | `sleep_stage_qc_v2` | 3.11 | UI, import, QC, Layer 1, export |
| Somnotate | `somnotate_env` | 3.9 | Somnotate preprocessing/training/inference |

The separation is necessary because Somnotate 0.5.0 depends on `pomegranate=0.14.4`; the working Windows Conda build is incompatible with Python 3.10+. `environment_somnotate.yml` pins the validated runtime package versions so that new model training and inference do not drift silently.

### Apple Silicon

The legacy Conda packages for `pomegranate 0.14.x` are available for Intel macOS (`osx-64`) but not native Apple Silicon (`osx-arm64`). On M-series Macs, create only the Somnotate environment with `--platform osx-64` and run it under Rosetta. The Dash environment remains native. This architecture must be regression-tested before v1.1.0.

## Tested Somnotate upstream

```text
Somnotate version: 0.5.0
Git commit: a20f33de62511d8c172e333896608b7fc166d0f0
```

The integration should be revalidated before moving to another upstream commit/version.

`tools/check_somnotate_source_contract.py` verifies that the app compatibility patches still apply cleanly to the pinned upstream example pipeline; CI runs this check on both Windows and macOS.

## Compatibility fixes implemented in the app

### Portable project paths

`recordings/<recording_id>` is treated as the canonical recording location. Old absolute manifest paths from `/Volumes/...` or another Windows drive are used only as a fallback.

Generated Somnotate manifest paths are rebased to the current project root, so a project can be moved between macOS and Windows without manually editing each manifest.

### pandas editable-string columns

Final-scoring text columns are normalized to object/string-compatible dtypes before editing. This avoids `LossySetitemError`/string-assignment failures seen with newer pandas versions.

### Somnotate label normalization

Raw Somnotate labels are normalized to canonical app labels:

```text
awake   -> Wake
non-REM -> NREM
REM     -> REM
```

Probability columns are normalized in the same way.

### Current Somnotate example-pipeline signal layout

The tested Somnotate example pipeline currently expects frontal EEG + occipital EEG + EMG. This app uses EEG + EMG and its historical models use the corresponding two-signal feature layout.

The app therefore patches only a temporary pipeline copy to use frontal EEG + EMG.

### lspopt integer window length

The tested upstream preprocessing code can pass a floating-point `nperseg` to `lspopt` when sampling frequency is read as a float. The app patches the temporary script to convert the spectrogram window length to an integer.

### Long-video synchronized review

The original recording video may remain on local or network storage. For interactive QC, the app creates short H.264/yuv420p clips in a per-user local cache and maps clip time back to source-video and recording time. The original video is not changed.

Default cache roots are `%LOCALAPPDATA%\SleepStageQC\video_cache` on Windows, `~/Library/Caches/SleepStageQC/video_cache` on macOS, and `$XDG_CACHE_HOME/SleepStageQC/video_cache` (or `~/.cache/...`) on Linux. The app prunes the cache at approximately 10 GB.

The moving red video tracer is browser-side so continuous playback does not require high-frequency Python callbacks or repeated signal-file reloads. FFmpeg must be available in the main app environment to prepare clips.

## Model versioning policy

New models trained through the app receive a `.metadata.json` file containing the runtime versions and Somnotate Git commit used for training.

For app-trained models, known mismatches in Python major/minor, Somnotate, scikit-learn, or pomegranate are blocked by default during inference. `--allow-version-mismatch` exists only for advanced debugging.

### Legacy bundled models

The two bundled legacy `.pickle` files have identical SHA-256 hashes:

```text
87991043cc6ec428b26e0a989b956b340c62221c31c0232484274ca82d38f14f
```

They were observed to contain a scikit-learn estimator serialized with scikit-learn 1.7.2. The standardized Somnotate runtime currently uses scikit-learn 1.6.1.

Because scikit-learn does not support cross-version loading of pickled estimators, these models are marked `LEGACY` and should be replaced or explicitly validated before final scientific analysis.

Somnotate model files are Python pickle files and must be treated as executable/trusted artifacts. Do not load `.pickle` models obtained from untrusted sources.

## Platform validation status

### Windows

Validated during the 1.1.0-rc1 Windows hardening pass:

- clean app environment creation;
- Dash startup;
- project copied from macOS/external drive;
- portable recording resolution;
- Somnotate 0.5.0 setup in a separate Python 3.9 environment;
- EEG+EMG temporary pipeline compatibility;
- Somnotate preprocessing;
- existing-model inference;
- state probabilities;
- result import;
- state/probability normalization;
- Apply Somnotate to visible QC window.
- synchronized EEG/EMG + video review using short local QC clips;
- browser-side moving video tracer and Play selection;
- 13 automated tests passing in the validated Windows environment.

### macOS

The app was developed and used on macOS before this hardening pass. **A complete regression run of the 1.1.0-rc1 branch is still required before tagging v1.1.0.**

Run the full macOS section in `RELEASE_CHECKLIST.md` rather than assuming that a cross-platform code change is sufficient.

## Non-blocking EDF header warnings

pyEDFlib can warn that EDF physical min/max header values are being truncated to the EDF field width. These warnings did not stop Somnotate preprocessing in the validated Windows run. They should be kept visible in logs but are not currently classified as a pipeline failure.
