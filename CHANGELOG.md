# Changelog

## 1.1.0-rc1 - 2026-08-18

- Added one-click Windows setup with `INSTALL_WINDOWS.bat` and `RUN_APP.bat`; Somnotate is downloaded, pinned, installed, and configured automatically.

Release candidate for the next cross-platform release after `v1.0.1`.

### Added

- Windows launcher and cross-platform setup diagnostics.
- Separate reproducible Somnotate environment file, including Apple Silicon/Rosetta guidance.
- Somnotate existing-model evaluation and training-quality workflows.
- Recording-level model QC metrics and metadata for newly trained models.
- Somnotate runtime preflight and tested-upstream reporting.
- Compatibility/reproducibility documentation and cross-platform result comparison tooling.
- Synchronized EEG/EMG + video review with short per-user local QC clips for long recordings.
- Local QC-clip reuse, automatic cache pruning, and a clear-cache control.
- Browser-side moving video tracer plus Play selection, previous/next epoch, and replay-epoch controls.
- Automated tests for portable Somnotate recording resolution, Somnotate training/QC export, and synchronized video-review cache behavior.
- GitHub bug-report template and cross-platform CI.
- `.gitattributes` line-ending rules for Windows/macOS source portability.

### Fixed

- Projects copied from macOS to Windows no longer rely on stale absolute `recording_dir` entries.
- MAT/EDF imports write portable relative recording paths.
- Final-scoring text columns remain writable with newer pandas versions.
- Somnotate `awake`/`non-REM` labels and probability columns normalize to `Wake`/`NREM`.
- Somnotate's current three-signal example configuration is adapted to the app's EEG+EMG workflow in a temporary pipeline copy.
- Floating-point Somnotate/lspopt spectrogram window lengths are converted to integers.
- Generated Somnotate manifests are rebased after moving a project between machines.
- Somnotate repository/model paths are validated before long workflows start.
- Scoring feedback is displayed next to visible-window scoring controls.
- Long-video review no longer requires repeated random seeking through a multi-hour source file.

### Release-candidate notes

- Windows setup diagnostics pass in the validated environment.
- The current Windows automated suite passes 13 tests.
- Full macOS regression and cross-platform Somnotate output comparison are still required before `v1.1.0`.
- The historical bundled Somnotate models were serialized with scikit-learn 1.7.2, while the standardized Python 3.9 Somnotate environment uses scikit-learn 1.6.1. Use a version-matched/validated model for final scientific analysis.

## 1.0.1 - 2026-07-16

- Simplified import sampling-rate fields and photometry terminology.

## 1.0.0 - 2026-07-16

- First stable cross-platform release.
- Added local AVI conversion and the initial Windows/macOS support path.
