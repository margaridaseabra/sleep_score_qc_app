# Changelog

## 0.9.0-rc1 - 2026-08-13

Cross-platform hardening release candidate.

### Added

- Windows launcher and cross-platform setup diagnostics.
- Separate reproducible Somnotate environment file, including Apple Silicon/Rosetta guidance.
- Application `VERSION` file and version display.
- Somnotate runtime preflight and tested-upstream reporting.
- Model training metadata containing package/runtime versions and Somnotate Git commit.
- Legacy model metadata and warnings.
- Compatibility/reproducibility documentation and cross-platform result comparison tool.
- GitHub Actions smoke tests on Windows and macOS.
- Structured GitHub bug-report template with setup/log fields.
- `.gitattributes` line-ending rules for Windows/macOS source portability.

### Fixed

- Projects copied from macOS to Windows no longer rely on stale absolute `recording_dir` entries.
- MAT/EDF imports now write portable relative recording paths.
- Final-scoring text columns remain writable with newer pandas versions.
- Somnotate `awake`/`non-REM` labels and probability columns are normalized to `Wake`/`NREM`.
- Somnotate's current three-signal example configuration is adapted to this app's EEG+EMG workflow in a temporary pipeline copy.
- Floating-point Somnotate/lspopt spectrogram window lengths are converted to integers.
- Generated Somnotate manifests are rebased after moving a project between machines.
- Somnotate repository/model paths are validated before launching long workflows.
- Scoring feedback is displayed next to the visible-window scoring controls.
- Windows launcher/setup scripts no longer contain stray leading characters.
- FFmpeg documentation now uses a one-line command that works in both Windows shells and macOS/Linux terminals.

### Known issue before v1.0.0

The historical bundled Somnotate model was serialized with scikit-learn 1.7.2, while the standardized Python 3.9 Somnotate environment uses scikit-learn 1.6.1. Retrain or validate a release model in the standardized environment before final scientific release.
