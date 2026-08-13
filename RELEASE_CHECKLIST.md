# Release checklist

Use this checklist before merging the cross-platform branch into `main` and before creating a GitHub release.

## Repository hygiene

- [ ] `VERSION` contains the intended release version.
- [ ] `CHANGELOG.md` is updated.
- [ ] `README.md`, `START_HERE.md`, and `COMPATIBILITY.md` agree on setup instructions.
- [ ] No temporary `.patch`, backup, local data, logs, or debug scripts are tracked.
- [ ] `.gitattributes` is present so Python/shell files use LF and Windows batch files use CRLF.
- [ ] `python -m compileall -q dash_app pipelines check_setup.py check_windows_setup.py tools tests` passes.
- [ ] `python check_setup.py` passes in a fresh app environment.

## Somnotate setup

- [ ] `environment_somnotate.yml` solves on the target Windows machine.
- [ ] Somnotate checkout is version 0.5.0 at commit `a20f33de62511d8c172e333896608b7fc166d0f0`.
- [ ] `python check_setup.py --require-somnotate --somnotate-root <path>` passes.
- [ ] `python tools/check_somnotate_source_contract.py <path-to-somnotate>` passes.
- [ ] A release model has epoch/runtime metadata.
- [ ] Release model file is from a trusted source and its checksum is recorded.
- [ ] No `InconsistentVersionWarning` occurs for the release model, or the model has been explicitly validated and the exception documented.

## Windows end-to-end

- [ ] Fresh `sleep_stage_qc_v2` environment creates from `environment.yml`.
- [ ] `run_app_windows.bat` starts the Dash app.
- [ ] Project copied from macOS/external storage loads without editing `recordings_manifest.csv`.
- [ ] MATLAB import works.
- [ ] EDF/BDF import works.
- [ ] Epoch features compute.
- [ ] Layer 1 runs.
- [ ] Existing-model Somnotate workflow: prepare -> preprocess -> score -> probabilities -> import-results.
- [ ] Somnotate row is continuous/correctly colored for Wake/NREM/REM.
- [ ] Somnotate probability traces for Wake/NREM/REM are shown.
- [ ] Apply Somnotate to selected interval works.
- [ ] Apply Somnotate to visible window works.
- [ ] Apply Layer 1 and Manual to visible window works.
- [ ] Manual Wake/NREM/REM editing works.
- [ ] Undo works.
- [ ] Final reset works.
- [ ] Fill empty Final with Somnotate preserves already reviewed labels.
- [ ] Dissociation analysis runs and review queue navigation works.
- [ ] MP4 playback works.
- [ ] AVI conversion works when FFmpeg is available.
- [ ] CSV export works.
- [ ] MATLAB export works.
- [ ] EDF+ export works.
- [ ] Stop/restart preserves saved Final scoring.

## macOS regression

Repeat the same functional checks on macOS, with special attention to:

- [ ] Fresh app environment creation/update.
- [ ] `./run_app.sh` launch.
- [ ] Existing projects still load.
- [ ] Somnotate environment/check-out still works on the target Mac architecture.
- [ ] On Apple Silicon, the `osx-64`/Rosetta Somnotate environment is explicitly tested.
- [ ] Somnotate existing-model workflow produces the same scoring/probability output as the validated Windows run for the same input/model.
- [ ] QC scoring/edit/undo/export behavior matches Windows.
- [ ] Video playback/conversion still works.

## Cross-platform scientific validation

For one fixed recording/model pair:

- [ ] Compare Windows vs macOS `somnotate_results_timeseries.csv` state labels using `python tools/compare_somnotate_results.py <windows.csv> <mac.csv>`.
- [ ] Compare Windows vs macOS probability arrays numerically.
- [ ] Record maximum/mean absolute probability difference.
- [ ] Confirm Final scoring written from identical source labels is identical.

## GitHub release

- [ ] Push the hardening branch.
- [ ] Open/review pull request into `main`.
- [ ] Merge only after Windows + macOS sections pass.
- [ ] Change `VERSION` from release candidate to final version.
- [ ] Create tag, e.g. `v1.0.0`.
- [ ] Create GitHub release notes from `CHANGELOG.md`.
