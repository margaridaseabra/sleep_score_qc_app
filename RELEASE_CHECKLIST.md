# Release checklist

Use this checklist before merging `windows-hardening` into `main` and before creating the `v1.1.0` GitHub release.

## 1. Preserve the known-good branch

- [ ] Current working Windows state is committed and pushed to `origin/windows-hardening`.
- [ ] `git status` is clean before integrating later changes.
- [ ] No temporary `.patch`, backup, ZIP, recording data, videos, logs, or debug scripts are tracked.

## 2. Repository hygiene

- [ ] `VERSION` contains the intended release candidate/final version.
- [ ] `CHANGELOG.md` is updated.
- [ ] `README.md`, `START_HERE.md`, and `COMPATIBILITY.md` agree on setup instructions.
- [ ] `.gitignore` excludes generated recordings, local videos, caches, logs, raw scientific data, and development artifacts.
- [ ] `.gitattributes` is present so Python/shell/docs use LF and Windows batch files use CRLF.
- [ ] `python -m compileall -q dash_app pipelines check_setup.py check_windows_setup.py tools tests` passes.
- [ ] `python -m pytest -q` passes.
- [ ] `git diff --check` passes.

## 3. Sync with current main

- [ ] Fetch the latest remote branches/tags.
- [ ] Integrate the current `origin/main` into `windows-hardening` only after the known-good checkpoint is pushed.
- [ ] Resolve conflicts carefully, especially in `pipelines/10_somnotate_layer.py` and Somnotate training/environment work.
- [ ] Rerun compile, diagnostics, and the full test suite after integration.

## 4. Fresh-clone user test

Use a new directory outside the development checkout.

- [ ] Clone the branch/release from GitHub into a fresh folder.
- [ ] Create `sleep_stage_qc_v2` from `environment.yml`.
- [ ] `python check_setup.py` passes without relying on the development checkout.
- [ ] Windows: `run_app_windows.bat` starts Dash.
- [ ] macOS: `./run_app.sh` starts Dash.
- [ ] App opens at `http://127.0.0.1:8050`.
- [ ] No untracked/generated files are required for startup.

## 5. Somnotate setup

- [ ] `environment_somnotate.yml` solves on the target Windows machine.
- [ ] Somnotate checkout is version 0.5.0 at commit `a20f33de62511d8c172e333896608b7fc166d0f0`.
- [ ] `python check_setup.py --require-somnotate --somnotate-root <path>` passes.
- [ ] `python tools/check_somnotate_source_contract.py <path-to-somnotate>` passes.
- [ ] Existing-model scoring works.
- [ ] Existing-model evaluation works on manually scored recordings.
- [ ] Train-own-model workflow and recording-level CV complete on a small validation set.
- [ ] A release/scientific model has epoch/runtime metadata.
- [ ] Release/scientific model file is trusted and its checksum is recorded.
- [ ] No unsupported model/runtime mismatch is silently accepted.

## 6. Windows end-to-end

- [ ] Fresh `sleep_stage_qc_v2` environment creates from `environment.yml`.
- [ ] Project copied from macOS/external storage loads without editing `recordings_manifest.csv`.
- [ ] MATLAB import works.
- [ ] EDF/BDF import works.
- [ ] Epoch features compute.
- [ ] Layer 1 runs.
- [ ] Existing-model Somnotate workflow: prepare -> preprocess -> score -> probabilities -> import-results.
- [ ] Somnotate Wake/NREM/REM row and probability traces are correct.
- [ ] Apply Somnotate to selected interval and visible window works.
- [ ] Apply Layer 1 and Manual to visible window works.
- [ ] Manual Wake/NREM/REM editing works.
- [ ] Undo, Final reset, and Fill empty Final with Somnotate work without overwriting reviewed labels.
- [ ] Dissociation analysis and review-queue navigation work.
- [ ] Link/save video path and offset.
- [ ] Selecting an interval creates/reuses a short local QC clip.
- [ ] Synchronized EEG/EMG panel loads the selected interval.
- [ ] Play selection plays smoothly from the beginning of the selection.
- [ ] Red tracer moves continuously with video time.
- [ ] Previous/next epoch and Replay epoch work.
- [ ] Clear local review cache works and never modifies the source video.
- [ ] AVI/MOV conversion works when FFmpeg is available.
- [ ] CSV, MATLAB, and EDF+ export work.
- [ ] Stop/restart preserves saved Final scoring and saved video settings.

## 7. macOS regression

Repeat the same functional checks on macOS, with special attention to:

- [ ] Fresh app environment creation/update.
- [ ] `./run_app.sh` launch.
- [ ] Existing projects still load.
- [ ] Somnotate environment/check-out works on the target Mac architecture.
- [ ] On Apple Silicon, the `osx-64`/Rosetta Somnotate environment is explicitly tested.
- [ ] Somnotate existing-model workflow produces equivalent scoring/probability output to the validated Windows run for the same input/model.
- [ ] QC scoring/edit/undo/export behavior matches Windows.
- [ ] Local QC video cache is created under `~/Library/Caches/SleepStageQC/video_cache`.
- [ ] Synchronized video playback and red tracer behave like Windows.

## 8. Cross-platform scientific validation

For one fixed recording/model pair:

- [ ] Compare Windows vs macOS `somnotate_results_timeseries.csv` labels with `python tools/compare_somnotate_results.py <windows.csv> <mac.csv>`.
- [ ] Compare Windows vs macOS probability arrays numerically.
- [ ] Record maximum/mean absolute probability difference.
- [ ] Confirm Final scoring written from identical source labels is identical.

## 9. GitHub release

- [ ] GitHub Actions runs the complete pytest suite on Windows and macOS.
- [ ] Push the hardening branch.
- [ ] Open/review pull request into `main`.
- [ ] Merge only after Windows + macOS validation passes.
- [ ] Change `VERSION` from `1.1.0-rc1` to `1.1.0`.
- [ ] Update release-candidate wording in docs/changelog.
- [ ] Create tag `v1.1.0`.
- [ ] Create GitHub release notes from `CHANGELOG.md`.
