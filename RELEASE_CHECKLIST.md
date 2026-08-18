# Release Checklist

Use this before sharing a release with the lab.

## Repository

- [ ] `VERSION` contains the intended release number.
- [ ] README and START_HERE describe the current supported workflow.
- [ ] No patches, ZIP backups, recordings, videos, caches, logs, or personal machine paths are tracked.
- [ ] Obsolete/legacy development files are removed.
- [ ] `.gitignore` protects raw recording and generated-data files.

## Windows installer

- [ ] Fresh GitHub ZIP contains `INSTALL_WINDOWS.bat` and `RUN_APP.bat`.
- [ ] `INSTALL_WINDOWS.bat` succeeds on Windows.
- [ ] Running `INSTALL_WINDOWS.bat` a second time also succeeds.
- [ ] App environment is created/updated automatically.
- [ ] Somnotate environment is created/updated automatically.
- [ ] Supported Somnotate source is downloaded automatically.
- [ ] Somnotate is pinned to `a20f33de62511d8c172e333896608b7fc166d0f0`.
- [ ] Final setup diagnostics report `Environment looks ready`.
- [ ] `RUN_APP.bat` starts the app without manual environment activation.
- [ ] Browser opens at `http://127.0.0.1:8050`.

## Functional Windows regression

- [ ] Project loads.
- [ ] MAT import works.
- [ ] EDF import works.
- [ ] Layer 1 works.
- [ ] Existing-model Somnotate scoring works.
- [ ] Existing-model evaluation works.
- [ ] Model training/QC workflow works.
- [ ] Wake/NREM/REM labels normalize correctly.
- [ ] Probability traces load.
- [ ] Visible-window source-scoring buttons work.
- [ ] Manual scoring works.
- [ ] Undo/reset works.
- [ ] Dissociation workflow works.
- [ ] Final scoring export works.
- [ ] Synchronized video review works.
- [ ] Local short QC clips are created and reused.
- [ ] EEG/EMG review panel loads.
- [ ] Red browser-side playhead moves with the video.
- [ ] Video cache can be cleared.

## Automated checks

- [ ] `python -m compileall -q dash_app pipelines check_setup.py check_windows_setup.py tools tests`
- [ ] `python -m pytest -q`
- [ ] `python check_setup.py`
- [ ] `git diff --check`
- [ ] GitHub Actions passes on Windows.
- [ ] GitHub Actions passes on macOS.

## macOS regression

- [ ] Fresh GitHub checkout/ZIP tested.
- [ ] App environment can be created.
- [ ] App launches.
- [ ] Project loading works.
- [ ] Layer 1 works.
- [ ] Somnotate workflow works using the documented supported environment.
- [ ] Video review works.
- [ ] Export works.
- [ ] Same-model comparison against validated Windows output is acceptable.

## Final release

- [ ] Release branch is synchronized with current `main`.
- [ ] Conflicts are resolved and tests rerun.
- [ ] Release candidate is reviewed.
- [ ] PR merged to `main`.
- [ ] Final release version set.
- [ ] Git tag created.
- [ ] GitHub release notes published.
