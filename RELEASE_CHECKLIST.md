# Release Checklist

Use this before sharing the app with the lab.

## Code package

- [ ] Private GitHub repository created
- [ ] README updated with setup and Somnotate instructions
- [ ] `environment.yml` committed
- [ ] `run_app.sh` committed and executable
- [ ] `.gitignore` added for generated files

## Somnotate package

- [ ] Somnotate repository URL documented
- [ ] Somnotate install steps documented
- [ ] One working Somnotate Python environment identified
- [ ] One tested trained model file identified, if relevant
- [ ] Training instructions included for new model workflow

## Validation

- [ ] App starts with `bash run_app.sh`
- [ ] Layer 1 import works on a test recording
- [ ] Somnotate existing-model workflow works on a test recording
- [ ] Somnotate training workflow works on a manually scored recording
- [ ] Export CSV and MAT work

