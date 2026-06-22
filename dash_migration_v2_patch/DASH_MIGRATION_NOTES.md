# Dash migration v2

This package contains a more complete Dash version of Sleep Stage QC v2.

## What is included

- Import `.mat` + Layer 1 tab that calls the existing pipeline scripts.
- QC / Review tab with fast Plotly selection, keyboard shortcuts, immediate scoring, undo, and export.
- Somnotate tab for existing models, training, and importing results.
- Dissociation tab that runs the existing dissociation pipeline and displays the output tables.
- About tab.

## Installation notes

The Dash app needs these extra packages compared with the original Streamlit environment:

```bash
pip install dash dash-bootstrap-components
```

Recommended run command:

```bash
bash run_dash_app.sh
```

Then open http://127.0.0.1:8050

## Safety

Keep the Streamlit app as a backup/reference version. This Dash version is meant to replace the interactive review workflow first, then gradually become the main app.
