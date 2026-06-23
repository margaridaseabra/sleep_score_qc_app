#!/usr/bin/env bash
set -e

# Legacy Streamlit version. Not recommended for new scoring.
if [ -f sleep_stage_qc_v2_app.py ]; then
  streamlit run sleep_stage_qc_v2_app.py
elif [ -f legacy_streamlit/sleep_stage_qc_v2_app.py ]; then
  streamlit run legacy_streamlit/sleep_stage_qc_v2_app.py
else
  echo "Legacy Streamlit app not found."
  exit 1
fi
