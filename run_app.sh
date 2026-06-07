#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_FILE="$APP_DIR/sleep_stage_qc_v2_app.py"
ENV_NAME="${SLEEP_STAGE_QC_ENV:-sleep_stage_qc_v2}"

if command -v conda >/dev/null 2>&1; then
    exec conda run -n "$ENV_NAME" streamlit run "$APP_FILE"
fi

exec streamlit run "$APP_FILE"