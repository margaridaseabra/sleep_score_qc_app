#!/usr/bin/env bash
set -euo pipefail
cat <<'MSG'
The Streamlit app is archived legacy code and is not part of the supported
Sleep Stage QC environment. Use the Dash app instead:

  ./run_app.sh

If you need the historical Streamlit code for reference, see legacy_streamlit/.
MSG
exit 1
