#!/usr/bin/env bash
# analyse.sh SESSION BROWSER_RUN PYTHON_RUN TAG: classify both runs, then the prefix curves
# of the stitch variants -> out/<TAG>_*.json
set -euo pipefail
cd "$(dirname "$0")"
SES="$1"; BR="$2"; PYR="$3"; TAG="$4"
PY="${PY:-/Users/mbrosch/Documents/GitKraken_mac/03_analysis-tools/MiniMe/.venv/bin/python}"
export PYTHONPATH=../../src
"$PY" classify.py --raw "$SES/raw" --browser "$BR/run" --python "$PYR" --stitch "$BR" \
  --out "out/${TAG}_classify.json" >/dev/null
for r in ${RULES:-base attach40 attach50}; do
  "$PY" prefix_stitch.py --raw "$SES/raw" --run "$BR" --rule "$r" --step "${STEP:-4}" \
    --out "out/${TAG}_prefix_$r.json" | tail -1
done
