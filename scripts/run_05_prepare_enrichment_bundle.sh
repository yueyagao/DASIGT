#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m dasigt.toppfun_transfer prepare "$@"

# 运行前更改enrichment_tool和summary_analysis
# bash scripts/run_05_prepare_enrichment_bundle.sh --config configs/config_ori.yaml