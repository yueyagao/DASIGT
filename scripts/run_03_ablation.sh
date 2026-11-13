#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m dasigt.ablation "$@"

# bash scripts/run_03_ablation.sh --config configs/config_abla_4.yaml