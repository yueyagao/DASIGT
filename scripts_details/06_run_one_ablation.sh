#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
DATASET="${1:-BRCA}"; FOLD="${2:-0}"; VARIANT="${3:-no_gene_prior}"
python -m dasigt.train --config configs/config.yaml --datasets "$DATASET" --folds "$FOLD" --ablation "$VARIANT"
