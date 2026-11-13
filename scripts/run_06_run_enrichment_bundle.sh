#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m dasigt.toppfun_transfer enrich "$@"


# 运行前更改config中的enrichment_tool
# bash scripts/run_06_run_enrichment_bundle.sh --bundle-dir ./summary_table_ori_toppgene/analysis_1/toppfun_transfer_bundle --config configs/config_ori.yaml --output-dir ./summary_table_ori_toppgene/toppfun_1 --limit 1

# bash scripts/run_06_run_enrichment_bundle.sh --bundle-dir ./summary_table_ori_kegg/analysis_1/toppfun_transfer_bundle --config configs/config_ori.yaml --output-dir ./summary_table_ori_kegg/kegg_1
# bash scripts/run_06_run_enrichment_bundle.sh --bundle-dir ./summary_table_ori_kegg/analysis_1/toppfun_transfer_bundle --config configs/config_ori.yaml --output-dir ./summary_table_ori_kegg/kegg_1


# test kegg mapping
# python ./scripts/diagnose_kegg_mapping.py --cache-dir ./summary_table_ori_kegg/kegg_1/cache --gene-file ./summary_table_ori_kegg/analysis_1/toppfun_transfer_bundle/input_gene_lists/BLCA_fold0_none_module_aligned_0.txt