"""Two-stage ToppFun workflow.

prepare: local-only; export portable gene lists and optional background gene sets from
         saved GMI assignment matrices.
enrich: network-enabled host; run ToppGene lookup/enrichment from a prepared bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import load_config
from .toppfun import ToppFunRunner


def _safe(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _assignment_path(cfg: Dict, dataset: str, fold: int, variant: str) -> Path:
    return Path(cfg["project"]["output_root"]) / dataset.upper() / f"fold_{int(fold)}" / variant / "gene_module_assignments.csv"


def _align_to_reference(assignment: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Match module columns to a reference fold using the same cosine/Hungarian strategy as analysis.py."""
    if assignment.empty or reference.empty or assignment.shape[1] != reference.shape[1]:
        out = assignment.copy()
        out.columns = [f"module_aligned_{j}" for j in range(out.shape[1])]
        return out
    shared = sorted(set(map(str, reference.index)) & set(map(str, assignment.index)))
    if not shared:
        out = assignment.copy()
        out.columns = [f"module_aligned_{j}" for j in range(out.shape[1])]
        return out
    ref = reference.copy()
    cur = assignment.copy()
    ref.index = ref.index.map(str)
    cur.index = cur.index.map(str)
    ref_lookup = {g: i for i, g in enumerate(ref.index)}
    cur_lookup = {g: i for i, g in enumerate(cur.index)}
    aa = np.stack([pd.to_numeric(ref.iloc[ref_lookup[g]], errors="coerce").to_numpy(float) for g in shared])
    bb = np.stack([pd.to_numeric(cur.iloc[cur_lookup[g]], errors="coerce").to_numpy(float) for g in shared])
    aa = aa / (np.linalg.norm(aa, axis=0, keepdims=True) + 1e-8)
    bb = bb / (np.linalg.norm(bb, axis=0, keepdims=True) + 1e-8)
    sim = aa.T @ bb
    rr, cc = linear_sum_assignment(-sim)
    order = np.arange(assignment.shape[1], dtype=int)
    for r, c in zip(rr, cc):
        order[r] = c
    aligned = assignment.iloc[:, order].copy()
    aligned.columns = [f"module_aligned_{j}" for j in range(aligned.shape[1])]
    return aligned


def prepare_bundle(config_path: str, output_dir: str = "", datasets: List[str] | None = None,
                   folds: List[int] | None = None, variants: List[str] | None = None) -> Path:
    """Prepare a portable gene-list bundle without any network access."""
    cfg = load_config(config_path)
    project = cfg["project"]
    analysis_cfg = cfg.get("analysis", {})
    tf_cfg = analysis_cfg.get("toppfun", {}) or {}
    datasets = [d.upper() for d in (datasets or project.get("datasets", []))]
    folds = [int(f) for f in (folds or cfg.get("training", {}).get("folds", [0, 1, 2, 3, 4]))]
    variants = [str(v) for v in (variants or tf_cfg.get("variants", ["none"]))]
    top_n = int(tf_cfg.get("top_genes_per_module", analysis_cfg.get("pathway_top_genes", 50)))
    max_modules = int(tf_cfg.get("max_modules_per_fold", 0))
    background_mode = str(tf_cfg.get("background", analysis_cfg.get("enrichment_background", "all_category_genes")))

    if top_n < 1:
        raise ValueError("top_genes_per_module must be >= 1")
    if not output_dir:
        output_dir = str(Path(project["summary_analysis"]) / "toppfun_transfer_bundle")
    bundle = Path(output_dir).resolve()
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "input_gene_lists").mkdir(parents=True)
    (bundle / "background_gene_lists").mkdir(parents=True)

    manifest_rows = []
    missing_files = []
    global reference_assignments
    reference_assignments = {}
    for ds in datasets:
        for fold in folds:
            for variant in variants:
                path = _assignment_path(cfg, ds, fold, variant)
                if not path.exists():
                    missing_files.append(str(path))
                    continue
                assignment = pd.read_csv(path, index_col=0)
                assignment.index = assignment.index.map(str)
                assignment = assignment.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                if assignment.empty or assignment.shape[1] == 0:
                    continue
                # Preserve the same per-(dataset, variant) module alignment used by analysis.py.
                reference_key = (ds, variant)
                if reference_key not in reference_assignments:
                    reference_assignments[reference_key] = assignment.copy()
                    aligned = assignment.copy()
                    aligned.columns = [f"module_aligned_{j}" for j in range(aligned.shape[1])]
                else:
                    aligned = _align_to_reference(assignment, reference_assignments[reference_key])
                module_cols = list(aligned.columns)
                if max_modules > 0:
                    module_cols = module_cols[:max_modules]
                background_genes = list(dict.fromkeys(g for g in assignment.index if g and g.lower() != "nan"))
                background_name = f"{_safe(ds)}_fold{fold}_{_safe(variant)}_background.txt"
                bg_rel = Path("background_gene_lists") / background_name
                (bundle / bg_rel).write_text("\n".join(background_genes) + ("\n" if background_genes else ""), encoding="utf-8")
                for module in module_cols:
                    values = aligned[module].to_numpy(dtype=float)
                    order = np.argsort(values)[::-1]
                    genes = []
                    for idx in order:
                        gene = str(aligned.index[idx]).strip()
                        if gene and gene.lower() != "nan" and gene not in genes:
                            genes.append(gene)
                        if len(genes) >= top_n:
                            break
                    module_name = str(module)
                    stem = f"{_safe(ds)}_fold{fold}_{_safe(variant)}_{_safe(module_name)}"
                    gene_rel = Path("input_gene_lists") / f"{stem}.txt"
                    (bundle / gene_rel).write_text("\n".join(genes) + ("\n" if genes else ""), encoding="utf-8")
                    manifest_rows.append({
                        "dataset": ds, "fold": fold, "ablation": variant, "module": module_name,
                        "top_n_requested": top_n, "input_gene_count": len(genes),
                        "input_file": gene_rel.as_posix(), "background_file": bg_rel.as_posix(),
                        "background_gene_count": len(background_genes), "assignment_source": str(path),
                    })

    if not manifest_rows:
        raise FileNotFoundError(
            "No gene_module_assignments.csv files were found for the requested datasets/folds/variants. "
            "Run DASIGT model analysis/training first and verify project.output_root."
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(bundle / "manifest.csv", index=False)
    # Store API settings with the bundle; secrets are not expected in this configuration.
    bundle_config = {
        "endpoint": tf_cfg.get("endpoint", "https://toppgene.cchmc.org/API"),
        "categories": tf_cfg.get("categories", ["GeneOntologyBiologicalProcess", "GeneOntologyMolecularFunction", "GeneOntologyCellularComponent", "Pathway"]),
        "correction": tf_cfg.get("correction", "Bonferroni"),
        "pvalue": float(tf_cfg.get("pvalue", 0.05)),
        "significance_qvalue": float(tf_cfg.get("significance_qvalue", 0.05)),
        "min_genes": int(tf_cfg.get("min_genes", 2)),
        "max_genes": int(tf_cfg.get("max_genes", 1500)),
        "max_results": int(tf_cfg.get("max_results", 100)),
        "top_genes_per_module": top_n,
        "background": background_mode,
        "lookup_batch_size": int(tf_cfg.get("lookup_batch_size", 250)),
        "request_interval_seconds": float(tf_cfg.get("request_interval_seconds", 0.2)),
        "timeout_seconds": float(tf_cfg.get("timeout_seconds", 30)),
        "retries": int(tf_cfg.get("retries", 2)),
        "use_cache": bool(tf_cfg.get("use_cache", True)),
        "variants": variants,
    }
    (bundle / "toppfun_config.json").write_text(json.dumps(bundle_config, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = f"""# DASIGT ToppFun Transfer Bundle\n\nThis bundle was prepared locally without network access.\n\n- Module gene lists: `input_gene_lists/`\n- Optional fold-specific background gene universes: `background_gene_lists/`\n- Manifest: `manifest.csv`\n- ToppFun parameters: `toppfun_config.json`\n\nRows: {len(manifest_rows)} module queries. Top genes per module: {top_n}. Background mode: {background_mode}.\n\nOn a network-enabled machine with DASIGT_Project installed, run:\n\n```bash\npython -m dasigt.toppfun_transfer enrich --bundle-dir . --output-dir toppfun_results\n```\n\nNo image/text feature tensors, patient-level predictions, clinical data, or full expression matrix are included.\n"""
    (bundle / "README.md").write_text(readme, encoding="utf-8")
    if missing_files:
        (bundle / "missing_assignment_files.txt").write_text("\n".join(missing_files) + "\n", encoding="utf-8")
        print(f"[ToppFun prepare] Missing assignment files: {len(missing_files)} (see {bundle / 'missing_assignment_files.txt'})")
    zip_path = bundle.parent / f"{bundle.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(bundle.rglob("*")):
            if file.is_file():
                zf.write(file, arcname=f"{bundle.name}/{file.relative_to(bundle).as_posix()}")
    print(f"[ToppFun prepare] Wrote {len(manifest_rows)} module gene lists.")
    print(f"[ToppFun prepare] Bundle directory: {bundle}")
    print(f"[ToppFun prepare] Transfer archive: {zip_path}")
    return zip_path


# Mutable only within a prepare invocation; each (dataset, ablation) has a reference fold.
reference_assignments: Dict[Tuple[str, str], pd.DataFrame] = {}


def enrich_bundle(bundle_dir: str, config_path: str = "", output_dir: str = "", limit: int = 0, continue_on_api_error: bool = False) -> Path:
    """Run ToppFun API on a previously prepared bundle; requires external network access."""
    bundle = Path(bundle_dir).resolve()
    manifest_path = bundle / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    bundle_cfg_path = bundle / "toppfun_config.json"
    bundle_cfg = json.loads(bundle_cfg_path.read_text(encoding="utf-8")) if bundle_cfg_path.exists() else {}
    if config_path:
        cfg = load_config(config_path)
        remote_cfg = dict(cfg.get("analysis", {}).get("toppfun", {}) or {})
        if bundle_cfg:
            # The bundle is authoritative for scientific settings (gene count, background,
            # categories, correction). Only transport/runtime settings may be overridden remotely.
            api_cfg = dict(bundle_cfg)
            for key in ("endpoint", "lookup_batch_size", "request_interval_seconds", "timeout_seconds", "retries", "use_cache"):
                if key in remote_cfg and remote_cfg[key] is not None:
                    api_cfg[key] = remote_cfg[key]
        else:
            api_cfg = remote_cfg
    else:
        api_cfg = bundle_cfg
    if not api_cfg:
        raise ValueError("No ToppFun configuration found; pass --config or include toppfun_config.json in the bundle.")
    if not output_dir:
        output_dir = str(Path.cwd() / "toppfun_results")
    out = Path(output_dir).resolve()
    runner = ToppFunRunner(api_cfg, out)
    manifest = pd.read_csv(manifest_path).fillna("")
    if limit > 0:
        manifest = manifest.head(limit)
    total = len(manifest)
    start_all = time.time()
    for i, row in enumerate(manifest.to_dict("records"), 1):
        gene_path = bundle / str(row["input_file"])
        genes = [x.strip() for x in gene_path.read_text(encoding="utf-8").splitlines() if x.strip()] if gene_path.exists() else []
        bg_path = bundle / str(row.get("background_file", "")) if row.get("background_file") else None
        bg = [x.strip() for x in bg_path.read_text(encoding="utf-8").splitlines() if x.strip()] if bg_path and bg_path.exists() else []
        metadata = {"dataset": str(row["dataset"]), "fold": int(row["fold"]), "ablation": str(row["ablation"]),
                    "module": str(row["module"]), "top_n_requested": int(row.get("top_n_requested", len(genes)))}
        print(f"[ToppFun enrich] {i}/{total} {metadata['dataset']} fold={metadata['fold']} variant={metadata['ablation']} module={metadata['module']} genes={len(genes)}", flush=True)
        qstart = time.time()
        result_rows, status, mapping_rows = runner.enrich_one_module(genes, bg, metadata)
        runner.result_rows.extend(result_rows)
        runner.status_rows.append(status)
        runner.mapping_rows.extend(mapping_rows)
        # Retain a copy of the exact submitted symbols in the result folder.
        target = runner.input_dir / gene_path.name
        if gene_path.exists() and gene_path.resolve() != target.resolve():
            shutil.copy2(gene_path, target)
        print(f"[ToppFun enrich] status={status.get('status')} terms={status.get('term_count', 0)} elapsed={time.time()-qstart:.1f}s", flush=True)
        if status.get("status") == "api_error" and not continue_on_api_error:
            print("[ToppFun enrich] Stopping after the first API error. Fix connectivity/request settings, then rerun; successful API responses are cached.", flush=True)
            break
    runner.write_outputs()
    summary = {"bundle": str(bundle), "output_dir": str(out), "modules_requested": total,
               "modules_processed": len(runner.status_rows),
               "elapsed_seconds": round(time.time() - start_all, 3),
               "successful": sum(r.get("status") == "ok" for r in runner.status_rows),
               "api_errors": sum(r.get("status") == "api_error" for r in runner.status_rows)}
    (out / "toppfun_transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ToppFun enrich] Finished: {summary}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare gene lists offline or enrich a transfer bundle with ToppFun API.")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Export module gene lists locally; no network requests are made.")
    prep.add_argument("--config", default="configs/config.yaml")
    prep.add_argument("--output-dir", default="")
    prep.add_argument("--datasets", nargs="+", default=None)
    prep.add_argument("--folds", nargs="+", type=int, default=None)
    prep.add_argument("--variants", nargs="+", default=None)
    enrich = sub.add_parser("enrich", help="Run ToppFun API for a prepared bundle on a network-enabled host.")
    enrich.add_argument("--bundle-dir", required=True)
    enrich.add_argument("--config", default="")
    enrich.add_argument("--output-dir", default="")
    enrich.add_argument("--limit", type=int, default=0, help="Optional number of modules for an API smoke test.")
    enrich.add_argument("--continue-on-api-error", action="store_true", help="Continue querying other modules after an API error; default is fail-fast.")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_bundle(args.config, args.output_dir, args.datasets, args.folds, args.variants)
    else:
        enrich_bundle(args.bundle_dir, args.config, args.output_dir, args.limit, args.continue_on_api_error)


if __name__ == "__main__":
    main()
