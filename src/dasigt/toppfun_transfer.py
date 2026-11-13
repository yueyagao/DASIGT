"""Two-stage gene-module enrichment workflow.

prepare: offline export of GMI module gene lists and optional fold backgrounds.
enrich: select ToppGene or KEGG backend on a network-enabled machine.
"""
from __future__ import annotations
import argparse, json, re, shutil, time, zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .config import load_config
from .gene_selection import select_module_genes


def _safe(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _assignment_path(cfg, dataset, fold, variant):
    return Path(cfg["project"]["output_root"]) / dataset.upper() / f"fold_{int(fold)}" / variant / "gene_module_assignments.csv"


def _align_to_reference(assignment: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    cur = assignment.copy(); ref = reference.copy()
    cur.index = cur.index.map(str); ref.index = ref.index.map(str)
    if cur.empty or ref.empty or cur.shape[1] != ref.shape[1]:
        cur.columns = [f"module_aligned_{i}" for i in range(cur.shape[1])]
        return cur
    shared = sorted(set(cur.index) & set(ref.index))
    if not shared:
        cur.columns = [f"module_aligned_{i}" for i in range(cur.shape[1])]
        return cur
    aa = np.stack([pd.to_numeric(ref.loc[g], errors="coerce").to_numpy(float) for g in shared])
    bb = np.stack([pd.to_numeric(cur.loc[g], errors="coerce").to_numpy(float) for g in shared])
    aa = aa / (np.linalg.norm(aa, axis=0, keepdims=True) + 1e-8)
    bb = bb / (np.linalg.norm(bb, axis=0, keepdims=True) + 1e-8)
    row_ind, col_ind = linear_sum_assignment(-(aa.T @ bb))
    order = np.arange(cur.shape[1])
    for r, c in zip(row_ind, col_ind): order[r] = c
    out = cur.iloc[:, order].copy()
    out.columns = [f"module_aligned_{i}" for i in range(out.shape[1])]
    return out


def prepare_bundle(config_path, output_dir="", datasets=None, folds=None, variants=None):
    cfg = load_config(config_path)
    pcfg = cfg["project"]
    acfg = cfg.get("analysis", {})
    ecfg = dict(acfg.get("toppfun", {}) or {})
    tool = str(acfg.get("enrichment_tool", "toppgene")).lower()
    tool = {"toppgene": "toppgene_api", "toppfun_api": "toppgene_api", "kegg": "kegg_api"}.get(tool, tool)
    if tool == "kegg_api":
        kegg_cfg = dict(acfg.get("kegg", {}) or {})
        ecfg.update(kegg_cfg)
        # Do not accidentally inherit the ToppGene endpoint when no KEGG block was provided.
        ecfg.setdefault("endpoint", "https://rest.kegg.jp")
        if not kegg_cfg.get("endpoint"):
            ecfg["endpoint"] = "https://rest.kegg.jp"
    datasets = [str(x).upper() for x in (datasets or pcfg.get("datasets", []))]
    folds = [int(x) for x in (folds or cfg.get("training", {}).get("folds", [0,1,2,3,4]))]
    variants = [str(x) for x in (variants or ecfg.get("variants", ["none"]))]
    # The top-level pathway_top_genes is the shared control for both backends.
    # Backend-specific top_genes_per_module is only a fallback for older configs.
    top_n = int(acfg.get("pathway_top_genes", ecfg.get("top_genes_per_module", 50)))
    max_modules = int(ecfg.get("max_modules_per_fold", 0))
    background_mode = str(ecfg.get("background", acfg.get("enrichment_background", "all_category_genes")))
    if tool not in {"toppgene_api", "kegg_api"}:
        raise ValueError("analysis.enrichment_tool must be 'toppgene'/'toppgene_api' or 'kegg'/'kegg_api'")
    if top_n != -1 and top_n < 1:
        raise ValueError("analysis.pathway_top_genes must be -1 (all genes assigned to a module) or a positive integer")
    if not output_dir: output_dir = str(Path(pcfg["summary_analysis"]) / "toppfun_transfer_bundle")
    bundle = Path(output_dir).resolve()
    if bundle.exists(): shutil.rmtree(bundle)
    (bundle / "input_gene_lists").mkdir(parents=True)
    (bundle / "background_gene_lists").mkdir(parents=True)

    manifest_rows, missing_files = [], []
    references: Dict[Tuple[str,str], pd.DataFrame] = {}
    for ds in datasets:
        for fold in folds:
            for variant in variants:
                source = _assignment_path(cfg, ds, fold, variant)
                if not source.exists():
                    missing_files.append(str(source)); continue
                assignment = pd.read_csv(source, index_col=0)
                assignment.index = assignment.index.map(str)
                assignment = assignment.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                if assignment.empty or not assignment.shape[1]: continue
                key=(ds,variant)
                if key not in references:
                    references[key]=assignment.copy()
                    aligned=assignment.copy(); aligned.columns=[f"module_aligned_{i}" for i in range(aligned.shape[1])]
                else:
                    aligned=_align_to_reference(assignment,references[key])
                module_cols=list(aligned.columns)
                if max_modules > 0: module_cols=module_cols[:max_modules]
                background=list(dict.fromkeys(g for g in assignment.index if g and g.lower() != "nan"))
                bg_rel=Path("background_gene_lists") / f"{_safe(ds)}_fold{fold}_{_safe(variant)}_background.txt"
                (bundle/bg_rel).write_text("\n".join(background)+("\n" if background else ""),encoding="utf-8")
                for module in module_cols:
                    genes, selection_mode = select_module_genes(aligned, module, top_n)
                    rel=Path("input_gene_lists") / f"{_safe(ds)}_fold{fold}_{_safe(variant)}_{_safe(module)}.txt"
                    (bundle/rel).write_text("\n".join(genes)+("\n" if genes else ""),encoding="utf-8")
                    manifest_rows.append({"dataset":ds,"fold":fold,"ablation":variant,"module":str(module),"top_n_requested":top_n,
                        "gene_selection_mode":selection_mode,"input_gene_count":len(genes),"input_file":rel.as_posix(),"background_file":bg_rel.as_posix(),
                        "background_gene_count":len(background),"assignment_source":str(source)})
    if not manifest_rows:
        raise FileNotFoundError("No gene_module_assignments.csv found. Run DASIGT training/analysis first and verify project.output_root.")
    pd.DataFrame(manifest_rows).to_csv(bundle/"manifest.csv",index=False)
    if missing_files: (bundle/"missing_assignment_files.txt").write_text("\n".join(missing_files)+"\n",encoding="utf-8")
    bundle_cfg={"tool":tool,"endpoint":ecfg.get("endpoint","https://toppgene.cchmc.org/API" if tool=="toppgene_api" else "https://rest.kegg.jp"),
        "categories":ecfg.get("categories",["GeneOntologyBiologicalProcess","GeneOntologyMolecularFunction","GeneOntologyCellularComponent","Pathway"]),
        "correction":ecfg.get("correction","Bonferroni"),"pvalue":float(ecfg.get("pvalue",0.05)),
        "significance_qvalue":float(ecfg.get("significance_qvalue",0.05)),"min_genes":int(ecfg.get("min_genes",2)),
        "max_genes":int(ecfg.get("max_genes",1500)),"max_results":int(ecfg.get("max_results",100)),"top_genes_per_module":top_n,
        "background":background_mode,"lookup_batch_size":int(ecfg.get("lookup_batch_size",250)),
        "request_interval_seconds":float(ecfg.get("request_interval_seconds",0.4 if tool=="kegg_api" else 0.2)),
        "timeout_seconds":float(ecfg.get("timeout_seconds",30)),"retries":int(ecfg.get("retries",2)),
        "use_cache":bool(ecfg.get("use_cache",True)),"variants":variants,"organism":ecfg.get("organism","hsa")}
    (bundle/"enrichment_config.json").write_text(json.dumps(bundle_cfg,indent=2,ensure_ascii=False),encoding="utf-8")
    selection_desc = ("all genes hard-assigned to each module by row-wise argmax of the soft membership matrix" if top_n == -1 else f"top {top_n} genes by module membership score")
    (bundle/"README.md").write_text(f"# DASIGT enrichment transfer bundle\n\nTool: `{tool}`. Module queries: {len(manifest_rows)}. Gene selection: {selection_desc}. Background: {background_mode}.\n\nWhen `pathway_top_genes=-1`, each gene is assigned to exactly one module for enrichment-list construction using the highest soft membership score; ties go to the first module column. This does not change the model's soft assignments or training.\n\nOn a network-enabled server, run `python -m dasigt.toppfun_transfer enrich --bundle-dir . --output-dir enrichment_results`.\n",encoding="utf-8")
    archive=bundle.parent/f"{bundle.name}.zip"
    if archive.exists(): archive.unlink()
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(bundle.rglob("*")):
            if file.is_file(): zf.write(file,arcname=f"{bundle.name}/{file.relative_to(bundle).as_posix()}")
    print(f"[prepare] tool={tool}; module lists={len(manifest_rows)}; bundle={bundle}; zip={archive}",flush=True)
    if missing_files: print(f"[prepare] missing assignment files={len(missing_files)}; see missing_assignment_files.txt",flush=True)
    return archive


def _make_runner(tool, config, output_dir):
    if tool == "toppgene_api":
        from .toppfun import ToppFunRunner
        return ToppFunRunner(config, output_dir)
    if tool == "kegg_api":
        from .kegg import KEGGEnrichmentRunner
        return KEGGEnrichmentRunner(config, output_dir)
    raise ValueError(f"Unsupported enrichment tool: {tool}")


def enrich_bundle(bundle_dir, config_path="", output_dir="", limit=0, continue_on_api_error=False):
    bundle=Path(bundle_dir).resolve()
    manifest_path=bundle/"manifest.csv"
    if not manifest_path.exists(): raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    bundle_cfg=json.loads((bundle/"enrichment_config.json").read_text(encoding="utf-8"))
    tool=str(bundle_cfg.get("tool","toppgene_api")).lower()
    tool={"toppgene":"toppgene_api","toppfun_api":"toppgene_api","kegg":"kegg_api"}.get(tool,tool)
    if config_path:
        cfg=load_config(config_path)
        remote_key = "kegg" if tool == "kegg_api" else "toppfun"
        remote=(cfg.get("analysis",{}).get(remote_key,{}) or {})
        # Tool and statistical settings are pinned in the transfer bundle. Runtime settings may be overridden.
        for key in ("endpoint","request_interval_seconds","timeout_seconds","retries","use_cache","organism"):
            if remote.get(key) is not None: bundle_cfg[key]=remote[key]
    if not output_dir: output_dir=str(Path.cwd()/"enrichment_results")
    out=Path(output_dir).resolve()
    runner=_make_runner(tool,bundle_cfg,out)
    manifest=pd.read_csv(manifest_path).fillna("")
    if limit>0: manifest=manifest.head(limit)
    started=time.time()
    for i,row in enumerate(manifest.to_dict("records"),1):
        gene_path=bundle/str(row["input_file"])
        genes=[x.strip() for x in gene_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        bg_path=bundle/str(row.get("background_file","")) if row.get("background_file") else None
        background=[x.strip() for x in bg_path.read_text(encoding="utf-8").splitlines() if x.strip()] if bg_path and bg_path.exists() else []
        meta={"dataset":str(row["dataset"]),"fold":int(row["fold"]),"ablation":str(row["ablation"]),"module":str(row["module"]),"top_n_requested":int(row.get("top_n_requested",len(genes)))}
        print(f"[{tool}] {i}/{len(manifest)} {meta['dataset']} fold={meta['fold']} module={meta['module']} n_genes={len(genes)}",flush=True)
        t0=time.time()
        rows,status,maps=runner.enrich_one_module(genes,background,meta)
        runner.result_rows.extend(rows); runner.status_rows.append(status); runner.mapping_rows.extend(maps)
        target=runner.input_dir/gene_path.name
        if gene_path.resolve()!=target.resolve(): shutil.copy2(gene_path,target)
        print(f"[{tool}] status={status.get('status')} terms={status.get('term_count',0)} elapsed={time.time()-t0:.1f}s",flush=True)
        if status.get("status")=="api_error" and not continue_on_api_error:
            print(f"[{tool}] stopping after API error; successful cache entries are retained. Fix connectivity and rerun.",flush=True); break
    runner.write_outputs()
    summary={"tool":tool,"bundle":str(bundle),"output_dir":str(out),"modules_requested":len(manifest),"modules_processed":len(runner.status_rows),
        "elapsed_seconds":round(time.time()-started,3),"successful":sum(x.get("status")=="ok" for x in runner.status_rows),
        "api_errors":sum(x.get("status")=="api_error" for x in runner.status_rows)}
    (out/"enrichment_transfer_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"[{tool}] finished: {summary}",flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description="Prepare module gene lists offline or run ToppGene/KEGG enrichment on a network-enabled host")
    sub=p.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare",help="Offline export; no network access")
    prep.add_argument("--config",default="configs/config.yaml"); prep.add_argument("--output-dir",default="")
    prep.add_argument("--datasets",nargs="+",default=None); prep.add_argument("--folds",nargs="+",type=int,default=None); prep.add_argument("--variants",nargs="+",default=None)
    enr=sub.add_parser("enrich",help="Run selected enrichment backend from a bundle")
    enr.add_argument("--bundle-dir",required=True); enr.add_argument("--config",default=""); enr.add_argument("--output-dir",default="")
    enr.add_argument("--limit",type=int,default=0); enr.add_argument("--continue-on-api-error",action="store_true")
    a=p.parse_args()
    if a.command=="prepare": prepare_bundle(a.config,a.output_dir,a.datasets,a.folds,a.variants)
    else: enrich_bundle(a.bundle_dir,a.config,a.output_dir,a.limit,a.continue_on_api_error)

if __name__=="__main__": main()
