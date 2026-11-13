from pathlib import Path
import os, re, json, shutil
import numpy as np
import pandas as pd
from .config import load_config
from .utils import patient_id, atomic_json


def first_existing_column(df, candidates):
    for c in candidates:
        if c and c in df.columns: return c
    return None


def discover_feature_files(folder):
    root=Path(folder)
    if not root.exists(): return {}
    out={}
    for p in root.rglob("*.pt"):
        pid=patient_id(p.name)
        if pid: out.setdefault(pid, []).append(p.resolve())
    return out


def load_gene2vec(path, expected_dim=None):
    genes=[]; vectors=[]; bad=0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts=line.strip().split()
            if len(parts)<3: continue
            try: vec=np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                bad+=1; continue
            if expected_dim and len(vec)!=expected_dim: continue
            if not np.isfinite(vec).all(): continue
            genes.append(parts[0]); vectors.append(vec)
    if not vectors: raise ValueError(f"No valid vectors parsed from {path}")
    dims={len(v) for v in vectors}
    if len(dims)!=1: raise ValueError(f"Gene2Vec vectors have inconsistent dimensions: {dims}")
    # Preserve first occurrence for duplicate names.
    seen={};
    for g,v in zip(genes,vectors): seen.setdefault(g,v)
    return seen


def load_expression(path, gene_map=None):
    df=pd.read_csv(path, index_col=0, low_memory=False)
    df.index=df.index.astype(str).str.strip()
    # TCGA expression tables can use SYMBOL|ENSEMBL, ?|ENSEMBL, or SYMBOL rows.
    mapped=[]
    for raw in df.index:
        left, sep, right=raw.partition("|")
        gene=left if left and left!="?" else (right if sep else left)
        if gene_map and raw in gene_map: gene=gene_map[raw]
        elif gene_map and gene in gene_map: gene=gene_map[gene]
        mapped.append(gene)
    df.index=mapped
    df=df.apply(pd.to_numeric, errors="coerce")
    df=df.replace([np.inf,-np.inf], np.nan)
    # Collapse duplicated symbols/IDs by mean, keeping all numeric expression columns.
    df=df.groupby(level=0, sort=False).mean()
    return df


def load_clinical(path, survival_cfg):
    df=pd.read_csv(path, low_memory=False)
    if len(df.columns) and str(df.columns[0]).startswith("Unnamed"):
        df=df.rename(columns={df.columns[0]:"case_id"})
    id_col=first_existing_column(df,[survival_cfg.get("clinical_id_col"),"case_id","Composite Element REF","bcr_patient_barcode","submitter_id"])
    if id_col is None: raise ValueError(f"No patient ID column found in {path}; columns={list(df.columns[:12])}")
    df["case_id"]=df[id_col].map(patient_id)
    event_col=first_existing_column(df,[survival_cfg.get("event_col"),"vital_status","event","status"])
    death_col=first_existing_column(df,[survival_cfg.get("death_time_col"),"days_to_death","days_to_death.value"])
    censor_col=first_existing_column(df,[survival_cfg.get("censor_time_col"),"days_to_last_followup","days_to_last_follow_up","days_to_last_known_alive"])
    time_col=first_existing_column(df,[survival_cfg.get("time_col"),"survival_time","survival_months","OS.time","time"])
    records=[]
    positive={str(x).strip().lower() for x in survival_cfg.get("event_positive_values",[1,"1","dead","deceased"])}
    for _,r in df.iterrows():
        pid=r["case_id"]
        if not pid: continue
        raw_event=r[event_col] if event_col else None
        if pd.notna(raw_event):
            event_text=str(raw_event).strip().lower()
            try:
                numeric_positive={float(x) for x in positive if x.replace(".","",1).isdigit()}
                event=int(float(raw_event) in numeric_positive)
            except (TypeError,ValueError):
                event=int(event_text in positive)
        else:
            event=int(death_col is not None and pd.notna(r[death_col]))
        if time_col and pd.notna(r[time_col]):
            time=float(r[time_col])
        else:
            col=death_col if event==1 else censor_col
            if col is None or pd.isna(r[col]): continue
            time=float(r[col])
        if not np.isfinite(time) or time<0: continue
        records.append({"case_id":pid,"time":time,"event":event,"censoring":1-event,
                        "raw_event":str(raw_event) if pd.notna(raw_event) else ""})
    out=pd.DataFrame(records)
    if out.empty: raise ValueError(f"No valid survival rows parsed from {path}")
    # Duplicate clinical rows: prefer an observed event if present; otherwise longest follow-up.
    out=out.sort_values(["case_id","event","time"],ascending=[True,False,False]).drop_duplicates("case_id")
    return out


def make_link(src, dst):
    src=Path(src).resolve(); dst=Path(dst)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve()==src: return
        if dst.is_dir() and not dst.is_symlink(): shutil.rmtree(dst)
        else: dst.unlink()
    try: dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        # If symlinks are unavailable, record source path rather than copying large feature files.
        dst.with_suffix(dst.suffix+".link.txt").write_text(str(src)+"\n",encoding="utf-8")


def run_data_prep(config_path):
    cfg=load_config(config_path); paths=cfg["paths"]; proj=cfg["project"]
    raw=Path(paths["raw_root"]); image_root=raw/paths["image_dir"]; text_root=raw/paths["text_dir"]
    expr_root=raw/paths["gene_expression_dir"]; clinical_root=raw/paths["clinical_dir"]
    gene2vec_path=Path(paths["gene2vec_file"])
    if not gene2vec_path.is_absolute(): gene2vec_path=raw/gene2vec_path
    if not gene2vec_path.exists(): raise FileNotFoundError(f"Gene2Vec not found: {gene2vec_path}")
    gene2vec=load_gene2vec(gene2vec_path,cfg["features"].get("gene2vec_dim"))
    gene_map=None
    if paths.get("gene_id_map_file"):
        mp=Path(paths["gene_id_map_file"])
        if mp.exists():
            mdf=pd.read_csv(mp); gene_map=dict(zip(mdf.iloc[:,0].astype(str),mdf.iloc[:,1].astype(str)))
    summary_modal=[]; summary_overlap=[]; summary_clin=[]; summary_genes=[]
    out_root=Path(proj["real_use_data_root"]); summary_root=Path(proj["summary_data_process"])
    for ds in proj["datasets"]:
        dsu=ds.upper(); expr_path=expr_root/f"{dsu}.csv"; clin_path=clinical_root/f"{dsu}.csv"
        img_ds=image_root/f"TCGA_{dsu}"; txt_ds=text_root/f"TCGA_{dsu}"
        img_scan=img_ds if img_ds.exists() else (image_root if any(image_root.glob("*.pt")) else img_ds)
        txt_scan=txt_ds if txt_ds.exists() else (text_root if any(text_root.glob("*.pt")) else txt_ds)
        img_files=discover_feature_files(img_scan)
        txt_files=discover_feature_files(txt_scan)
        if not expr_path.exists(): raise FileNotFoundError(f"Expression matrix not found: {expr_path}")
        if not clin_path.exists(): raise FileNotFoundError(f"Clinical file not found: {clin_path}")
        expr=load_expression(expr_path,gene_map)
        gene_overlap=[g for g in expr.index if g in gene2vec]
        if not gene_overlap:
            raise ValueError(f"No overlap between gene expression IDs and Gene2Vec symbols for {ds}; provide paths.gene_id_map_file")
        expr=expr.loc[gene_overlap]
        # Map aliquot/sample columns to patient-level IDs, average multiple samples for one patient.
        col_groups={}
        for col in expr.columns:
            pid=patient_id(col)
            if pid: col_groups.setdefault(pid,[]).append(col)
        expr_pat=pd.DataFrame({pid:expr[cols].mean(axis=1,skipna=True) for pid,cols in col_groups.items()})
        clinical=load_clinical(clin_path,cfg["survival"])
        clinical_ids=set(clinical.case_id)
        img_ids=set(img_files); txt_ids=set(txt_files); gene_ids=set(expr_pat.columns)
        overlap=img_ids & txt_ids & gene_ids & clinical_ids
        dsu_dir=out_root/dsu
        for sub in ["image_path","sent_path"]: (dsu_dir/sub/"pt_files").mkdir(parents=True,exist_ok=True)
        # Feature files are symlinked only for complete tri-modal patients.
        manifest=[]
        clin_idx=clinical.set_index("case_id")
        for pid in sorted(overlap):
            for src in img_files[pid]:
                make_link(src,dsu_dir/"image_path"/"pt_files"/src.name)
            # Multiple image files remain separate; manifest records all paths.
            text_src=txt_files[pid][0]
            make_link(text_src,dsu_dir/"sent_path"/"pt_files"/text_src.name)
            rr=clin_idx.loc[pid]
            manifest.append({"case_id":pid,"image_paths":";".join(str(dsu_dir/"image_path"/"pt_files"/p.name) for p in img_files[pid]),
                             "text_path":str(dsu_dir/"sent_path"/"pt_files"/text_src.name),
                             "time":float(rr.time),"event":int(rr.event),"censoring":int(rr.censoring)})
        # Keep expression values only for complete tri-modal patients; keep all genes that have a prior.
        complete_expr=expr_pat.reindex(columns=sorted(overlap))
        complete_expr.to_csv(dsu_dir/"gene_expression_prior_overlap.csv")
        make_link(gene2vec_path,dsu_dir/"gene2vec.txt")
        man=pd.DataFrame(manifest)
        man.to_csv(dsu_dir/"manifest.csv",index=False)
        # Metadata/statistics
        stats=[("image_feature_patients",len(img_ids)),("text_feature_patients",len(txt_ids)),
               ("gene_expression_patients",len(gene_ids)),("clinical_patients",len(clinical_ids)),
               ("all_four_way_intersection",len(overlap)),("image_only_not_complete",len(img_ids-overlap)),
               ("text_only_not_complete",len(txt_ids-overlap)),("gene_only_not_complete",len(gene_ids-overlap)),
               ("clinical_only_not_complete",len(clinical_ids-overlap)),
               ("complete_observed_events",int((man.event==1).sum()) if len(man) else 0),
               ("complete_censored",int((man.censoring==1).sum()) if len(man) else 0)]
        for metric,value in stats: summary_modal.append({"dataset":dsu,"metric":metric,"value":value})
        summary_overlap.append({"dataset":dsu,"image_patients":len(img_ids),"text_patients":len(txt_ids),"gene_patients":len(gene_ids),
                                "clinical_patients":len(clinical_ids),"complete_intersection":len(overlap),
                                "image_text_overlap":len(img_ids&txt_ids),"image_gene_overlap":len(img_ids&gene_ids),
                                "text_gene_overlap":len(txt_ids&gene_ids),"all_four_overlap":len(overlap)})
        summary_clin.append({"dataset":dsu,"complete_patients":len(man),"observed_events":int((man.event==1).sum()) if len(man) else 0,
                             "censored":int((man.censoring==1).sum()) if len(man) else 0,
                             "event_rate":float(man.event.mean()) if len(man) else np.nan})
        summary_genes.append({"dataset":dsu,"expression_gene_rows":len(expr),"gene2vec_genes":len(gene2vec),"overlap_genes":len(gene_overlap),
                              "complete_patients":len(overlap),"expression_source":str(expr_path)})
        print(f"[{dsu}] image={len(img_ids)} text={len(txt_ids)} gene={len(gene_ids)} clinical={len(clinical_ids)} complete={len(overlap)} genes_prior_overlap={len(gene_overlap)}")
    summary_root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summary_modal).to_csv(summary_root/"modality_patient_counts.csv",index=False)
    pd.DataFrame(summary_overlap).to_csv(summary_root/"modality_overlap_counts.csv",index=False)
    pd.DataFrame(summary_clin).to_csv(summary_root/"complete_cohort_censoring.csv",index=False)
    pd.DataFrame(summary_genes).to_csv(summary_root/"gene_prior_overlap.csv",index=False)
    atomic_json({"raw_root":str(raw),"image_root":str(image_root),"text_root":str(text_root),"gene2vec":str(gene2vec_path),
                 "datasets":proj["datasets"],"gene2vec_dim":len(next(iter(gene2vec.values()))),"feature_patients_image":len(img_files),
                 "feature_patients_text":len(txt_files)},summary_root/"preprocess_metadata.json")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/config.yaml"); a=p.parse_args(); run_data_prep(a.config)
