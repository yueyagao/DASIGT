from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .config import load_config
from .toppfun import ToppFunRunner

ROUTES=["P","P+G","P+R","P+G+R"]
EXPERTS=["EX1","EX2","EX3","EX4"]


def entropy(prob):
    p=np.asarray(prob,dtype=float); p=p[p>0]
    return float(-(p*np.log(p)).sum()) if len(p) else 0.0


def _read_fold_result(folder):
    p=Path(folder)/"fold_result.json"
    return json.loads(p.read_text()) if p.exists() else None


def run_analysis(config_path):
    cfg=load_config(config_path); out_root=Path(cfg["project"]["output_root"]); ana=Path(cfg["project"]["summary_analysis"])
    ana.mkdir(parents=True,exist_ok=True)
    route_rows=[]; patient_rows=[]; gene_rows=[]; module_stability_rows=[]; metric_rows=[]
    module_reference={}; module_reference_top={}
    toppfun_cfg = cfg.get("analysis", {}).get("toppfun", {}) or {}
    toppfun_runner = None
    if bool(toppfun_cfg.get("enabled", False)) and str(cfg.get("analysis", {}).get("enrichment_tool", "")).lower() == "toppfun_api":
        toppfun_runner = ToppFunRunner(toppfun_cfg, ana / "toppfun")
        print("ToppFun API enrichment is enabled. Gene lists will be sent to the remote ToppGene service; cached results will be reused.")
    for ds in cfg["project"]["datasets"]:
        ds=ds.upper()
        for fold in cfg["training"]["folds"]:
            base=out_root/ds/f"fold_{fold}"
            # Collect the default model and every ablation result separately.
            if not base.exists(): continue
            for variant_dir in base.iterdir():
                if not variant_dir.is_dir(): continue
                variant=variant_dir.name
                result=_read_fold_result(variant_dir)
                if result: metric_rows.append(result)
                for split in ("val","test"):
                    pred_path=variant_dir/f"{split}_predictions.csv"
                    if not pred_path.exists(): continue
                    pred=pd.read_csv(pred_path)
                    if pred.empty: continue
                    for route_id,label in enumerate(ROUTES):
                        col=f"modality_prob_{route_id}"
                        if col not in pred: continue
                        sel=(pred.selected_route==route_id) if "selected_route" in pred else np.zeros(len(pred),dtype=bool)
                        route_rows.append({"dataset":ds,"fold":fold,"ablation":variant,"split":split,"kind":"modality","route":label,
                                           "n_patients":int(sel.sum()),"frequency":float(sel.mean()),"mean_route_probability":float(pred[col].mean()),
                                           "mean_divergence_placeholder":np.nan})
                    for exp_id,label in enumerate(EXPERTS):
                        col=f"expert_prob_{exp_id}"
                        if col not in pred: continue
                        sel=(pred.selected_expert==exp_id) if "selected_expert" in pred else np.zeros(len(pred),dtype=bool)
                        route_rows.append({"dataset":ds,"fold":fold,"ablation":variant,"split":split,"kind":"expert","route":label,
                                           "n_patients":int(sel.sum()),"frequency":float(sel.mean()),"mean_route_probability":float(pred[col].mean()),
                                           "mean_divergence_placeholder":np.nan})
                    for _,r in pred.iterrows():
                        d={"dataset":ds,"fold":fold,"ablation":variant,"split":split,"case_id":r.case_id,
                           "time":r.time,"event":r.event,"censoring":r.censoring,"risk":r.risk,
                           "selected_route":ROUTES[int(r.selected_route)],"selected_expert":EXPERTS[int(r.selected_expert)],"loss":r.loss}
                        d.update({c:float(r[c]) for c in pred.columns if c.startswith("module_score_")})
                        patient_rows.append(d)
                gene_path=variant_dir/"gene_module_assignments.csv"
                if gene_path.exists():
                    assign=pd.read_csv(gene_path,index_col=0)
                    vals=assign.to_numpy(); key=(ds,variant); stability=np.nan
                    if key in module_reference:
                        ref_genes,ref_vals=module_reference[key]
                        shared=sorted(set(ref_genes)&set(assign.index))
                        if shared:
                            ref_lookup={g:i for i,g in enumerate(ref_genes)}; cur_lookup={g:i for i,g in enumerate(assign.index)}
                            aa=np.stack([ref_vals[ref_lookup[g]] for g in shared]); bb=np.stack([vals[cur_lookup[g]] for g in shared])
                            aa=aa/(np.linalg.norm(aa,axis=0,keepdims=True)+1e-8); bb=bb/(np.linalg.norm(bb,axis=0,keepdims=True)+1e-8)
                            sim=aa.T@bb; rr,cc=linear_sum_assignment(-sim)
                            order=np.zeros(vals.shape[1],dtype=int)
                            for r0,c0 in zip(rr,cc): order[r0]=c0
                            vals=vals[:,order]; assign=assign.iloc[:,order]
                            ref_top=module_reference_top[key]; cur_top=[]; jacc=[]
                            topn=int(cfg["analysis"].get("pathway_top_genes",50))
                            for j in range(vals.shape[1]):
                                ix=np.argsort(vals[:,j])[-topn:][::-1]; genes_now=set(assign.index[ix]); genes_ref=ref_top[j]
                                jacc.append(len(genes_now&genes_ref)/max(1,len(genes_now|genes_ref))); cur_top.append(genes_now)
                            stability=float(np.mean(jacc))
                    else:
                        topn=int(cfg["analysis"].get("pathway_top_genes",50)); tops=[]
                        for j in range(vals.shape[1]): tops.append(set(assign.index[np.argsort(vals[:,j])[-topn:]]))
                        module_reference[key]=(list(assign.index),vals.copy()); module_reference_top[key]=tops
                    for j,col in enumerate(assign.columns):
                        top_idx=np.argsort(vals[:,j])[-int(cfg["analysis"].get("pathway_top_genes",50)):][::-1]
                        for rank,ix in enumerate(top_idx,1):
                            gene_rows.append({"dataset":ds,"fold":fold,"ablation":variant,"module":f"module_aligned_{j}","gene":assign.index[ix],"membership":float(vals[ix,j]),"rank":rank})
                    module_stability_rows.append({"dataset":ds,"fold":fold,"ablation":variant,"n_genes":assign.shape[0],"n_modules":assign.shape[1],
                                                  "mean_membership_entropy":float(np.mean([-np.sum(p[p>0]*np.log(p[p>0])) for p in vals])),
                                                  "mean_top_gene_jaccard_to_reference_fold":stability})
                    if toppfun_runner is not None:
                        toppfun_assignment = assign.copy()
                        toppfun_assignment.columns = [f"module_aligned_{j}" for j in range(toppfun_assignment.shape[1])]
                        toppfun_runner.run_assignment_matrix(
                            assignment=toppfun_assignment,
                            dataset=ds,
                            fold=int(fold),
                            ablation=variant,
                            top_genes=int(toppfun_cfg.get("top_genes_per_module", cfg.get("analysis", {}).get("pathway_top_genes", 50))),
                        )
    if metric_rows:
        met=pd.DataFrame(metric_rows); met.to_csv(ana/"fold_metrics_all_variants.csv",index=False)
        agg=met.groupby(["dataset","ablation"],dropna=False).agg(folds=("fold","nunique"),val_cindex_mean=("val_cindex","mean"),val_cindex_std=("val_cindex","std"),
              test_cindex_mean=("test_cindex","mean"),test_cindex_std=("test_cindex","std"),test_loss_mean=("test_loss","mean"),runtime_seconds_mean=("runtime_seconds","mean")).reset_index()
        agg.to_csv(ana/"five_fold_performance_summary.csv",index=False)
    if route_rows:
        rr=pd.DataFrame(route_rows); rr.to_csv(ana/"routing_per_fold.csv",index=False)
        pooled=rr.groupby(["dataset","ablation","split","kind","route"],dropna=False).agg(n_patients=("n_patients","sum"),mean_route_probability=("mean_route_probability","mean")).reset_index()
        total=pooled.groupby(["dataset","ablation","split","kind"])["n_patients"].transform("sum")
        pooled["frequency"]=pooled.n_patients/total.replace(0,np.nan)
        pooled["entropy_group"] = pooled.groupby(["dataset","ablation","split","kind"])["frequency"].transform(lambda x: entropy(x.to_numpy()))
        pooled.to_csv(ana/"routing_five_fold_aggregate.csv",index=False)
    if patient_rows:
        patients_df=pd.DataFrame(patient_rows)
        score_cols=[c for c in patients_df.columns if c.startswith("module_score_")]
        if score_cols:
            scores=patients_df[score_cols].to_numpy(dtype=float)
            best_idx=np.argmax(np.abs(scores),axis=1)
            patients_df["selected_module_by_abs_score"]=[f"module_aligned_{i}" for i in best_idx]
            patients_df["selected_module_score"]=[float(scores[r,i]) for r,i in enumerate(best_idx)]
            ms=(patients_df.groupby(["dataset","fold","ablation","split","selected_module_by_abs_score"],dropna=False)
                .size().rename("n_patients").reset_index())
            denom=ms.groupby(["dataset","fold","ablation","split"])["n_patients"].transform("sum")
            ms["frequency"]=ms.n_patients/denom.replace(0,np.nan)
            ms.to_csv(ana/"module_selection_per_fold.csv",index=False)
            pooled=(ms.groupby(["dataset","ablation","split","selected_module_by_abs_score"],dropna=False)["n_patients"].sum().reset_index())
            denom=pooled.groupby(["dataset","ablation","split"])["n_patients"].transform("sum")
            pooled["frequency"]=pooled.n_patients/denom.replace(0,np.nan)
            pooled.to_csv(ana/"module_selection_five_fold_aggregate.csv",index=False)
        patients_df.to_csv(ana/"patient_level_analysis.csv",index=False)
    if gene_rows:
        genes=pd.DataFrame(gene_rows); genes.to_csv(ana/"gene_module_top_genes_per_fold.csv",index=False)
        freq=genes.groupby(["dataset","ablation","module","gene"],dropna=False).agg(fold_count=("fold","nunique"),mean_membership=("membership","mean"),best_rank=("rank","min")).reset_index()
        freq.to_csv(ana/"gene_module_top_gene_frequency_five_fold.csv",index=False)
    if module_stability_rows: pd.DataFrame(module_stability_rows).to_csv(ana/"gene_module_statistics_per_fold.csv",index=False)
    toppfun_outputs = {}
    if toppfun_runner is not None:
        toppfun_outputs = toppfun_runner.write_outputs()
        print(f"ToppFun enrichment outputs written to {ana / 'toppfun'}")
    # Explicit analysis index explains which results are patient-level, fold-level, and pooled.
    index={"fold_metrics":"fold_metrics_all_variants.csv","five_fold_performance":"five_fold_performance_summary.csv",
           "routing_per_fold":"routing_per_fold.csv","routing_pooled":"routing_five_fold_aggregate.csv",
           "patient_level":"patient_level_analysis.csv","module_selection_per_fold":"module_selection_per_fold.csv",
           "module_selection_pooled":"module_selection_five_fold_aggregate.csv","module_top_genes":"gene_module_top_genes_per_fold.csv",
           "top_gene_fold_frequency":"gene_module_top_gene_frequency_five_fold.csv","module_stats":"gene_module_statistics_per_fold.csv",
           "toppfun": toppfun_outputs if toppfun_outputs else None,
           "note":"When analysis.toppfun.enabled=true and enrichment_tool=toppfun_api, the analysis also runs/caches ToppFun enrichment for configured GMI modules. Otherwise outputs report gene-module membership and top genes only."}
    (ana/"analysis_index.json").write_text(json.dumps(index,indent=2),encoding="utf-8")
    print(f"Analysis outputs written to {ana}")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/config.yaml"); a=p.parse_args(); run_analysis(a.config)
