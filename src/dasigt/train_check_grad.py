from pathlib import Path
import json, math, time, warnings
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, SGD
from .config import load_config
from .dataset import prepare_fold
from .model import DASIGT
from .survival import discrete_survival_nll, concordance_index
from .utils import seed_everything, atomic_json


def _watched_parameters(model):
    """Parameters whose gradients and actual optimizer updates are diagnostically useful."""
    candidates = {
        "gmi.prior_shift.weight": model.gmi.prior_shift.weight,
        "gmi.centroids": model.gmi.centroids,
        "gmi.gene_encoder.0.weight": model.gmi.gene_encoder[0].weight,
        "gmi.gene_encoder.3.weight": model.gmi.gene_encoder[3].weight,
        "modality_gate.weight": model.modality_gate.weight,
        "modality_gate.bias": model.modality_gate.bias,
        "expert_gate.weight": model.expert_gate.weight,
        "expert_gate.bias": model.expert_gate.bias,
        "classifier.weight": model.classifier[-1].weight,
    }
    return {name: param for name, param in candidates.items() if param is not None and param.requires_grad}


def _grad_statistics(grad):
    if grad is None:
        return {"grad_norm": 0.0, "grad_finite": False, "grad_nonzero": False, "grad_status": "NONE"}
    g = grad.detach()
    finite = bool(torch.isfinite(g).all().item())
    norm = float(torch.linalg.vector_norm(g.float()).cpu()) if finite else float("nan")
    nonzero = bool(torch.count_nonzero(g).item())
    status = "OK" if finite and nonzero else ("ZERO" if finite else "NONFINITE")
    return {"grad_norm": norm, "grad_finite": finite, "grad_nonzero": nonzero, "grad_status": status}


def _activation_gradient_rows(details, epoch, step, case_id):
    """Capture gradients of soft assignments and routing logits (not optimizer parameters)."""
    rows = []
    tensors = {
        "gmi.assignment": details.get("gene", {}).get("assignment"),
        "modality_logits": details.get("modality_logits"),
        "expert_logits": details.get("expert_logits"),
    }
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        stats = _grad_statistics(tensor.grad)
        rows.append({"epoch": epoch + 1, "optimizer_step": step, "case_id": case_id,
                     "kind": "activation", "name": name, **stats,
                     "param_norm_before": float("nan"), "update_norm": float("nan"),
                     "update_ratio": float("nan")})
    return rows


def _gradient_path_audit_rows(loss, named_tensors, epoch, step, case_id, source_loss="survival"):
    """Use autograd.grad to test whether a loss is connected to selected graph nodes.

    DISCONNECTED means the tensor is not an ancestor of this loss. ZERO means a path
    exists but its local gradient is numerically zero for this sample; neither status
    alone proves a bug, so inspect several samples and the active ablation.
    """
    rows = []
    valid_names, valid_tensors = [], []
    for name, tensor in named_tensors.items():
        if tensor is None:
            rows.append({"epoch": epoch + 1, "optimizer_step": step, "case_id": case_id,
                         "source_loss": source_loss, "name": name, "grad_status": "MISSING",
                         "grad_norm": float("nan"), "requires_grad": False})
        elif not tensor.requires_grad:
            rows.append({"epoch": epoch + 1, "optimizer_step": step, "case_id": case_id,
                         "source_loss": source_loss, "name": name, "grad_status": "NO_REQUIRES_GRAD",
                         "grad_norm": float("nan"), "requires_grad": False})
        else:
            valid_names.append(name)
            valid_tensors.append(tensor)
    if valid_tensors and loss.requires_grad:
        grads = torch.autograd.grad(loss, valid_tensors, retain_graph=True, allow_unused=True)
        for name, grad in zip(valid_names, grads):
            if grad is None:
                status, norm = "DISCONNECTED", float("nan")
            else:
                g = grad.detach().float()
                finite = bool(torch.isfinite(g).all().item())
                norm = float(torch.linalg.vector_norm(g).cpu()) if finite else float("nan")
                nonzero = bool(torch.count_nonzero(g).item()) if finite else False
                status = "NONFINITE" if not finite else ("ZERO" if not nonzero else "OK")
            rows.append({"epoch": epoch + 1, "optimizer_step": step, "case_id": case_id,
                         "source_loss": source_loss, "name": name, "grad_status": status,
                         "grad_norm": norm, "requires_grad": True})
    elif valid_tensors and not loss.requires_grad:
        for name in valid_names:
            rows.append({"epoch": epoch + 1, "optimizer_step": step, "case_id": case_id,
                         "source_loss": source_loss, "name": name, "grad_status": "LOSS_NO_REQUIRES_GRAD",
                         "grad_norm": float("nan"), "requires_grad": True})
    return rows


def _device(cfg):
    dev=cfg["project"].get("device","cuda:0")
    if str(dev).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(f"Requested {dev} but CUDA is unavailable; using CPU."); return torch.device("cpu")
    return torch.device(dev)


def _evaluate(model, ds, device, train_expression, save_details=True):
    model.eval(); rows=[]; total_loss=[]; route_rows=[]
    with torch.no_grad():
        for i in range(len(ds)):
            item=ds[i]
            im=item["image"].to(device); tx=item["text"].to(device); ge=item["gene"].to(device)
            hazards,survival,risk,gene_loss,details=model(im,tx,ge,return_details=True)
            loss=discrete_survival_nll(hazards,torch.tensor([item["y"]],device=device),torch.tensor([item["event"]],device=device))
            total_loss.append(float(loss.cpu()))
            mprob=torch.softmax(details["modality_logits"],-1)[0].cpu().numpy()
            eprob=torch.softmax(details["expert_logits"],-1)[0].cpu().numpy()
            msel=int(details["modality_weights"].argmax(-1)[0].cpu()); esel=int(details["expert_weights"].argmax(-1)[0].cpu())
            row={"case_id":item["case_id"],"time":item["time"],"event":item["event"],"censoring":item["censoring"],"y":item["y"],
                 "risk":float(risk[0].cpu()),"loss":float(loss.cpu()),"gene_loss":float(gene_loss.cpu()),
                 "selected_route":msel,"selected_expert":esel}
            for j,v in enumerate(hazards[0].cpu().numpy()): row[f"hazard_{j}"]=float(v)
            for j,v in enumerate(survival[0].cpu().numpy()): row[f"survival_{j}"]=float(v)
            for j,v in enumerate(mprob): row[f"modality_prob_{j}"]=float(v)
            for j,v in enumerate(eprob): row[f"expert_prob_{j}"]=float(v)
            for j,v in enumerate(details["gene"]["module_scores"].cpu().numpy()): row[f"module_score_{j}"]=float(v)
            rows.append(row)
            if save_details:
                route_rows.append({"case_id":item["case_id"],"modality_logits":details["modality_logits"][0].cpu().tolist(),
                                   "modality_weights":details["modality_weights"][0].cpu().tolist(),
                                   "expert_logits":details["expert_logits"][0].cpu().tolist(),
                                   "expert_weights":details["expert_weights"][0].cpu().tolist()})
    pred=pd.DataFrame(rows)
    ci=concordance_index(pred.time,pred.event,pred.risk) if len(pred) else float("nan")
    return {"loss":float(np.mean(total_loss)) if total_loss else float("nan"),"cindex":ci,"predictions":pred,"routes":route_rows}


def run_fold(cfg, dataset_name, fold, ablation=None):
    seed_everything(int(cfg["project"].get("seed",7))+fold)
    dsname=dataset_name.upper(); output=Path(cfg["project"]["output_root"])/dsname/f"fold_{fold}"/(ablation or cfg["model"].get("ablation","none"))
    output.mkdir(parents=True,exist_ok=True)
    split_root=Path(cfg["paths"]["split_root"])
    split_pattern=cfg["paths"].get("split_pattern","{dataset}/splits_{fold}.csv")
    split_rel=split_pattern.format(dataset=dsname,dataset_lower=dsname.lower(),fold=fold)
    split_path=split_root/split_rel
    if not split_path.exists():
        # Also accept original EXMP convention split_root/5foldcv_xxx/tcga_dataset/splits_n.csv
        alt=split_root/f"tcga_{dsname.lower()}"/f"splits_{fold}.csv"
        if alt.exists(): split_path=alt
    if not split_path.exists(): raise FileNotFoundError(f"Split file not found: {split_path}")
    real_root=Path(cfg["project"]["real_use_data_root"])
    fold_data=prepare_fold(real_root,split_path,cfg,dsname,fold,output)
    device=_device(cfg)
    prior=fold_data["prior"].to(device); train_expr=fold_data["train_expression"].to(device)
    model=DASIGT(prior,train_expr,cfg,ablation=ablation).to(device)
    opt_name=cfg["training"].get("optimizer","adamw").lower()
    optimizer=AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"]) if opt_name=="adamw" else SGD(model.parameters(),lr=cfg["training"]["learning_rate"],momentum=0.9,weight_decay=cfg["training"]["weight_decay"])
    use_amp=bool(cfg["training"].get("use_amp",False) and device.type=="cuda")
    # PyTorch 2.2.1-compatible GradScaler API.
    scaler=torch.cuda.amp.GradScaler(enabled=use_amp)
    debug_grad=bool(cfg["training"].get("debug_gradient_tracking",False))
    debug_path=bool(cfg["training"].get("debug_gradient_path_check",False))
    debug_every=max(1,int(cfg["training"].get("debug_gradient_every_n_steps",1)))
    watched=_watched_parameters(model) if debug_grad else {}
    grad_path=output/"gradient_diagnostics.csv"
    path_audit_path=output/"gradient_path_audit.csv"
    if debug_grad and grad_path.exists():
        grad_path.unlink()
    if debug_path and path_audit_path.exists():
        path_audit_path.unlink()
    global_step=0
    best=-float("inf"); best_epoch=-1; patience=0; history=[]; start=time.time()
    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train(); losses=[]; epoch_grad_rows=[]; epoch_path_audit_rows=[]; optimizer.zero_grad(set_to_none=True)
        for idx in range(len(fold_data["train"])):
            item=fold_data["train"][idx]
            im=item["image"].to(device); tx=item["text"].to(device); ge=item["gene"].to(device)
            with torch.autocast(device_type=device.type,enabled=use_amp):
                hazards,survival,risk,gene_loss,details=model(im,tx,ge,return_details=True)
                surv_loss=discrete_survival_nll(hazards,torch.tensor([item["y"]],device=device),torch.tensor([item["event"]],device=device))
                loss=surv_loss+float(cfg["model"].get("lambda_gene",10.0))*gene_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, patient={item['case_id']}")
            accum=max(1,int(cfg["training"].get("gradient_accumulation_steps",1)))
            step_now=((idx+1)%accum==0 or idx+1==len(fold_data["train"]))
            audit_step=global_step+1
            should_audit_path=debug_path and step_now and (audit_step % debug_every == 0)
            if should_audit_path:
                # Audit the survival loss separately from gene_loss. This is essential:
                # gene_loss can give GMI parameters gradients even if the survival path is broken.
                gene_details=details.get("gene", {})
                path_tensors={
                    "classifier_logits": details.get("logits"),
                    "fused_representation": details.get("fused"),
                    "modality_logits": details.get("modality_logits"),
                    "expert_logits": details.get("expert_logits"),
                    "gmi.assignment": gene_details.get("assignment"),
                    "gmi.pi_first_u": gene_details.get("pi_first_u"),
                    "gmi.pi_first_v": gene_details.get("pi_first_v"),
                    "gmi.pi_last_u": gene_details.get("pi_last_u"),
                    "gmi.pi_last_v": gene_details.get("pi_last_v"),
                    "gmi.prior_shift.weight": model.gmi.prior_shift.weight,
                    "gmi.centroids": model.gmi.centroids,
                    "gmi.gene_encoder.0.weight": model.gmi.gene_encoder[0].weight,
                }
                epoch_path_audit_rows.extend(_gradient_path_audit_rows(
                    surv_loss, path_tensors, epoch, audit_step, item["case_id"], source_loss="survival"))
                # The gene regularizer should reach the assignment/centroids/prior path,
                # but it is not expected to depend on the Power Iteration intermediates.
                gene_loss_tensors={
                    "gmi.assignment": gene_details.get("assignment"),
                    "gmi.prior_shift.weight": model.gmi.prior_shift.weight,
                    "gmi.centroids": model.gmi.centroids,
                }
                epoch_path_audit_rows.extend(_gradient_path_audit_rows(
                    gene_loss, gene_loss_tensors, epoch, audit_step, item["case_id"], source_loss="gene"))
            scaler.scale(loss / accum).backward()
            if step_now:
                global_step += 1
                should_log=debug_grad and (global_step % debug_every == 0)
                clip_norm=cfg["training"].get("grad_clip_norm")
                # Unscale once before inspecting gradients / clipping under AMP.
                if use_amp and (should_log or clip_norm):
                    scaler.unscale_(optimizer)

                before={}
                if should_log:
                    for name,param in watched.items():
                        grad_stats=_grad_statistics(param.grad)
                        before[name]=param.detach().clone()
                        epoch_grad_rows.append({"epoch":epoch+1,"optimizer_step":global_step,
                            "case_id":item["case_id"],"kind":"parameter","name":name,**grad_stats,
                            "param_norm_before":float(torch.linalg.vector_norm(param.detach().float()).cpu()),
                            "update_norm":float("nan"),"update_ratio":float("nan")})
                    epoch_grad_rows.extend(_activation_gradient_rows(details,epoch,global_step,item["case_id"]))

                if clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),float(clip_norm))
                scaler.step(optimizer)
                scaler.update()

                if should_log:
                    # Measure the actual parameter change caused by optimizer.step(),
                    # not just the existence of a gradient.
                    row_idx={(r["kind"],r["name"]):r for r in epoch_grad_rows
                             if r["optimizer_step"]==global_step and r["kind"]=="parameter"}
                    for name,param in watched.items():
                        old_param=before[name]
                        delta=float(torch.linalg.vector_norm((param.detach()-old_param).float()).cpu())
                        old_norm=float(torch.linalg.vector_norm(old_param.float()).cpu())
                        r=row_idx[("parameter",name)]
                        r["update_norm"]=delta
                        r["update_ratio"]=delta/(old_norm+1e-12)
                    concise=[]
                    for name in ("gmi.prior_shift.weight","gmi.centroids","modality_gate.weight","expert_gate.weight"):
                        r=row_idx.get(("parameter",name))
                        if r is not None:
                            concise.append(f"{name}: grad={r['grad_norm']:.3e}, update={r['update_norm']:.3e}")
                    print(f"[gradient-debug] {dsname} fold={fold} epoch={epoch+1} step={global_step} | " + " | ".join(concise))
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        val=_evaluate(model,fold_data["val"],device,train_expr)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"val_loss":val["loss"],"val_cindex":val["cindex"]}
        history.append(row); print(f"[{dsname} fold={fold} ablation={ablation or 'none'}] epoch {epoch+1}/{cfg['training']['max_epochs']} train={row['train_loss']:.4f} val_c={row['val_cindex']:.4f}")
        if debug_grad and epoch_grad_rows:
            pd.DataFrame(epoch_grad_rows).to_csv(grad_path,mode="a",header=not grad_path.exists(),index=False)
        if debug_path and epoch_path_audit_rows:
            pd.DataFrame(epoch_path_audit_rows).to_csv(path_audit_path,mode="a",header=not path_audit_path.exists(),index=False)
        score=val["cindex"]
        if (not (output/"best.pt").exists()) or (np.isfinite(score) and score>best):
            if np.isfinite(score): best=score
            best_epoch=epoch; patience=0
            torch.save({"model_state":model.state_dict(),"config":cfg,"dataset":dsname,"fold":fold,"ablation":ablation or "none",
                        "genes":fold_data["genes"],"fold_metadata":fold_data["metadata"]},output/"best.pt")
        else:
            patience+=1
            if patience>=int(cfg["training"]["patience"]): break
    pd.DataFrame(history).to_csv(output/"training_history.csv",index=False)
    checkpoint=torch.load(output/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val=_evaluate(model,fold_data["val"],device,train_expr)
    test=_evaluate(model,fold_data["test"],device,train_expr)
    if cfg["training"].get("save_predictions",True):
        val["predictions"].to_csv(output/"val_predictions.csv",index=False); test["predictions"].to_csv(output/"test_predictions.csv",index=False)
        pd.DataFrame(val["routes"]).to_json(output/"val_routing.jsonl",orient="records",lines=True)
        pd.DataFrame(test["routes"]).to_json(output/"test_routing.jsonl",orient="records",lines=True)
        # Export gene-level soft module assignments and learned eigengene loadings.
        model.eval()
        with torch.no_grad():
            sample=fold_data["train"][0]["gene"].to(device)
            _,_,gd=model.gmi(sample,return_details=True)
        assign=gd["assignment"].cpu().numpy(); loadings=gd["loadings"].cpu().numpy()
        pd.DataFrame(assign,index=fold_data["genes"],columns=[f"module_{i}" for i in range(assign.shape[1])]).to_csv(output/"gene_module_assignments.csv")
        pd.DataFrame(loadings,index=fold_data["genes"],columns=[f"module_{i}" for i in range(loadings.shape[1])]).to_csv(output/"gene_module_loadings.csv")
    summary={"dataset":dsname,"fold":fold,"ablation":ablation or "none","best_epoch":best_epoch,
             "best_val_cindex":float(best) if np.isfinite(best) else None,
             "val_cindex":float(val["cindex"]) if np.isfinite(val["cindex"]) else None,
             "test_cindex":float(test["cindex"]) if np.isfinite(test["cindex"]) else None,
             "val_loss":float(val["loss"]) if np.isfinite(val["loss"]) else None,
             "test_loss":float(test["loss"]) if np.isfinite(test["loss"]) else None,
             "n_train":len(fold_data["train"]),"n_val":len(fold_data["val"]),"n_test":len(fold_data["test"]),
             "val_test_overlap_n":len(fold_data["metadata"]["val_test_overlap"]),"runtime_seconds":time.time()-start}
    atomic_json(summary,output/"fold_result.json")
    return summary


def run_training(config_path, datasets=None, folds=None, ablation=None):
    cfg=load_config(config_path); datasets=datasets or cfg["project"]["datasets"]; folds=folds or cfg["training"]["folds"]
    rows=[]
    for ds in datasets:
        for fold in folds: rows.append(run_fold(cfg,ds,int(fold),ablation=ablation))
    out=Path(cfg["project"]["output_root"])/"fold_results.csv"; out.parent.mkdir(parents=True,exist_ok=True)
    current=pd.DataFrame(rows)
    if out.exists():
        old=pd.read_csv(out)
        merged=pd.concat([old,current],ignore_index=True)
        keys=[k for k in ("dataset","fold","ablation") if k in merged.columns]
        if keys: merged=merged.drop_duplicates(keys,keep="last")
        current=merged
    current.to_csv(out,index=False)
    return current

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--datasets",nargs="*"); p.add_argument("--folds",nargs="*",type=int); p.add_argument("--ablation",default=None); a=p.parse_args()
    run_training(a.config,a.datasets,a.folds,a.ablation)
