"""Training entry point for DASIGT with true multi-patient mini-batches."""
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import prepare_fold, collate_patients
from .model import DASIGT
from .survival import discrete_survival_nll, discrete_survival_nll_per_sample, concordance_index
from .utils import seed_everything, atomic_json


def _watched_parameters(model):
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
    return {name: p for name, p in candidates.items() if p is not None and p.requires_grad}


def _grad_statistics(grad):
    if grad is None:
        return {"grad_norm": 0.0, "grad_finite": False, "grad_nonzero": False, "grad_status": "NONE"}
    g = grad.detach()
    finite = bool(torch.isfinite(g).all().item())
    norm = float(torch.linalg.vector_norm(g.float()).cpu()) if finite else float("nan")
    nonzero = bool(torch.count_nonzero(g).item()) if finite else False
    status = "OK" if finite and nonzero else ("ZERO" if finite else "NONFINITE")
    return {"grad_norm": norm, "grad_finite": finite, "grad_nonzero": nonzero, "grad_status": status}


def _activation_gradient_rows(details, epoch, step, case_id):
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
    elif valid_tensors:
        for name in valid_names:
            rows.append({"epoch": epoch + 1, "optimizer_step": step, "case_id": case_id,
                         "source_loss": source_loss, "name": name, "grad_status": "LOSS_NO_REQUIRES_GRAD",
                         "grad_norm": float("nan"), "requires_grad": True})
    return rows


def _device(cfg):
    dev = cfg["project"].get("device", "cuda:0")
    if str(dev).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(f"Requested {dev} but CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(dev)


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_loader(dataset, cfg, device, shuffle=False):
    batch_size = max(1, int(cfg["training"].get("batch_size", 1)))
    num_workers = max(0, int(cfg["project"].get("num_workers", 0)))
    pin_memory = bool(cfg["training"].get("pin_memory", True) and device.type == "cuda")
    generator = torch.Generator()
    generator.manual_seed(int(cfg["project"].get("seed", 7)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_patients,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        generator=generator if shuffle else None,
    )


def _move_batch(batch, device):
    return {
        "case_id": batch["case_id"],
        "image": batch["image"].to(device, non_blocking=True),
        "image_mask": batch["image_mask"].to(device, non_blocking=True),
        "text": batch["text"].to(device, non_blocking=True),
        "text_mask": batch["text_mask"].to(device, non_blocking=True),
        "gene": batch["gene"].to(device, non_blocking=True),
        "time": batch["time"].to(device, non_blocking=True),
        "event": batch["event"].to(device, non_blocking=True),
        "censoring": batch["censoring"].to(device, non_blocking=True),
        "y": batch["y"].to(device, non_blocking=True),
    }


def _evaluate(model, ds, device, train_expression, cfg, save_details=True):
    """Batched evaluation. inference_seconds includes data loading and metric/prediction assembly."""
    model.eval()
    rows, total_loss_sum, total_n, route_rows = [], 0.0, 0, []
    loader = _make_loader(ds, cfg, device, shuffle=False)
    _sync_device(device)
    start = time.perf_counter()
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            hazards, survival, risk, gene_loss, details = model(
                batch["image"], batch["text"], batch["gene"],
                image_mask=batch["image_mask"], text_mask=batch["text_mask"], return_details=True
            )
            batch_n = len(batch["case_id"])
            batch_surv_loss = discrete_survival_nll(hazards, batch["y"], batch["event"])
            total_loss_sum += float(batch_surv_loss.cpu()) * batch_n
            total_n += batch_n

            mprob = torch.softmax(details["modality_logits"], -1).cpu().numpy()
            eprob = torch.softmax(details["expert_logits"], -1).cpu().numpy()
            msel = details["modality_weights"].argmax(-1).cpu().numpy()
            esel = details["expert_weights"].argmax(-1).cpu().numpy()
            hazards_np = hazards.cpu().numpy()
            survival_np = survival.cpu().numpy()
            risk_np = risk.cpu().numpy()
            gene_scores_np = details["gene"]["module_scores"].cpu().numpy()
            times_np = batch["time"].cpu().numpy()
            events_np = batch["event"].cpu().numpy().astype(int)
            censoring_np = batch["censoring"].cpu().numpy()
            y_np = batch["y"].cpu().numpy()
            gene_loss_value = float(gene_loss.cpu())

            individual_losses = discrete_survival_nll_per_sample(hazards, batch["y"], batch["event"]).cpu().numpy()
            individual_losses = np.asarray(individual_losses, dtype=float)
            for i, pid in enumerate(batch["case_id"]):
                row = {
                    "case_id": pid, "time": float(times_np[i]), "event": int(events_np[i]),
                    "censoring": int(censoring_np[i]), "y": int(y_np[i]),
                    "risk": float(risk_np[i]), "loss": float(individual_losses[i]),
                    "gene_loss": gene_loss_value, "selected_route": int(msel[i]),
                    "selected_expert": int(esel[i]),
                }
                for j, value in enumerate(hazards_np[i]): row[f"hazard_{j}"] = float(value)
                for j, value in enumerate(survival_np[i]): row[f"survival_{j}"] = float(value)
                for j, value in enumerate(mprob[i]): row[f"modality_prob_{j}"] = float(value)
                for j, value in enumerate(eprob[i]): row[f"expert_prob_{j}"] = float(value)
                for j, value in enumerate(gene_scores_np[i]): row[f"module_score_{j}"] = float(value)
                rows.append(row)
                if save_details:
                    route_rows.append({
                        "case_id": pid,
                        "modality_logits": details["modality_logits"][i].cpu().tolist(),
                        "modality_weights": details["modality_weights"][i].cpu().tolist(),
                        "expert_logits": details["expert_logits"][i].cpu().tolist(),
                        "expert_weights": details["expert_weights"][i].cpu().tolist(),
                    })
    _sync_device(device)
    inference_seconds = time.perf_counter() - start
    pred = pd.DataFrame(rows)
    ci = concordance_index(pred.time, pred.event, pred.risk) if len(pred) else float("nan")
    return {
        "loss": total_loss_sum / max(total_n, 1), "cindex": ci,
        "predictions": pred, "routes": route_rows,
        "inference_seconds": inference_seconds, "n_patients": total_n,
    }


def run_fold(cfg, dataset_name, fold, ablation=None):
    seed = int(cfg["project"].get("seed", 7)) + int(fold)
    seed_everything(seed)
    dsname = dataset_name.upper()
    batch_size = max(1, int(cfg["training"].get("batch_size", 1)))
    accumulation_steps = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    effective_batch_size = batch_size * accumulation_steps
    output = Path(cfg["project"]["output_root"]) / dsname / f"fold_{fold}" / (ablation or cfg["model"].get("ablation", "none"))
    output.mkdir(parents=True, exist_ok=True)

    split_root = Path(cfg["paths"]["split_root"])
    split_pattern = cfg["paths"].get("split_pattern", "{dataset}/splits_{fold}.csv")
    split_rel = split_pattern.format(dataset=dsname, dataset_lower=dsname.lower(), fold=fold)
    split_path = split_root / split_rel
    if not split_path.exists():
        alt = split_root / f"tcga_{dsname.lower()}" / f"splits_{fold}.csv"
        if alt.exists():
            split_path = alt
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    fold_data = prepare_fold(cfg["project"]["real_use_data_root"], split_path, cfg, dsname, fold, output)
    device = _device(cfg)
    prior = fold_data["prior"].to(device)
    train_expr = fold_data["train_expression"].to(device)
    model = DASIGT(prior, train_expr, cfg, ablation=ablation).to(device)

    opt_name = cfg["training"].get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    else:
        optimizer = SGD(model.parameters(), lr=cfg["training"]["learning_rate"], momentum=0.9, weight_decay=cfg["training"]["weight_decay"])
    use_amp = bool(cfg["training"].get("use_amp", False) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # compatible with PyTorch 2.2.1
    debug_grad = bool(cfg["training"].get("debug_gradient_tracking", False))
    debug_path = bool(cfg["training"].get("debug_gradient_path_check", False))
    debug_every = max(1, int(cfg["training"].get("debug_gradient_every_n_steps", 1)))
    watched = _watched_parameters(model) if debug_grad else {}
    grad_path = output / "gradient_diagnostics.csv"
    path_audit_path = output / "gradient_path_audit.csv"
    if debug_grad and grad_path.exists(): grad_path.unlink()
    if debug_path and path_audit_path.exists(): path_audit_path.unlink()

    train_loader = _make_loader(fold_data["train"], cfg, device, shuffle=True)
    best = -float("inf")
    best_epoch = -1
    patience = 0
    history = []
    total_start = time.perf_counter()
    global_step = 0
    lambda_gene = float(cfg["model"].get("lambda_gene", 10.0))
    early_stopping = bool(cfg["training"].get("early_stopping", True))

    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        epoch_grad_rows = []
        epoch_path_audit_rows = []
        optimizer.zero_grad(set_to_none=True)
        accumulated_samples = 0
        n_batches = len(train_loader)
        _sync_device(device)
        train_start = time.perf_counter()

        for batch_idx, raw_batch in enumerate(train_loader):
            batch = _move_batch(raw_batch, device)
            batch_n = len(batch["case_id"])
            case_label = "|".join(batch["case_id"])
            with torch.autocast(device_type=device.type, enabled=use_amp):
                hazards, survival, risk, gene_loss, details = model(
                    batch["image"], batch["text"], batch["gene"],
                    image_mask=batch["image_mask"], text_mask=batch["text_mask"], return_details=True
                )
                surv_loss = discrete_survival_nll(hazards, batch["y"], batch["event"])
                loss = surv_loss + lambda_gene * gene_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={case_label}")

            step_now = ((batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == n_batches)
            next_step = global_step + 1
            should_audit_path = debug_path and step_now and (next_step % debug_every == 0)
            if should_audit_path:
                # Run graph-connectivity probes before backward frees graph buffers.
                gene_details = details.get("gene", {})
                path_tensors = {
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
                    surv_loss, path_tensors, epoch, next_step, case_label, source_loss="survival"
                ))
                gene_loss_tensors = {
                    "gmi.assignment": gene_details.get("assignment"),
                    "gmi.prior_shift.weight": model.gmi.prior_shift.weight,
                    "gmi.centroids": model.gmi.centroids,
                }
                epoch_path_audit_rows.extend(_gradient_path_audit_rows(
                    gene_loss, gene_loss_tensors, epoch, next_step, case_label, source_loss="gene"
                ))

            # Backpropagate a batch sum; divide accumulated gradients by the exact
            # number of patients in this optimizer step (including a short last batch).
            scaler.scale(loss * batch_n).backward()
            accumulated_samples += batch_n
            epoch_loss_sum += float(loss.detach().cpu()) * batch_n
            epoch_samples += batch_n

            if step_now:
                if use_amp:
                    scaler.unscale_(optimizer)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.div_(float(max(accumulated_samples, 1)))

                global_step += 1
                should_log = debug_grad and (global_step % debug_every == 0)
                clip_norm = cfg["training"].get("grad_clip_norm")
                before = {}
                if should_log:
                    for name, param in watched.items():
                        grad_stats = _grad_statistics(param.grad)
                        before[name] = param.detach().clone()
                        epoch_grad_rows.append({
                            "epoch": epoch + 1, "optimizer_step": global_step, "case_id": case_label,
                            "kind": "parameter", "name": name, **grad_stats,
                            "param_norm_before": float(torch.linalg.vector_norm(param.detach().float()).cpu()),
                            "update_norm": float("nan"), "update_ratio": float("nan"),
                        })
                    epoch_grad_rows.extend(_activation_gradient_rows(details, epoch, global_step, case_label))

                if clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
                scaler.step(optimizer)
                scaler.update()

                if should_log:
                    row_idx = {(r["kind"], r["name"]): r for r in epoch_grad_rows
                               if r["optimizer_step"] == global_step and r["kind"] == "parameter"}
                    for name, param in watched.items():
                        old_param = before[name]
                        delta = float(torch.linalg.vector_norm((param.detach() - old_param).float()).cpu())
                        old_norm = float(torch.linalg.vector_norm(old_param.float()).cpu())
                        r = row_idx[("parameter", name)]
                        r["update_norm"] = delta
                        r["update_ratio"] = delta / (old_norm + 1e-12)
                    concise = []
                    for name in ("gmi.prior_shift.weight", "gmi.centroids", "modality_gate.weight", "expert_gate.weight"):
                        r = row_idx.get(("parameter", name))
                        if r is not None:
                            concise.append(f"{name}: grad={r['grad_norm']:.3e}, update={r['update_norm']:.3e}")
                    print(f"[gradient-debug] {dsname} fold={fold} epoch={epoch+1} step={global_step} | " + " | ".join(concise))

                optimizer.zero_grad(set_to_none=True)
                accumulated_samples = 0

        _sync_device(device)
        train_seconds = time.perf_counter() - train_start
        train_loss = epoch_loss_sum / max(epoch_samples, 1)
        val = _evaluate(model, fold_data["val"], device, train_expr, cfg)
        row = {
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val["loss"],
            "val_cindex": val["cindex"], "train_time_seconds": train_seconds,
            "val_inference_time_seconds": val["inference_seconds"],
            "batch_size": batch_size, "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "n_train": epoch_samples, "n_val": val["n_patients"],
        }
        history.append(row)
        print(
            f"[{dsname} fold={fold} ablation={ablation or 'none'}] "
            f"epoch {epoch+1}/{cfg['training']['max_epochs']} "
            f"train_loss={train_loss:.4f} val_cindex={val['cindex']:.4f} "
            f"train_time={train_seconds:.2f}s val_inference_time={val['inference_seconds']:.2f}s "
            f"batch_size={batch_size} effective_batch_size={effective_batch_size}"
        )

        if debug_grad and epoch_grad_rows:
            pd.DataFrame(epoch_grad_rows).to_csv(grad_path, mode="a", header=not grad_path.exists(), index=False)
        if debug_path and epoch_path_audit_rows:
            pd.DataFrame(epoch_path_audit_rows).to_csv(path_audit_path, mode="a", header=not path_audit_path.exists(), index=False)

        score = val["cindex"]
        best_path = output / "best.pt"
        improved = np.isfinite(score) and score > best
        if improved or not best_path.exists():
            if np.isfinite(score): best = score
            best_epoch = epoch + 1
            patience = 0
            torch.save({
                "model_state": model.state_dict(), "config": cfg, "dataset": dsname,
                "fold": fold, "ablation": ablation or "none", "genes": fold_data["genes"],
                "fold_metadata": fold_data["metadata"], "batch_size": batch_size,
            }, best_path)
        else:
            patience += 1
            if early_stopping and patience >= int(cfg["training"].get("patience", 10)):
                print(f"[{dsname} fold={fold}] early stopping at epoch {epoch+1}")
                break

    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val = _evaluate(model, fold_data["val"], device, train_expr, cfg)
    test = _evaluate(model, fold_data["test"], device, train_expr, cfg)

    if cfg["training"].get("save_predictions", True):
        val["predictions"].to_csv(output / "val_predictions.csv", index=False)
        test["predictions"].to_csv(output / "test_predictions.csv", index=False)
        pd.DataFrame(val["routes"]).to_json(output / "val_routing.jsonl", orient="records", lines=True)
        pd.DataFrame(test["routes"]).to_json(output / "test_routing.jsonl", orient="records", lines=True)
        model.eval()
        with torch.no_grad():
            sample = fold_data["train"][0]["gene"].to(device)
            _, _, gd = model.gmi(sample, return_details=True)
        assign = gd["assignment"].cpu().numpy()
        loadings = gd["loadings"].cpu().numpy()
        pd.DataFrame(assign, index=fold_data["genes"], columns=[f"module_{i}" for i in range(assign.shape[1])]).to_csv(output / "gene_module_assignments.csv")
        pd.DataFrame(loadings, index=fold_data["genes"], columns=[f"module_{i}" for i in range(loadings.shape[1])]).to_csv(output / "gene_module_loadings.csv")

    total_seconds = time.perf_counter() - total_start
    mean_train_seconds = float(np.mean([x["train_time_seconds"] for x in history])) if history else float("nan")
    mean_val_seconds = float(np.mean([x["val_inference_time_seconds"] for x in history])) if history else float("nan")
    summary = {
        "dataset": dsname, "fold": fold, "ablation": ablation or "none", "best_epoch": best_epoch,
        "best_val_cindex": float(best) if np.isfinite(best) else None,
        "val_cindex": float(val["cindex"]) if np.isfinite(val["cindex"]) else None,
        "test_cindex": float(test["cindex"]) if np.isfinite(test["cindex"]) else None,
        "val_loss": float(val["loss"]) if np.isfinite(val["loss"]) else None,
        "test_loss": float(test["loss"]) if np.isfinite(test["loss"]) else None,
        "n_train": len(fold_data["train"]), "n_val": len(fold_data["val"]), "n_test": len(fold_data["test"]),
        "val_test_overlap_n": len(fold_data["metadata"]["val_test_overlap"]),
        "batch_size": batch_size, "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "mean_epoch_train_time_seconds": mean_train_seconds,
        "mean_epoch_val_inference_time_seconds": mean_val_seconds,
        "final_val_inference_time_seconds": val["inference_seconds"],
        "test_inference_time_seconds": test["inference_seconds"],
        "total_runtime_seconds": total_seconds,
        # Backward-compatible field consumed by the existing analysis.py aggregator.
        "runtime_seconds": total_seconds,
    }
    atomic_json(summary, output / "fold_result.json")
    return summary


def run_training(config_path, datasets=None, folds=None, ablation=None):
    cfg = load_config(config_path)
    datasets = datasets or cfg["project"]["datasets"]
    folds = folds or cfg["training"]["folds"]
    rows = []
    for ds in datasets:
        for fold in folds:
            rows.append(run_fold(cfg, ds, int(fold), ablation=ablation))
    out = Path(cfg["project"]["output_root"]) / "fold_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    current = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        merged = pd.concat([old, current], ignore_index=True)
        keys = [k for k in ("dataset", "fold", "ablation") if k in merged.columns]
        if keys:
            merged = merged.drop_duplicates(keys, keep="last")
        current = merged
    current.to_csv(out, index=False)
    return current


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--folds", nargs="*", type=int)
    p.add_argument("--ablation", default=None)
    a = p.parse_args()
    run_training(a.config, a.datasets, a.folds, a.ablation)
