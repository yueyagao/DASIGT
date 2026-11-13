from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .utils import safe_torch_load, patient_id
from .survival import fit_time_cutpoints, discretize_time


def read_split_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    mapping = {}
    for col in ("train", "val", "test"):
        if col not in df.columns:
            raise ValueError(f"Split file must contain train,val,test columns: {path}")
        vals = []
        for v in df[col].tolist():
            pid = patient_id(v)
            if pid:
                vals.append(pid)
        mapping[col] = list(dict.fromkeys(vals))
    return mapping


def load_feature_tokens(paths, expected_dim, max_tokens):
    if isinstance(paths, str):
        paths = [x for x in paths.split(";") if x]
    chunks = []
    for p in paths:
        x = safe_torch_load(p)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError(f"Expected [tokens,dim] features in {p}; got shape {tuple(x.shape)}")
        if x.shape[-1] != expected_dim:
            raise ValueError(f"Feature dim mismatch for {p}: got {x.shape[-1]}, expected {expected_dim}")
        chunks.append(x)
    if not chunks:
        raise FileNotFoundError("No feature files supplied")
    x = torch.cat(chunks, dim=0).float()
    if max_tokens and x.shape[0] > max_tokens:
        idx = torch.linspace(0, x.shape[0] - 1, int(max_tokens)).long()
        x = x[idx]
    if x.shape[0] == 0:
        raise ValueError("Feature file contains zero tokens")
    return x


class PatientDataset(Dataset):
    def __init__(self, patients, manifest, expression, genes, means, stds, cfg, cutpoints):
        self.patients = list(patients)
        self.manifest = manifest.set_index("case_id")
        self.expression = expression
        self.genes = list(genes)
        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)
        self.cfg = cfg
        self.cutpoints = cutpoints
        self.bins = cfg["survival"]["time_bins"]
        self.features = cfg["features"]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        row = self.manifest.loc[pid]
        image = load_feature_tokens(row.image_paths, self.features["image_dim"], self.features.get("max_image_tokens"))
        text = load_feature_tokens(row.text_path, self.features["text_dim"], self.features.get("max_text_tokens"))
        raw = self.expression.loc[self.genes, pid].to_numpy(dtype=np.float32)
        gene = (raw - self.means) / self.stds
        gene = np.nan_to_num(gene, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y = discretize_time(float(row.time), self.cutpoints, self.bins)
        return {
            "case_id": pid,
            "image": image,
            "text": text,
            "gene": torch.from_numpy(gene),
            "time": float(row.time),
            "event": int(row.event),
            "censoring": int(row.censoring),
            "y": int(y),
        }


def collate_patients(items):
    """Pad variable-length image/text tokens and return True=valid masks.

    Image/text tensors are padded only to the longest sequence in the current batch.
    The model must pass the masks into every attention and pooling operation.
    """
    if not items:
        raise ValueError("Cannot collate an empty batch")
    image_list = [x["image"].float() for x in items]
    text_list = [x["text"].float() for x in items]
    image_lengths = torch.tensor([x.shape[0] for x in image_list], dtype=torch.long)
    text_lengths = torch.tensor([x.shape[0] for x in text_list], dtype=torch.long)
    if torch.any(image_lengths <= 0) or torch.any(text_lengths <= 0):
        raise ValueError("Each patient must have at least one image and one text token")

    image = pad_sequence(image_list, batch_first=True, padding_value=0.0)
    text = pad_sequence(text_list, batch_first=True, padding_value=0.0)
    image_mask = torch.arange(image.shape[1]).unsqueeze(0) < image_lengths.unsqueeze(1)
    text_mask = torch.arange(text.shape[1]).unsqueeze(0) < text_lengths.unsqueeze(1)

    return {
        "case_id": [x["case_id"] for x in items],
        "image": image,
        "image_mask": image_mask,
        "text": text,
        "text_mask": text_mask,
        "gene": torch.stack([x["gene"] for x in items], dim=0),
        "time": torch.tensor([x["time"] for x in items], dtype=torch.float32),
        "event": torch.tensor([x["event"] for x in items], dtype=torch.float32),
        "censoring": torch.tensor([x["censoring"] for x in items], dtype=torch.long),
        "y": torch.tensor([x["y"] for x in items], dtype=torch.long),
        "image_lengths": image_lengths,
        "text_lengths": text_lengths,
    }


def prepare_fold(real_data_dir, split_csv, cfg, dataset_name, fold, output_dir):
    real_dir = Path(real_data_dir) / dataset_name.upper()
    manifest = pd.read_csv(real_dir / "manifest.csv")
    manifest["case_id"] = manifest.case_id.map(patient_id)
    manifest = manifest.drop_duplicates("case_id").set_index("case_id", drop=False)
    split = read_split_csv(split_csv)
    valid_ids = set(manifest.index)
    missing = {k: sorted(set(v) - valid_ids) for k, v in split.items()}
    if any(missing.values()):
        warnings.warn(f"Fold {fold} split has IDs not in complete tri-modal manifest: { {k: len(v) for k, v in missing.items()} }")
    split = {k: [p for p in v if p in valid_ids] for k, v in split.items()}
    if not split["train"]:
        raise ValueError(f"No usable train patients in {split_csv}")
    overlap = sorted(set(split["val"]) & set(split["test"]))
    if overlap and not cfg["training"].get("allow_val_test_overlap", True):
        raise ValueError(f"Validation/test overlap ({len(overlap)}) not allowed")
    if overlap:
        warnings.warn(f"Fold {fold}: validation/test overlap contains {len(overlap)} patients; permitted by config.")

    expr = pd.read_csv(real_dir / "gene_expression_prior_overlap.csv", index_col=0, low_memory=False)
    expr.index = expr.index.astype(str)
    expr.columns = [patient_id(c) for c in expr.columns]
    expr = expr.loc[:, ~pd.Index(expr.columns).duplicated()]

    prior_map = {}
    with open(real_dir / "gene2vec.txt", "r", encoding="utf-8", errors="ignore") as f:
        dim = cfg["features"]["gene2vec_dim"]
        for line in f:
            parts = line.strip().split()
            if len(parts) == dim + 1:
                try:
                    prior_map[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
                except ValueError:
                    pass
    available = [g for g in expr.index if g in prior_map]
    expr = expr.loc[available]

    train_expr = expr.reindex(columns=split["train"])
    selection = cfg["features"].get("gene_selection", "variance")
    if selection != "variance":
        raise ValueError(f"Unsupported gene_selection={selection}; currently only training-fold variance ranking is implemented.")
    variances = train_expr.var(axis=1, skipna=True).fillna(0)
    topn = min(int(cfg["features"]["top_gene_num"]), len(variances))
    genes = variances.sort_values(ascending=False, kind="stable").head(topn).index.tolist()
    train_expr = train_expr.loc[genes]
    means = train_expr.mean(axis=1).fillna(0).to_numpy(np.float32)
    stds = train_expr.std(axis=1).replace(0, 1).fillna(1).to_numpy(np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds)
    train_mat = train_expr.T.to_numpy(np.float32)
    train_mat = np.nan_to_num((train_mat - means[None, :]) / stds[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    train_mat = torch.tensor(train_mat, dtype=torch.float32)

    events = manifest.loc[split["train"], "event"].to_numpy(dtype=int)
    times = manifest.loc[split["train"], "time"].to_numpy(dtype=float)
    cutpoints = fit_time_cutpoints(times, events, cfg["survival"]["time_bins"], events_only=cfg["survival"].get("fit_cutpoints_on_train_events_only", True))
    prior = np.stack([prior_map[g] for g in genes]).astype(np.float32)

    fold_dir = Path(output_dir) / dataset_name.upper() / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": dataset_name.upper(), "fold": fold,
        "train_ids": split["train"], "val_ids": split["val"], "test_ids": split["test"],
        "val_test_overlap": overlap, "selected_genes": genes,
        "gene_means": means.tolist(), "gene_stds": stds.tolist(),
        "time_cutpoints": cutpoints.tolist(), "n_train": len(split["train"]),
        "n_val": len(split["val"]), "n_test": len(split["test"]),
    }
    (fold_dir / "fold_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.DataFrame({"gene": genes, "train_variance": variances.loc[genes].to_numpy()}).to_csv(fold_dir / "selected_genes.csv", index=False)

    def ds(which):
        return PatientDataset(split[which], manifest, expr, genes, means, stds, cfg, cutpoints)

    return {
        "train": ds("train"), "val": ds("val"), "test": ds("test"),
        "prior": torch.tensor(prior), "train_expression": train_mat,
        "metadata": meta, "manifest": manifest, "expression": expr,
        "genes": genes, "means": means, "stds": stds, "cutpoints": cutpoints,
    }
