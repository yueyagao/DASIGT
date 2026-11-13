from pathlib import Path
import random, os, re, json
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def patient_id(value):
    if value is None: return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}: return ""
    # TCGA patient IDs are the first three hyphen-separated components.
    m = re.match(r"^(TCGA-[A-Za-z0-9]+-[A-Za-z0-9]+)", s)
    return m.group(1) if m else s


def safe_torch_load(path, map_location="cpu"):
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict):
        for key in ("features", "feature", "embeddings", "x", "data"):
            if key in obj and torch.is_tensor(obj[key]):
                obj = obj[key]; break
        else:
            tensors = [v for v in obj.values() if torch.is_tensor(v)]
            if len(tensors) == 1: obj = tensors[0]
            else: raise ValueError(f"Cannot infer feature tensor from dict in {path}; keys={list(obj)}")
    if not torch.is_tensor(obj): obj = torch.as_tensor(obj)
    obj = obj.float()
    if obj.ndim == 1: obj = obj.unsqueeze(0)
    if obj.ndim > 2: obj = obj.reshape(-1, obj.shape[-1])
    return obj


def atomic_json(data, path):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
