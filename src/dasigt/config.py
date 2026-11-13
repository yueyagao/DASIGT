from pathlib import Path
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(path).resolve().parent.parent
    for key in ("output_root", "real_use_data_root", "summary_data_process", "summary_analysis"):
        p = Path(cfg["project"][key])
        cfg["project"][key] = str((base / p).resolve()) if not p.is_absolute() else str(p)
    for key in ("raw_root", "split_root", "gene_id_map_file"):
        val = cfg["paths"].get(key, "")
        if val and not Path(val).is_absolute():
            cfg["paths"][key] = str((base / val).resolve())
    return cfg


def dump_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
