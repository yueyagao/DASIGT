from .config import load_config
from .train import run_training


def run_ablations(config_path, datasets=None, folds=None):
    cfg=load_config(config_path)
    variants=cfg.get("ablation",{}).get("variants",[])
    for variant in variants:
        print(f"\n===== Ablation: {variant} =====")
        run_training(config_path,datasets=datasets,folds=folds,ablation=variant)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--datasets",nargs="*"); p.add_argument("--folds",nargs="*",type=int); a=p.parse_args()
    run_ablations(a.config,a.datasets,a.folds)
