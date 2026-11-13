"""Single entry point for DASIGT workflows."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dasigt.data_prep import run_data_prep
from dasigt.train import run_training
from dasigt.ablation import run_ablations
from dasigt.analysis import run_analysis


def main():
    p=argparse.ArgumentParser()
    p.add_argument("task",choices=["data","train","ablation","analysis"])
    p.add_argument("--config",default="configs/config.yaml")
    p.add_argument("--datasets",nargs="*")
    p.add_argument("--folds",nargs="*",type=int)
    p.add_argument("--variant",default=None)
    a=p.parse_args()
    if a.task=="data": run_data_prep(a.config)
    elif a.task=="train": run_training(a.config,a.datasets,a.folds,a.variant)
    elif a.task=="ablation": run_ablations(a.config,a.datasets,a.folds)
    elif a.task=="analysis": run_analysis(a.config)
if __name__=="__main__": main()
