"""
Inference entry point.

Usage:
    python scripts/inference.py
    python scripts/inference.py --view sagittal \
        --checkpoint_path ./best_model_sagittal_sota_1.pth \
        --batch_size 16 --use_tta true
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from configs import load_config, apply_overrides, print_config, resolve_view_paths
from src.engine import run_inference


def parse_args():
    p = argparse.ArgumentParser(description="Inference on MRNet validation set")
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--root_dir", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, default=None)

    p.add_argument("--view", type=str, default=None,
                   choices=["sagittal", "coronal", "axial"])
    p.add_argument("--target_depth", type=int, default=None)
    p.add_argument("--cache_to_ram", type=lambda x: x.lower() == "true",
                   default=None)

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--zoom_mid", type=float, default=None)
    p.add_argument("--zoom_close", type=float, default=None)
    p.add_argument("--use_tta", type=lambda x: x.lower() == "true",
                   default=None, help="true/false")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    overrides = {
        "paths.root_dir":            args.root_dir,
        "paths.checkpoint_path":     args.checkpoint_path,

        "data.view":                 args.view,
        "data.target_depth":         args.target_depth,
        "data.cache_to_ram":         args.cache_to_ram,

        "inference.batch_size":      args.batch_size,
        "inference.num_workers":     args.num_workers,
        "inference.zoom_mid":        args.zoom_mid,
        "inference.zoom_close":      args.zoom_close,
        "inference.use_tta":         args.use_tta,
    }
    cfg = apply_overrides(cfg, overrides)
    cfg = resolve_view_paths(cfg)
    print_config(cfg, title="Inference Config")

    run_inference(cfg)


if __name__ == "__main__":
    main()