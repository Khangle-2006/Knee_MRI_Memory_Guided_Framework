"""
Grad-CAM visualization entry point.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --view axial \
        --checkpoint_path ./best_model_axial.pth \
        --prob_threshold 0.6 --target_count 10
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from configs import load_config, apply_overrides, print_config, resolve_view_paths
from src.engine import run_visualization


def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM++ visualization")
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--root_dir", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)

    p.add_argument("--view", type=str, default=None,
                   choices=["sagittal", "coronal", "axial"])

    p.add_argument("--prob_threshold", type=float, default=None)
    p.add_argument("--target_count", type=int, default=None,
                   help="Number of cases per task to visualize "
                        "(use -1 to scan all).")
    p.add_argument("--slice_idx", type=int, default=None)

    p.add_argument("--zoom_mid", type=float, default=None)
    p.add_argument("--zoom_close", type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    overrides = {
        "paths.root_dir":             args.root_dir,
        "paths.checkpoint_path":      args.checkpoint_path,
        "paths.save_dir":             args.save_dir,

        "data.view":                  args.view,

        "visualize.prob_threshold":   args.prob_threshold,
        "visualize.target_count":     args.target_count,
        "visualize.slice_idx":        args.slice_idx,

        "inference.zoom_mid":         args.zoom_mid,
        "inference.zoom_close":       args.zoom_close,
    }
    cfg = apply_overrides(cfg, overrides)
    cfg = resolve_view_paths(cfg)
    print_config(cfg, title="Visualization Config")

    run_visualization(cfg)


if __name__ == "__main__":
    main()