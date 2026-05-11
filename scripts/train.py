"""
Training entry point.

Usage:
    # Defaults (everything from configs/default.yaml)
    python scripts/train.py

    # Override anything from the CLI
    python scripts/train.py --view sagittal --epochs 100 --batch_size 32 \
                            --lr_backbone 1e-5
"""
import argparse
import os
import sys

# Make `src.*` and `configs.*` importable when running as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from configs import load_config, apply_overrides, print_config, resolve_view_paths
from src.engine import run_training


def parse_args():
    p = argparse.ArgumentParser(description="Train SingleViewMemoryNet on MRNet")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config (default: configs/default.yaml).")

    # ---- Paths ----
    p.add_argument("--root_dir", type=str, default=None)
    p.add_argument("--backbone_weights", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, default=None)

    # ---- Data ----
    p.add_argument("--view", type=str, default=None,
                   choices=["sagittal", "coronal", "axial"])
    p.add_argument("--target_depth", type=int, default=None)
    p.add_argument("--cache_to_ram", type=lambda x: x.lower() == "true",
                   default=None, help="true/false")

    # ---- Training ----
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr_backbone", type=float, default=None)
    p.add_argument("--lr_memory", type=float, default=None)
    p.add_argument("--lr_classifier", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--pos_weight", type=float, nargs=3, default=None,
                   metavar=("ABN", "ACL", "MEN"),
                   help="Three pos-weight values for [abnormal, acl, meniscus].")
    p.add_argument("--pct_start", type=float, default=None)
    p.add_argument("--div_factor", type=float, default=None)
    p.add_argument("--final_div_factor", type=float, default=None)
    p.add_argument("--use_dataparallel", type=lambda x: x.lower() == "true",
                   default=None, help="true/false")

    # ---- Multi-scale crop ----
    p.add_argument("--zoom_mid", type=float, default=None)
    p.add_argument("--zoom_close", type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    overrides = {
        "paths.root_dir":             args.root_dir,
        "paths.backbone_weights":     args.backbone_weights,
        "paths.checkpoint_path":      args.checkpoint_path,

        "data.view":                  args.view,
        "data.target_depth":          args.target_depth,
        "data.cache_to_ram":          args.cache_to_ram,

        "train.batch_size":           args.batch_size,
        "train.num_workers":          args.num_workers,
        "train.epochs":               args.epochs,
        "train.lr_backbone":          args.lr_backbone,
        "train.lr_memory":            args.lr_memory,
        "train.lr_classifier":        args.lr_classifier,
        "train.weight_decay":         args.weight_decay,
        "train.dropout":              args.dropout,
        "train.label_smoothing":      args.label_smoothing,
        "train.pos_weight":           args.pos_weight,
        "train.pct_start":            args.pct_start,
        "train.div_factor":           args.div_factor,
        "train.final_div_factor":     args.final_div_factor,
        "train.use_dataparallel":     args.use_dataparallel,

        "crop.zoom_mid":              args.zoom_mid,
        "crop.zoom_close":             args.zoom_close,
    }
    cfg = apply_overrides(cfg, overrides)
    cfg = resolve_view_paths(cfg)
    print_config(cfg, title="Training Config")

    run_training(cfg)


if __name__ == "__main__":
    main()