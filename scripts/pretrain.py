"""
Backbone pretraining entry point (ModAn-MulSupCon on RadImageNet).

Usage:
    python scripts/pretrain.py
    python scripts/pretrain.py --epochs 50 --batch_size 256 --lr 0.03
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from configs import load_config, apply_overrides, print_config
from src.engine import run_pretraining


def parse_args():
    p = argparse.ArgumentParser(description="ModAn-MulSupCon backbone pretraining")
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--root_dir", type=str, default=None,
                   help="RadImageNet root directory.")
    p.add_argument("--checkpoint_path", type=str, default=None)

    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--tau_threshold", type=float, default=None)
    p.add_argument("--embed_dim", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    overrides = {
        "pretrain.root_dir":         args.root_dir,
        "pretrain.checkpoint_path":  args.checkpoint_path,
        "pretrain.batch_size":       args.batch_size,
        "pretrain.num_workers":      args.num_workers,
        "pretrain.epochs":           args.epochs,
        "pretrain.lr":               args.lr,
        "pretrain.momentum":         args.momentum,
        "pretrain.weight_decay":     args.weight_decay,
        "pretrain.temperature":      args.temperature,
        "pretrain.tau_threshold":    args.tau_threshold,
        "pretrain.embed_dim":        args.embed_dim,
    }
    cfg = apply_overrides(cfg, overrides)
    print_config(cfg, title="Pretrain Config")

    run_pretraining(cfg)


if __name__ == "__main__":
    main()
