"""
Evaluation engine: TTA inference + optimal threshold reporting.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import MRNetDataset, FastMRNetDataset, make_collate_fn
from ..models import SingleViewMemoryNet
from ..processors import InferenceProcessor
from ..utils import get_optimal_threshold, compute_metrics


def evaluate_model(model, dataloader, processor, view, device, use_tta=True):
    model.eval()
    all_preds, all_labels = [], []

    tta_note = "with TTA" if use_tta else "no TTA"
    print(f"--> Evaluating ({tta_note}) with Optimal Thresholding (Youden's J)...")

    with torch.no_grad():
        for batch_views, batch_labels in dataloader:
            views_gpu = {k: v.to(device, non_blocking=True)
                         for k, v in batch_views.items()}

            inputs_orig = processor.process_batch(views_gpu, flip=False)
            with torch.amp.autocast('cuda'):
                logits_orig, _ = model(inputs_orig)
            preds = torch.sigmoid(logits_orig)

            if use_tta:
                inputs_flip = processor.process_batch(views_gpu, flip=True)
                with torch.amp.autocast('cuda'):
                    logits_flip, _ = model(inputs_flip)
                preds = (preds + torch.sigmoid(logits_flip)) / 2.0

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    tasks = ['Abnormal', 'ACL', 'Meniscus']
    print("-" * 100)
    print(f"{'TASK':<10} | {'AUC':<7} | {'ACC':<7} | {'SENS':<7} | "
          f"{'SPEC':<7} | {'F1':<7} | {'Best Thresh':<10}")
    print("-" * 100)

    final_metrics = []
    for i, task in enumerate(tasks):
        y_t = all_labels[:, i]
        y_p = all_preds[:, i]

        opt_thresh = get_optimal_threshold(y_t, y_p)
        m = compute_metrics(y_t, y_p, threshold=opt_thresh)

        print(f"{task:<10} | {m['AUC']:.4f}  | {m['ACC']:.4f}  | "
              f"{m['SENS']:.4f}  | {m['SPEC']:.4f}  | "
              f"{m['F1']:.4f}  | {m['Thresh']:.4f}")
        final_metrics.append(m)

    print("-" * 100)
    avg_auc = np.mean([x['AUC'] for x in final_metrics])
    avg_f1  = np.mean([x['F1']  for x in final_metrics])
    print(f"{'AVERAGE':<10} | {avg_auc:.4f}  | Mean F1: {avg_f1:.4f}")
    print("=" * 100)

    return final_metrics


def run_inference(cfg):
    """Top-level entry point for `scripts/inference.py`."""
    view = cfg.data.view
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_valid = MRNetDataset(cfg.paths.root_dir, 'valid', plane=view,
                             transform=None,
                             cache_to_ram=cfg.data.cache_to_ram)
    val_dataset = FastMRNetDataset(raw_valid, target_depth=cfg.data.target_depth,
                                   view=view)

    collate = make_collate_fn(view)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.inference.batch_size, shuffle=False,
        num_workers=cfg.inference.num_workers,
        collate_fn=collate, pin_memory=True,
    )

    processor = InferenceProcessor(
        device, view=view,
        zoom_mid=cfg.inference.zoom_mid, zoom_close=cfg.inference.zoom_close,
        imagenet_mean=tuple(cfg.norm.mean), imagenet_std=tuple(cfg.norm.std),
    ).to(device)

    model = SingleViewMemoryNet(dropout=0.0).to(device)

    if os.path.exists(cfg.paths.checkpoint_path):
        ckpt = torch.load(cfg.paths.checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        # Strip the 'module.' prefix if the checkpoint was saved via DataParallel
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        print(f"--> Weights Loaded from {cfg.paths.checkpoint_path}")
    else:
        print(f"!!! Checkpoint not found at {cfg.paths.checkpoint_path}")

    return evaluate_model(model, val_loader, processor, view, device,
                          use_tta=cfg.inference.use_tta)
