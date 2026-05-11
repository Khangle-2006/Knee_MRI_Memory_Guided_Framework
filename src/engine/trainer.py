"""
Training engine for `SingleViewMemoryNet`.

Handles:
- dataloader construction (using the configured view)
- model + DataParallel + optimizer + scheduler setup
- mixed-precision train/val loop
- best-checkpoint saving
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from ..data import MRNetDataset, FastMRNetDataset, make_collate_fn
from ..models import SingleViewMemoryNet
from ..losses import LabelSmoothingBCE
from ..processors import GPUProcessor
from ..utils import load_backbone_weights


def run_training(cfg):
    """
    Run end-to-end training. Expects a fully-resolved `cfg` (SimpleNamespace).
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    view = cfg.data.view
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- Data ----------
    print(f"--> Loading Dataset for View: {view.upper()} ...")
    raw_train = MRNetDataset(cfg.paths.root_dir, 'train', plane=view,
                             transform=None,
                             cache_to_ram=cfg.data.cache_to_ram)
    raw_valid = MRNetDataset(cfg.paths.root_dir, 'valid', plane=view,
                             transform=None,
                             cache_to_ram=cfg.data.cache_to_ram)

    train_dataset = FastMRNetDataset(raw_train, target_depth=cfg.data.target_depth,
                                     view=view)
    valid_dataset = FastMRNetDataset(raw_valid, target_depth=cfg.data.target_depth,
                                     view=view)

    collate = make_collate_fn(view)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=True,
        collate_fn=collate, persistent_workers=True, prefetch_factor=4,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
        collate_fn=collate, persistent_workers=True, prefetch_factor=2,
    )

    # ---------- GPU processor (augmentation + multi-scale crop) ----------
    gpu_processor = GPUProcessor(
        device, view=view,
        zoom_mid=cfg.crop.zoom_mid, zoom_close=cfg.crop.zoom_close,
        imagenet_mean=tuple(cfg.norm.mean),
        imagenet_std=tuple(cfg.norm.std),
    ).to(device)

    # ---------- Model ----------
    model = SingleViewMemoryNet(dropout=cfg.train.dropout).to(device)

    if os.path.exists(cfg.paths.backbone_weights):
        model = load_backbone_weights(model, cfg.paths.backbone_weights)

    if cfg.train.use_dataparallel and torch.cuda.device_count() > 1:
        print(f"--> Enabling DataParallel on {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    def _params(name):
        if hasattr(model, 'module'):
            return getattr(model.module, name).parameters()
        return getattr(model, name).parameters()

    optimizer = optim.AdamW(
        [
            {'params': _params('backbone'),    'lr': cfg.train.lr_backbone},
            {'params': _params('memory'),      'lr': cfg.train.lr_memory},
            {'params': _params('classifier'),  'lr': cfg.train.lr_classifier},
        ],
        weight_decay=cfg.train.weight_decay,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg.train.lr_backbone, cfg.train.lr_memory, cfg.train.lr_classifier],
        steps_per_epoch=len(train_loader), epochs=cfg.train.epochs,
        pct_start=cfg.train.pct_start,
        div_factor=cfg.train.div_factor,
        final_div_factor=cfg.train.final_div_factor,
    )

    pos_weight = torch.tensor(cfg.train.pos_weight, dtype=torch.float32).to(device)
    criterion = LabelSmoothingBCE(smoothing=cfg.train.label_smoothing,
                                  pos_weight=pos_weight)
    scaler = torch.amp.GradScaler('cuda')

    # ---------- Training loop ----------
    best_score = 0.0
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"--> START SINGLE-VIEW ({view.upper()}) TRAINING with {n_gpus} GPUs ...")

    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_views, batch_labels in train_loader:
            views_gpu = {view: batch_views[view].to(device, non_blocking=True)}
            labels_gpu = batch_labels.to(device, non_blocking=True)

            input_tensor = gpu_processor.process(views_gpu, is_training=True)

            with torch.amp.autocast('cuda'):
                logits, _ = model(input_tensor)
                loss = criterion(logits, labels_gpu)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()

        epoch_duration = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_views, batch_labels in valid_loader:
                views_gpu = {view: batch_views[view].to(device, non_blocking=True)}
                labels_gpu = batch_labels.to(device, non_blocking=True)

                input_tensor = gpu_processor.process(views_gpu, is_training=False)
                with torch.amp.autocast('cuda'):
                    logits, _ = model(input_tensor)

                preds = torch.sigmoid(logits)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels_gpu.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        try:
            auc_abn = roc_auc_score(all_labels[:, 0], all_preds[:, 0])
            auc_acl = roc_auc_score(all_labels[:, 1], all_preds[:, 1])
            auc_men = roc_auc_score(all_labels[:, 2], all_preds[:, 2])
            avg_valid = (auc_acl + auc_men) / 2.0

            print(
                f"Ep {epoch + 1:03d} | Loss: {avg_loss:.4f} | "
                f"ABN: {auc_abn:.3f} | ACL: {auc_acl:.3f} | "
                f"MEN: {auc_men:.3f} | AVG: {avg_valid:.3f} | "
                f"Time: {epoch_duration:.1f}s"
            )

            if avg_valid > best_score:
                best_score = avg_valid
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(
                    {'model_state_dict': model_to_save.state_dict(),
                     'score': best_score},
                    cfg.paths.checkpoint_path,
                )
                print(f"   >>> [SAVE] BEST MODEL! ({best_score:.4f})")

        except Exception as e:
            print(f"Error val: {e}")

    print("--> TRAINING FINISHED.")
    return best_score
