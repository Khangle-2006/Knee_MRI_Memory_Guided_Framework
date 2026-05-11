"""
ModAn-MulSupCon contrastive pretraining engine for the ResNet18 backbone.

Saves `encoder.state_dict()` to disk so that downstream training can pick it
up via `load_backbone_weights`.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from ..data import RadImageNetDataset, TwoCropTransform
from ..models import ResNetSimCLR
from ..losses import ModAnMulSupConLoss


def run_pretraining(cfg):
    """Top-level entry point for `scripts/pretrain.py`."""

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])

    dataset = RadImageNetDataset(
        root_dir=cfg.pretrain.root_dir,
        transform=TwoCropTransform(train_transform),
    )

    num_modalities = len(dataset.modalities)
    num_anatomies = len(dataset.anatomies)

    dataloader = DataLoader(
        dataset, batch_size=cfg.pretrain.batch_size, shuffle=True,
        num_workers=cfg.pretrain.num_workers, pin_memory=True, drop_last=True,
    )

    # Build a 1-channel ResNet18 (3x3 conv1, stride 1)
    base_resnet = resnet18(weights=None)
    base_resnet.conv1 = nn.Conv2d(
        in_channels=1, out_channels=64,
        kernel_size=3, stride=1, padding=1, bias=False,
    )

    model = ResNetSimCLR(base_resnet, out_dim=cfg.pretrain.embed_dim).cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.pretrain.lr,
        momentum=cfg.pretrain.momentum,
        weight_decay=cfg.pretrain.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.pretrain.epochs,
    )
    criterion = ModAnMulSupConLoss(
        temperature=cfg.pretrain.temperature,
        threshold=cfg.pretrain.tau_threshold,
    )

    print(f"Start Pre-training (Forced 1-Channel): SGD, "
          f"LR={cfg.pretrain.lr}, Batch={cfg.pretrain.batch_size}")

    model.train()
    for epoch in range(cfg.pretrain.epochs):
        running_loss = 0.0
        start = time.time()

        for i, (images, mod_labels, anat_labels) in enumerate(dataloader):
            images = torch.cat(images, dim=0).cuda()  # (2B, 1, 224, 224)

            mod_onehot = F.one_hot(mod_labels, num_classes=num_modalities).float().cuda()
            anat_onehot = F.one_hot(anat_labels, num_classes=num_anatomies).float().cuda()
            labels_single = torch.cat([mod_onehot, anat_onehot], dim=1)
            labels = torch.cat([labels_single, labels_single], dim=0)

            optimizer.zero_grad()
            _, features = model(images)
            loss = criterion(features, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch + 1} | Step {i}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()

        epoch_time = time.time() - start
        print(f"=== Epoch {epoch + 1} Done | "
              f"Avg Loss: {running_loss / len(dataloader):.4f} | "
              f"Time: {epoch_time:.0f}s ===")

        torch.save(model.encoder.state_dict(), cfg.pretrain.checkpoint_path)

    print("PRE-TRAINING FINISHED.")
