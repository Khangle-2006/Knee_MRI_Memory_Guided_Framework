"""
Grad-CAM++ visualization engine.

Scans a validation split, finds high-confidence positives per task, and
saves side-by-side (original | heatmap) PNGs.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ..data import MRNetDataset
from ..models import SingleViewMemoryNet
from ..utils import (
    ModelWrapper, preprocess_input, get_bg_image, overlay_medical_style,
)


def _visualize_task(dataset, model, cam, task_name, task_index,
                    view, save_dir, prob_threshold, target_count,
                    zoom_mid, zoom_close, mean, std, slice_idx, device):
    print(f"\n--> Scanning for High Confidence {task_name} cases "
          f"(> {prob_threshold}) on view '{view}'...")

    count_found = 0
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    for idx in indices:
        if count_found >= target_count:
            break

        case_data, label = dataset[idx]
        vol_raw = case_data[view]
        input_tensor_2d = preprocess_input(
            vol_raw, zoom_mid=zoom_mid, zoom_close=zoom_close,
            mean=mean, std=std,
        ).to(device)

        with torch.no_grad():
            logits = model(input_tensor_2d)
            probs = torch.sigmoid(logits[0]).cpu().numpy()

        task_prob = probs[task_index]
        if task_prob <= prob_threshold:
            continue

        count_found += 1
        print(f"   Found Case {idx}: {task_name} Prob = {task_prob:.4f}")

        input_tensor_2d.requires_grad = True
        rgb_img = get_bg_image(vol_raw, slice_idx=slice_idx)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Left: original slice
        axes[0].imshow(rgb_img)
        gt_label = label.numpy().astype(int)[task_index]
        axes[0].set_title(
            f"Case {idx} - Original MRI ({view})\nGT {task_name}: {gt_label}",
            fontsize=14,
        )
        axes[0].axis('off')

        # Right: Grad-CAM heatmap
        targets = [ClassifierOutputTarget(task_index)]
        grayscale_cam_batch = cam(input_tensor=input_tensor_2d, targets=targets)
        grayscale_cam = grayscale_cam_batch[slice_idx, :]
        visualization = overlay_medical_style(rgb_img, grayscale_cam)

        axes[1].imshow(visualization)
        axes[1].set_title(
            f"Grad-CAM++ Localization\nPrediction: {task_prob:.3f}",
            fontsize=14,
        )
        axes[1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(
            save_dir, f"{view}_{task_name}_only_{task_prob:.2f}_case_{idx}.png"
        )
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()


def run_visualization(cfg):
    view = cfg.data.view
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(cfg.paths.save_dir, exist_ok=True)
    dataset = MRNetDataset(cfg.paths.root_dir, split_type='valid', plane=view,
                           cache_to_ram=False)

    print("--> Loading Model...")
    base_model = SingleViewMemoryNet(dropout=0.0).to(device)
    if os.path.exists(cfg.paths.checkpoint_path):
        ckpt = torch.load(cfg.paths.checkpoint_path, map_location=device)
        state_dict = {k.replace('module.', ''): v
                      for k, v in ckpt['model_state_dict'].items()}
        base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    model = ModelWrapper(base_model)
    target_layers = [base_model.backbone.features_2d[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    total_valid_cases = len(dataset)
    print(f"--> Total Validation Cases: {total_valid_cases}")

    target_count = (total_valid_cases
                    if cfg.visualize.target_count < 0
                    else cfg.visualize.target_count)

    mean = tuple(cfg.norm.mean)
    std = tuple(cfg.norm.std)

    # Run ACL (task_index=1) and Meniscus (task_index=2)
    for task_name, task_index in [("ACL", 1), ("Meniscus", 2)]:
        _visualize_task(
            dataset, model, cam, task_name, task_index, view,
            cfg.paths.save_dir, cfg.visualize.prob_threshold, target_count,
            cfg.inference.zoom_mid, cfg.inference.zoom_close,
            mean, std, cfg.visualize.slice_idx, device,
        )
