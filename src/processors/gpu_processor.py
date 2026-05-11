"""
GPU-side processor used during training.

Performs multi-scale center cropping (global / mid / close-up), per-slice
normalization, and on-the-fly augmentation (affine + random erasing).

The `view` is parameterized so the same class works for any plane.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class GPUProcessor(nn.Module):
    def __init__(self, device, view='sagittal',
                 zoom_mid=0.65, zoom_close=0.55,
                 imagenet_mean=(0.485, 0.456, 0.406),
                 imagenet_std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.device = device
        self.view = view
        self.zoom_mid = zoom_mid
        self.zoom_close = zoom_close

        self.normalize = transforms.Normalize(mean=list(imagenet_mean),
                                              std=list(imagenet_std))
        self.geo_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1),
                                    scale=(0.85, 1.15), shear=10),
        ])
        self.eraser = transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=0)

    def create_multi_scale_view(self, tensor_batch):
        """Generate global + mid + close-up views and concat along channel dim."""
        B, C, D, H, W = tensor_batch.shape

        # Global view (full slice)
        global_view = F.interpolate(tensor_batch, size=(32, 224, 224),
                                    mode='trilinear', align_corners=False)

        # Mid view (with random center jitter at training time)
        crop_h_mid, crop_w_mid = int(H * self.zoom_mid), int(W * self.zoom_mid)
        sh_mid, sw_mid = (H - crop_h_mid) // 2, (W - crop_w_mid) // 2

        if self.training:
            max_dy = (H - crop_h_mid) // 4
            max_dx = (W - crop_w_mid) // 4
            dy = torch.randint(-max_dy, max_dy + 1, (1,), device=self.device).item()
            dx = torch.randint(-max_dx, max_dx + 1, (1,), device=self.device).item()
            sh_mid = np.clip(sh_mid + dy, 0, H - crop_h_mid)
            sw_mid = np.clip(sw_mid + dx, 0, W - crop_w_mid)

        mid_crop = tensor_batch[:, :, :, sh_mid:sh_mid + crop_h_mid,
                                          sw_mid:sw_mid + crop_w_mid]
        mid_view = F.interpolate(mid_crop, size=(32, 224, 224),
                                 mode='trilinear', align_corners=False)

        # Close-up view (centered, no jitter)
        crop_h_close, crop_w_close = int(H * self.zoom_close), int(W * self.zoom_close)
        sh_close, sw_close = (H - crop_h_close) // 2, (W - crop_w_close) // 2
        close_crop = tensor_batch[:, :, :, sh_close:sh_close + crop_h_close,
                                            sw_close:sw_close + crop_w_close]
        close_view = F.interpolate(close_crop, size=(32, 224, 224),
                                   mode='trilinear', align_corners=False)

        return torch.cat([global_view, mid_view, close_view], dim=1)

    def process(self, views_dict, is_training=True):
        raw_tensor = views_dict[self.view]
        ms_tensor = self.create_multi_scale_view(raw_tensor)
        B, C, D, H, W = ms_tensor.shape
        flat = ms_tensor.view(B * D, C, H, W)
        flat = self.normalize(flat)
        if is_training:
            flat = self.geo_transforms(flat)
            flat = self.eraser(flat)
        return flat.reshape(B, C, D, H, W)
