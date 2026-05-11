"""
Visualization helpers used by the Grad-CAM script.

- `ModelWrapper`: reshapes 2D Grad-CAM input into the 5D layout expected by
  `SingleViewMemoryNet`.
- `preprocess_input`: builds the multi-scale input from a raw volume.
- `get_bg_image`: extract a single background slice for overlay.
- `overlay_medical_style`: medical-glow heatmap overlay using JET colormap.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class ModelWrapper(nn.Module):
    """
    Adapter exposing the 5D model as if it were a per-slice 2D classifier,
    which is what pytorch_grad_cam expects.
    """

    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x_2d_batch):
        x_3d = x_2d_batch.permute(1, 0, 2, 3)
        x_5d = x_3d.unsqueeze(0)
        logits, _ = self.model(x_5d)
        return logits.repeat(x_2d_batch.shape[0], 1)


def preprocess_input(tensor_3d_raw, zoom_mid=0.70, zoom_close=0.55,
                     mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)):
    """
    Convert a raw (D, H, W) volume into the (D, 3, 224, 224) tensor expected
    by `ModelWrapper`.
    """
    D = tensor_3d_raw.shape[0]
    indices = torch.linspace(0, D - 1, 32).long() if D != 32 else torch.arange(32)
    tensor_3d = tensor_3d_raw[indices]

    _, H, W = tensor_3d.shape
    tensor_3d = tensor_3d.unsqueeze(0).unsqueeze(0)

    global_view = F.interpolate(tensor_3d, size=(32, 224, 224),
                                mode='trilinear', align_corners=False)

    crop_h, crop_w = int(H * zoom_mid), int(W * zoom_mid)
    sh, sw = (H - crop_h) // 2, (W - crop_w) // 2
    mid_crop = tensor_3d[:, :, :, sh:sh + crop_h, sw:sw + crop_w]
    mid_view = F.interpolate(mid_crop, size=(32, 224, 224),
                             mode='trilinear', align_corners=False)

    crop_h, crop_w = int(H * zoom_close), int(W * zoom_close)
    sh, sw = (H - crop_h) // 2, (W - crop_w) // 2
    close_crop = tensor_3d[:, :, :, sh:sh + crop_h, sw:sw + crop_w]
    close_view = F.interpolate(close_crop, size=(32, 224, 224),
                               mode='trilinear', align_corners=False)

    input_tensor = torch.cat([global_view, mid_view, close_view], dim=1)
    input_tensor = input_tensor.view(3, 32, 224, 224).permute(1, 0, 2, 3)

    transform_norm = transforms.Normalize(mean=list(mean), std=list(std))
    input_tensor = transform_norm(input_tensor)
    return input_tensor


def get_bg_image(tensor_3d_raw, slice_idx=16):
    """Extract a single normalized RGB slice for visualization overlay."""
    D = tensor_3d_raw.shape[0]
    indices = torch.linspace(0, D - 1, 32).long() if D != 32 else torch.arange(32)
    tensor_3d = tensor_3d_raw[indices]
    raw_slice = tensor_3d[slice_idx].cpu().numpy()
    raw_slice = (raw_slice - np.min(raw_slice)) / (np.max(raw_slice) - np.min(raw_slice) + 1e-8)
    return np.stack([raw_slice] * 3, axis=-1)


def overlay_medical_style(img_rgb, heatmap):
    """Apply a glow-style JET heatmap overlay onto a grayscale RGB slice."""
    if heatmap.shape[:2] != img_rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    heatmap = np.power(heatmap, 3)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    img_bgr = np.uint8(255 * img_rgb)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    alpha = heatmap[:, :, np.newaxis] * 0.55
    overlay = img_bgr * (1.0 - alpha) + heatmap_color * alpha

    return cv2.cvtColor(np.uint8(overlay), cv2.COLOR_BGR2RGB)
