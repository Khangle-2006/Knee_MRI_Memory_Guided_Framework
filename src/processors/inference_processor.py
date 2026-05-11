"""
Deterministic multi-scale processor used at inference / TTA time.

- No random jitter or augmentation.
- Optional `flip=True` to produce the horizontally-flipped view.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class InferenceProcessor(nn.Module):
    def __init__(self, device, view='sagittal',
                 zoom_mid=0.70, zoom_close=0.55,
                 imagenet_mean=(0.485, 0.456, 0.406),
                 imagenet_std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.device = device
        self.view = view
        self.zoom_mid = zoom_mid
        self.zoom_close = zoom_close
        self.normalize = transforms.Normalize(mean=list(imagenet_mean),
                                              std=list(imagenet_std))

    def create_multi_scale_view(self, tensor_batch):
        B, C, D, H, W = tensor_batch.shape

        global_view = F.interpolate(tensor_batch, size=(32, 224, 224),
                                    mode='trilinear', align_corners=False)

        crop_h_mid, crop_w_mid = int(H * self.zoom_mid), int(W * self.zoom_mid)
        sh_mid, sw_mid = (H - crop_h_mid) // 2, (W - crop_w_mid) // 2

        crop_h_close, crop_w_close = int(H * self.zoom_close), int(W * self.zoom_close)
        sh_close, sw_close = (H - crop_h_close) // 2, (W - crop_w_close) // 2

        mid_crop = tensor_batch[:, :, :, sh_mid:sh_mid + crop_h_mid,
                                          sw_mid:sw_mid + crop_w_mid]
        mid_view = F.interpolate(mid_crop, size=(32, 224, 224),
                                 mode='trilinear', align_corners=False)

        close_crop = tensor_batch[:, :, :, sh_close:sh_close + crop_h_close,
                                            sw_close:sw_close + crop_w_close]
        close_view = F.interpolate(close_crop, size=(32, 224, 224),
                                   mode='trilinear', align_corners=False)

        return torch.cat([global_view, mid_view, close_view], dim=1)

    def process_batch(self, views_gpu, flip=False):
        raw_tensor = views_gpu[self.view]
        if flip:
            raw_tensor = torch.flip(raw_tensor, dims=[-1])
        ms_tensor = self.create_multi_scale_view(raw_tensor)
        B, C, D, H, W = ms_tensor.shape
        flat = ms_tensor.view(B * D, C, H, W)
        norm = self.normalize(flat)
        return norm.reshape(B, C, D, H, W)
