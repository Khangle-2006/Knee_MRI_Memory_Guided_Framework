"""
Image transforms used across the project.

- `apply_clahe_3d`: per-slice Contrast Limited Adaptive Histogram Equalization
  that returns a uint8 volume (memory-friendly cache format).
- `TwoCropTransform`: classic SimCLR-style two-view augmentation wrapper.
"""
import cv2
import numpy as np


def apply_clahe_3d(volume, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE slice-by-slice and return a uint8 volume.

    Args:
        volume: ndarray of shape (D, H, W).
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.

    Returns:
        ndarray of shape (D, H, W), dtype=uint8.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_volume = []
    for i in range(volume.shape[0]):
        slice_img = volume[i]

        # Normalize to standard 0-255 uint8 before CLAHE
        if slice_img.max() != 0:
            slice_img_uint8 = cv2.normalize(
                slice_img, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
        else:
            slice_img_uint8 = slice_img.astype(np.uint8)

        enhanced_slice = clahe.apply(slice_img_uint8)
        enhanced_volume.append(enhanced_slice)

    return np.stack(enhanced_volume, axis=0)


class TwoCropTransform:
    """Produce two augmented views of the same input for contrastive learning."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
