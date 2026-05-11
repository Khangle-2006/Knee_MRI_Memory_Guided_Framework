"""
Fast wrapper around `MRNetDataset` for training / inference.

- Resamples each volume to a fixed depth (default 32).
- Provides a `make_collate_fn(view)` factory so the same code can run on any
  of the three anatomical planes ('sagittal', 'coronal', 'axial').
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class FastMRNetDataset(Dataset):
    """
    Wrap an `MRNetDataset` to enforce a fixed temporal depth and a (1, D, H, W)
    tensor layout.

    Args:
        original_dataset: an instantiated `MRNetDataset`.
        target_depth: number of slices after resampling (default 32).
        view: optional explicit view key. If None, the first key emitted by
            the underlying dataset is used (which is the plane it was loaded
            with).
    """

    def __init__(self, original_dataset, target_depth=32, view=None):
        self.dataset = original_dataset
        self.target_depth = target_depth
        # Prefer explicit view, otherwise fall back to the dataset's plane
        self.view = view if view is not None else getattr(
            original_dataset, "plane", "sagittal"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_dict, label = self.dataset[index]
        # Take the first (and only) tensor — guaranteed to be `self.view`
        tensor_3d = list(data_dict.values())[0]

        if isinstance(tensor_3d, np.ndarray):
            tensor_3d = torch.from_numpy(tensor_3d)

        if tensor_3d.ndim == 3:
            D, H, W = tensor_3d.shape
            tensor_3d = tensor_3d.unsqueeze(0)
        else:
            _, D, H, W = tensor_3d.shape

        if D != self.target_depth:
            indices = torch.linspace(0, D - 1, self.target_depth).long()
            tensor_3d = tensor_3d[:, indices, :, :]

        return {self.view: tensor_3d}, label


def make_collate_fn(view):
    """Factory: return a collate_fn that stacks tensors under `view` key."""
    def fast_collate(batch):
        tensors = [item[0][view] for item in batch]
        labels = [item[1] for item in batch]
        return {view: torch.stack(tensors)}, torch.stack(labels)
    return fast_collate
