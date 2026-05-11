"""
MRNet dataset.

Loads volumetric MRI exams (sagittal / coronal / axial) from disk,
applies CLAHE, and returns a {view: tensor} dictionary plus a 3-label vector
[abnormal, acl, meniscus].
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import apply_clahe_3d


class MRNetDataset(Dataset):
    """
    MRNet volumetric dataset.

    Args:
        root_dir: Root of MRNet-v1.0.
        split_type: 'train' or 'valid'.
        plane: 'sagittal', 'coronal' or 'axial'.
        transform: optional callable applied after CLAHE (unused by default).
        cache_to_ram: cache uint8 volumes in RAM after first load.
    """

    def __init__(self, root_dir, split_type='train', plane='sagittal',
                 transform=None, cache_to_ram=True):
        self.root_dir = root_dir
        self.split_type = split_type
        self.plane = plane
        self.transform = transform
        self.cache_to_ram = cache_to_ram

        # Read CSV with id as string to preserve leading zeros
        self.abnormal = pd.read_csv(
            os.path.join(root_dir, f'{split_type}-abnormal.csv'),
            header=None, names=['id', 'label'], dtype={'id': str}
        )
        self.acl = pd.read_csv(
            os.path.join(root_dir, f'{split_type}-acl.csv'),
            header=None, names=['id', 'label'], dtype={'id': str}
        )
        self.meniscus = pd.read_csv(
            os.path.join(root_dir, f'{split_type}-meniscus.csv'),
            header=None, names=['id', 'label'], dtype={'id': str}
        )

        self.case_ids = self.abnormal['id'].tolist()
        self.labels = {
            'abnormal': dict(zip(self.abnormal['id'], self.abnormal['label'])),
            'acl': dict(zip(self.acl['id'], self.acl['label'])),
            'meniscus': dict(zip(self.meniscus['id'], self.meniscus['label']))
        }

        self.cache = {}

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]

        if self.cache_to_ram and case_id in self.cache:
            img_array = self.cache[case_id]  # uint8
        else:
            # Force 4-digit filename (e.g. 0626.npy)
            try:
                formatted_id = f"{int(case_id):04d}"
            except ValueError:
                formatted_id = str(case_id)

            img_path = os.path.join(
                self.root_dir, self.split_type, self.plane, f'{formatted_id}.npy'
            )

            # Fallback: filename without leading zeros (e.g. 626.npy)
            if not os.path.exists(img_path):
                img_path_alt = os.path.join(
                    self.root_dir, self.split_type, self.plane, f'{case_id}.npy'
                )
                if os.path.exists(img_path_alt):
                    img_path = img_path_alt
                else:
                    raise FileNotFoundError(
                        f"Cannot find file for case {case_id} at {img_path}"
                    )

            img_tensor = np.load(img_path)
            img_array = apply_clahe_3d(img_tensor)  # uint8

            if self.cache_to_ram:
                self.cache[case_id] = img_array

        # Normalize to float [0, 1] and convert to tensor
        final_tensor = torch.FloatTensor(img_array.astype(np.float32) / 255.0)

        label_tensor = torch.FloatTensor([
            self.labels['abnormal'][case_id],
            self.labels['acl'][case_id],
            self.labels['meniscus'][case_id]
        ])

        return {self.plane: final_tensor}, label_tensor
