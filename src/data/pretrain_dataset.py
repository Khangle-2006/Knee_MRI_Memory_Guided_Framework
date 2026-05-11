"""
RadImageNet pretraining dataset.

Auto-discovers MODALITY/ANATOMY/PATHOLOGY/image.* structure under root_dir
and emits (image, modality_idx, anatomy_idx) tuples for ModAn-MulSupCon
pretraining.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class RadImageNetDataset(Dataset):
    """
    Expected layout: root_dir/MODALITY/ANATOMY/PATHOLOGY/image.png
    """

    def __init__(self, root_dir, transform=None, save_map_path='class_mapping.json'):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # 1. Discover modalities (level 1)
        self.modalities = sorted(
            [d.name for d in os.scandir(root_dir) if d.is_dir()]
        )
        self.modality_to_idx = {n: i for i, n in enumerate(self.modalities)}

        # 2. Discover anatomies (level 2) across all modalities
        anatomy_names = set()
        for mod in self.modalities:
            mod_path = os.path.join(root_dir, mod)
            anats = [d.name for d in os.scandir(mod_path) if d.is_dir()]
            anatomy_names.update(anats)

        self.anatomies = sorted(list(anatomy_names))
        self.anatomy_to_idx = {a: i for i, a in enumerate(self.anatomies)}

        print(f"--> Found {len(self.modalities)} Modalities: {self.modalities}")
        print(f"--> Found {len(self.anatomies)} Anatomies: {self.anatomies}")

        # Persist mapping for later reference
        with open(save_map_path, 'w') as f:
            json.dump({
                'modality_map': self.modality_to_idx,
                'anatomy_map': self.anatomy_to_idx
            }, f, indent=4)
        print(f"--> Saved class mapping to '{save_map_path}'")

        # 3. Crawl image files
        print("Indexing image files...")
        self._crawl_data()
        print(f"--> Total: {len(self.samples)} images ready for training.")

    def _crawl_data(self):
        for mod_name in self.modalities:
            mod_idx = self.modality_to_idx[mod_name]
            mod_path = os.path.join(self.root_dir, mod_name)

            for anat_name in os.listdir(mod_path):
                anat_path = os.path.join(mod_path, anat_name)
                if not os.path.isdir(anat_path):
                    continue

                if anat_name not in self.anatomy_to_idx:
                    continue
                anat_idx = self.anatomy_to_idx[anat_name]

                # Walk pathology folders -> image files
                for pathol_name in os.listdir(anat_path):
                    pathol_path = os.path.join(anat_path, pathol_name)
                    if not os.path.isdir(pathol_path):
                        continue

                    with os.scandir(pathol_path) as entries:
                        for entry in entries:
                            if entry.is_file() and entry.name.lower().endswith(
                                ('.png', '.jpg', '.jpeg', '.bmp')
                            ):
                                self.samples.append(
                                    (entry.path, mod_idx, anat_idx)
                                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mod_label, anat_label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, mod_label, anat_label

        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros((3, 224, 224)), mod_label, anat_label
