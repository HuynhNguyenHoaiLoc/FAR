"""Historical Dataset implementation for reproducing pre-Clean-V2 experiments.

You must explicitly set FAR_ALLOW_LEGACY_SPLITS=1 before instantiation.
Never use this API for new Fold 0 / M2-M4 runs.
"""
import os


import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset



ALLOWED_SPLITS = {
    "SEG_TRAIN",
    "SEG_VAL",
    "META_FIT",
    "CALIBRATION",
    "OUTER_TEST",
    "NEVER_POLYP_TEST",
}


class FARPolypDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        patient_ids,
        split,
        augment=False,
        image_ids=None
    ):
        if os.environ.get("FAR_ALLOW_LEGACY_SPLITS") != "1":
            raise RuntimeError(
                "LEGACY ONLY: set FAR_ALLOW_LEGACY_SPLITS=1 to reproduce historical runs."
            )
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split: {split}")

        if augment and split != "SEG_TRAIN":
            raise ValueError(
                "Augmentation is only allowed for SEG_TRAIN"
            )

        patient_ids = {str(p) for p in patient_ids}

        if not patient_ids:
            raise ValueError("patient_ids cannot be empty")

        df = pd.read_csv(manifest_path)

        self.df = df[
            df["patient_id"].astype(str).isin(patient_ids)
        ].reset_index(drop=True)

        # Nếu có danh sách ảnh chính thức, chỉ đọc đúng các ảnh đó
        if image_ids is not None:
            image_ids = {str(x) for x in image_ids}

            if not image_ids:
                raise ValueError("image_ids cannot be empty")

            available_ids = set(self.df["image_id"].astype(str))
            missing_ids = image_ids - available_ids

            if missing_ids:
                raise ValueError(
                    f"Unknown image IDs: {sorted(missing_ids)[:5]}"
                )

            self.df = self.df[
                self.df["image_id"].astype(str).isin(image_ids)
            ].reset_index(drop=True)

        if self.df.empty:
            raise ValueError("No images found for patient_ids")

        self.split = split
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Import lazily: historical preprocess requires FAR_DATA_ROOT.
        try:
            from .preprocess import load_sample
        except ImportError:
            from preprocess import load_sample
        image, mask = load_sample(row)

        # Augmentation đồng bộ ảnh và mask
        if self.augment:
            if torch.rand(()).item() < 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()

        # Image: HWC -> CHW
        image = np.transpose(image, (2, 0, 1)).copy()

        # Mask: HW -> CHW
        mask = np.expand_dims(mask, axis=0).copy()

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return {
            "image": image,
            "mask": mask,
            "image_id": str(row["image_id"]),
            "patient_id": str(row["patient_id"]),
            "split": self.split,
        }
