"""Synchronized RGB/mask transforms for FAR Clean V2, Module 1 A02.

- RGB is resized bilinearly; labels are resized nearest-neighbor.
- Random geometry is sampled once and applied to both RGB and mask.
- Random augmentation is ONLY enabled by the Dataset on SEG_TRAIN.
- Input normalization matches ImageNet/PraNet inference at 256x256.
"""
from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)


class JointSegTransform:
    def __init__(self, size: int = 256, augment: bool = False,
                 rng: Optional[random.Random] = None) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)
        self.augment = bool(augment)
        self.rng = rng if rng is not None else random

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.size != mask.size:
            raise ValueError(f"RGB/mask native-size mismatch: {image.size} vs {mask.size}")
        image = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment:
            if self.rng.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if self.rng.random() < 0.5:
                image, mask = ImageOps.flip(image), ImageOps.flip(mask)
            if self.rng.random() < 0.3:
                angle = self.rng.uniform(-15.0, 15.0)
                image = image.rotate(angle, Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, Image.Resampling.NEAREST, fillcolor=0)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        rgb = np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1))
        gt = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        return torch.from_numpy(rgb), torch.from_numpy(gt[None, :, :].copy())
