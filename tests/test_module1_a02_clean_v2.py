"""Synthetic geometry/split regression tests; real Kaggle evidence is separate."""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / "code" / "module1"
sys.path.insert(0, str(MODULE))
from clean_v2_dataset import CleanV2SegDataset, load_fold_index
from clean_v2_transforms import JointSegTransform


def fixture(tmp_path, with_val=True):
    root = tmp_path / "Dataset"
    rows = []
    specs = [(f"Patient{n}", "Polyp", "SEG_TRAIN", "TRAIN") for n in range(1, 11)]
    specs.extend([("Patient11", "Healthy", "SEG_TRAIN", "TRAIN")])
    if with_val:
        specs.append(("Patient12", "Polyp", "SEG_VAL", "VAL"))
    for patient, category, split, outer in specs:
        directory = root / patient / category
        directory.mkdir(parents=True)
        img = np.zeros((30, 40, 3), np.uint8)
        img[8:18, 13:24, 0] = 255
        Image.fromarray(img).save(directory / "example.png")
        mask = ""
        if category == "Polyp":
            mask_dir = root / patient / "BinaryMask"
            mask_dir.mkdir()
            gt = np.zeros((30, 40), np.uint8)
            gt[8:18, 13:24] = 255
            Image.fromarray(gt).save(mask_dir / "example.png")
            mask = f"{patient}/BinaryMask/example.png"
        rows.append({"fold": "0", "image_id": f"{patient}_{category}",
                     "patient_id": patient, "split_group_id": patient,
                     "category": category, "image_path": f"{patient}/{category}/example.png",
                     "mask_path": mask, "sha256": f"unique_{patient}",
                     "outer_split": outer, "inner_split": split,
                     "protocol_version": "inner_clean_v2_v1"})
    path = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return root, path


def test_fold_patient_group_split_and_shapes(tmp_path):
    root, csv = fixture(tmp_path)
    df = load_fold_index(csv, "0", strict_clean_v2=False)
    assert len(df) == 12
    train = CleanV2SegDataset(root, csv, 0, "SEG_TRAIN", return_metadata=True,
                              strict_clean_v2=False)
    assert len(train) == 10
    sample = train[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    assert set(torch.unique(sample["mask"]).tolist()).issubset({0.0, 1.0})
    assert sample["split"] == "SEG_TRAIN"


def test_healthy_mask_zeros_and_val_no_augmentation(tmp_path):
    root, csv = fixture(tmp_path)
    train_all = CleanV2SegDataset(root, csv, "0", "SEG_TRAIN", include_healthy=True,
                                  augment=False, return_metadata=True, strict_clean_v2=False)
    healthy = next(train_all[i] for i in range(len(train_all))
                   if train_all[i]["category"] == "Healthy")
    assert healthy["mask"].sum().item() == 0
    val = CleanV2SegDataset(root, csv, "0", "SEG_VAL", return_metadata=True,
                            strict_clean_v2=False)
    assert torch.equal(val[0]["image"], val[0]["image"])
    with pytest.raises(ValueError, match="forbidden"):
        CleanV2SegDataset(root, csv, "0", "SEG_VAL", augment=True,
                          strict_clean_v2=False)


def test_joint_transform_uses_identical_geometry():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    a[12:33, 15:37, 0] = 255
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[12:33, 15:37] = 255
    t = JointSegTransform(64, augment=True, rng=random.Random(2))
    x, y = t(Image.fromarray(a), Image.fromarray(gt))
    mask = y[0].numpy() > 0.5
    # For 0/255 red-only synthetic foreground, ImageNet-normalized red>0
    # detects the RGB target, including under flips/nearest rotation.
    img_region = x[0].numpy() > 0.0
    assert np.logical_and(mask, img_region).sum() / max(mask.sum(), 1) > 0.85
    assert set(np.unique(y.numpy())).issubset({0., 1.})


def test_detect_group_leakage(tmp_path):
    _root, csv = fixture(tmp_path)
    df = pd.read_csv(csv, dtype=str)
    df.loc[df["patient_id"] == "Patient12", "split_group_id"] = "Patient1"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="group leakage"):
        load_fold_index(csv, 0, strict_clean_v2=False)


def test_detect_outer_test_leakage(tmp_path):
    _root, csv = fixture(tmp_path)
    df = pd.read_csv(csv, dtype=str)
    df.loc[df["patient_id"] == "Patient1", "outer_split"] = "TEST"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Outer split mismatch"):
        load_fold_index(csv, 0, strict_clean_v2=False)


def test_real_pair_auditor_with_synthetic_fixture(tmp_path):
    root, csv = fixture(tmp_path)
    out = tmp_path / "audit"
    cp = subprocess.run(
        [sys.executable, str(MODULE / "a02_verify.py"), "--dataset", str(root),
         "--splits", str(csv), "--fold", "0", "--out", str(out),
         "--samples", "10", "--no-strict-counts"],
        capture_output=True, text=True,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert (out / "ten_real_pairs.png").is_file()
    assert (out / "ten_real_pairs.csv").is_file()
    import json
    result = json.loads((out / "a02_audit.json").read_text())
    assert result["result"] == "PASS"
    assert result["gt_pairs_verified"] == 10
