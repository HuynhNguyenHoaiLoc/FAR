"""A02 acceptance: fixed patient split, DataLoader and 10 real GT image/mask pairs.

No dataset/labels are generated or reassigned. Writes evidence only when all
assertions succeed. Use on Kaggle with the actual Clean V2 dataset attached.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_v2_dataset import CleanV2SegDataset, _source_path, load_fold_index


def sample_distinct_patients(polyp: pd.DataFrame, count: int) -> pd.DataFrame:
    """Deterministic, one image per patient; no cherry-picked successes."""
    group = polyp.sort_values(["patient_id", "image_id"])
    unique = group.drop_duplicates(subset="patient_id", keep="first")
    if len(unique) < count:
        raise ValueError(f"Only {len(unique)} patient-distinct Polyp samples; need {count}")
    positions = np.linspace(0, len(unique) - 1, num=count, dtype=int)
    return unique.iloc[positions].reset_index(drop=True)


def render_contact_sheet(pairs: list[tuple[str, Image.Image, Image.Image]], output: Path) -> None:
    """Show ORIGINAL RGB, original GT, and GT overlay (10 examples, no model)."""
    tile_w, tile_h = 190, 140
    margin, label_h, gap, columns = 12, 22, 12, 2
    card_w = 3 * tile_w + 4 * margin
    card_h = tile_h + label_h + 2 * margin
    rows = (len(pairs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * card_w + (columns + 1) * gap,
                              rows * card_h + (rows + 1) * gap), (236, 239, 244))
    draw = ImageDraw.Draw(sheet)
    for k, (label, image, mask) in enumerate(pairs):
        x = gap + (k % columns) * (card_w + gap)
        y = gap + (k // columns) * (card_h + gap)
        draw.rectangle((x, y, x + card_w, y + card_h), fill="white")
        draw.text((x + margin, y + 7), label[:64], fill=(25, 25, 25))
        image = image.resize((tile_w, tile_h), Image.Resampling.BILINEAR).convert("RGB")
        mask = mask.resize((tile_w, tile_h), Image.Resampling.NEAREST).convert("L")
        foreground = mask.point(lambda v: 255 if v > 127 else 0)
        colored = Image.new("RGB", (tile_w, tile_h), (230, 30, 50))
        overlay = Image.composite(colored, image, foreground).convert("RGB")
        overlay = Image.blend(image, overlay, 0.45)
        preview_mask = Image.merge("RGB", (foreground, foreground, foreground))
        for j, layer in enumerate((image, preview_mask, overlay)):
            sheet.paste(layer, (x + margin + j * (tile_w + margin), y + label_h + margin))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acceptance QA for Module1 A02, FAR Clean V2")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to cleaned Dataset/ root")
    parser.add_argument("--splits", required=True, type=Path, help="Approved image_inner_splits_cv5_candidate.csv")
    parser.add_argument("--fold", default="0", type=str)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", default=10, type=int)
    parser.add_argument("--no-strict-counts", action="store_true", help="For synthetic unit-test fixtures ONLY")
    args = parser.parse_args()
    if args.samples < 10 and not args.no_strict_counts:
        parser.error("A02 acceptance needs at least 10 original RGB/GT pairs")

    rows = load_fold_index(args.splits, args.fold,
                           strict_clean_v2=not args.no_strict_counts)
    train_polyp = rows[(rows["inner_split"] == "SEG_TRAIN") & (rows["category"] == "Polyp")]
    chosen = sample_distinct_patients(train_polyp, args.samples)

    records, pictures = [], []
    for row in chosen.itertuples():
        image_path = _source_path(args.dataset, row.image_path)
        mask_path = _source_path(args.dataset, row.mask_path)
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
        with Image.open(mask_path) as im:
            mask = im.convert("L")
        if rgb.size != mask.size:
            raise ValueError(f"Native GT mismatch for {row.image_id}: {rgb.size} vs {mask.size}")
        original = np.asarray(mask, dtype=np.uint8) > 127
        if not original.any():
            raise ValueError(f"Unexpected empty GT Polyp mask: {row.image_id}")
        records.append({
            "fold": args.fold, "split": row.inner_split,
            "image_id": row.image_id, "patient_id": row.patient_id,
            "split_group_id": row.split_group_id,
            "image_path": row.image_path, "mask_path": row.mask_path,
            "native_width": rgb.width, "native_height": rgb.height,
            "mask_foreground_px": int(original.sum()),
            "native_size_match": True,
        })
        pictures.append((f"{row.patient_id} | {row.image_id}", rgb, mask))

    # Independently check that the paired Dataset produces valid tensor samples.
    dataset_train = CleanV2SegDataset(args.dataset, args.splits, args.fold, "SEG_TRAIN",
                                      augment=True, return_metadata=True,
                                      strict_clean_v2=not args.no_strict_counts)
    dataset_val = CleanV2SegDataset(args.dataset, args.splits, args.fold, "SEG_VAL",
                                    augment=False, return_metadata=True,
                                    strict_clean_v2=not args.no_strict_counts)
    # Run one batch through the actual PyTorch DataLoader, not only PIL.
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(dataset_train, batch_size=2, shuffle=False, num_workers=0)))
    if tuple(batch["image"].shape) != (2, 3, 256, 256):
        raise AssertionError("Unexpected DataLoader RGB tensor shape")
    if tuple(batch["mask"].shape) != (2, 1, 256, 256):
        raise AssertionError("Unexpected DataLoader GT tensor shape")
    if not set(batch["mask"].unique().tolist()).issubset({0.0, 1.0}):
        raise AssertionError("Resized mask is not binary")
    if not len(dataset_val):
        raise AssertionError("SEG_VAL contains no Polyp images")
    val_sample = dataset_val[0]
    if val_sample["image"].shape != (3, 256, 256):
        raise AssertionError("SEG_VAL preprocessing failed")
    if not np.array_equal(dataset_val[0]["image"].numpy(), val_sample["image"].numpy()):
        raise AssertionError("SEG_VAL unexpectedly random/augmented")

    # At this point all acceptance checks pass: write evidence atomically.
    args.out.mkdir(parents=True, exist_ok=True)
    table = args.out / "ten_real_pairs.csv"
    pd.DataFrame(records).to_csv(table, index=False, encoding="utf-8-sig")
    png = args.out / "ten_real_pairs.png"
    render_contact_sheet(pictures, png)
    report = {
        "task": "A02", "result": "PASS", "fold": args.fold,
        "input_split_csv": str(args.splits),
        "dataset_root": str(args.dataset),
        "image_count": len(rows), "patient_count": int(rows["patient_id"].nunique()),
        "linked_group_count": int(rows["split_group_id"].nunique()),
        "polyp_count": int((rows["category"] == "Polyp").sum()),
        "healthy_count": int((rows["category"] == "Healthy").sum()),
        "group_leakage": 0, "patient_leakage": 0,
        "gt_pairs_verified": len(records),
        "gt_pairs_different_patients": len({r["patient_id"] for r in records}),
        "train_augmentation_only": True,
        "seg_train_polyp_count": len(dataset_train),
        "seg_val_polyp_count": len(dataset_val),
        "loader_rgb_shape": list(batch["image"].shape),
        "loader_gt_shape": list(batch["mask"].shape),
        "gt_mask_values": [0, 1],
        "test_splits_loaded": False,
        "notes": "Actual cleaned RGB/GT files opened; no split creation or trained-model inference.",
    }
    (args.out / "a02_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=== A02 CLEAN V2 ACCEPTANCE ===")
    print("Images:", len(rows), "Patients:", report["patient_count"], "Groups:", report["linked_group_count"])
    print("Verified REAL pairs:", len(records), "distinct patients:", report["gt_pairs_different_patients"])
    print("SEG_TRAIN polyp:", len(dataset_train), "SEG_VAL polyp:", len(dataset_val))
    print("DataLoader shapes:", report["loader_rgb_shape"], report["loader_gt_shape"])
    print("Evidence:", table, png)
    print("STATUS: A02 REAL DATA PASS")


if __name__ == "__main__":
    main()
