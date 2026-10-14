
from pathlib import Path
from datetime import datetime
from PIL import Image
import pandas as pd
import json

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "dataset_manifest.csv"
OUTPUT = ROOT / "data" / "data_check.json"

df = pd.read_csv(MANIFEST)
errors = []

counts = {
    "patients": int(df["patient_id"].nunique()),
    "total_images": int(len(df)),
    "polyp_images": int((df["is_polyp"] == 1).sum()),
    "healthy_images": int((df["is_polyp"] == 0).sum()),
    "duplicate_image_ids": int(df["image_id"].duplicated().sum()),
    "patients_with_polyp": 0,
    "never_polyp_patients": 0,
    "checked_images": 0,
    "checked_masks": 0,
}

patient_stats = df.groupby("patient_id")["is_polyp"].sum()
counts["patients_with_polyp"] = int((patient_stats > 0).sum())
counts["never_polyp_patients"] = int((patient_stats == 0).sum())

print("Checking images and masks...")

for row in df.itertuples(index=False):
    image_path = ROOT / row.original_path

    try:
        with Image.open(image_path) as img:
            img.load()
            image_size = img.size

            if img.mode != "RGB":
                raise ValueError(f"Image mode: {img.mode}")

            counts["checked_images"] += 1

        if row.is_polyp == 1:
            if pd.isna(row.mask_path):
                raise ValueError("Missing mask path")

            mask_path = ROOT / row.mask_path

            with Image.open(mask_path) as mask:
                mask.load()

                if mask.size != image_size:
                    raise ValueError("Image-mask size mismatch")

                gray = mask.convert("L")
                histogram = gray.histogram()

                foreground = sum(histogram[128:])
                total = gray.width * gray.height

                if not (0 < foreground < total):
                    raise ValueError("Invalid mask foreground")

                counts["checked_masks"] += 1

        elif not pd.isna(row.mask_path):
            raise ValueError("Healthy image has mask path")

    except Exception as e:
        errors.append({
            "image_id": row.image_id,
            "error": str(e)
        })

expected = {
    "patients": 455,
    "total_images": 8181,
    "polyp_images": 432,
    "healthy_images": 7749,
    "duplicate_image_ids": 0,
    "patients_with_polyp": 187,
    "never_polyp_patients": 268,
    "checked_images": 8181,
    "checked_masks": 432,
}

passed = (
    all(counts[k] == v for k, v in expected.items())
    and len(errors) == 0
)

report = {
    "dataset": "FAR-POLYP-SEG",
    "checked_at": datetime.now().astimezone().isoformat(),
    "manifest": "data/dataset_manifest.csv",
    "status": "PASS" if passed else "FAIL",
    "counts": counts,
    "mask_validation": {
        "source_format": "JPEG",
        "grayscale_threshold": 128,
        "binary_values_after_normalization": [0, 255],
        "source_files_modified": False
    },
    "error_count": len(errors),
    "errors": errors
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n=== DATA CHECK REPORT ===")
print("Images checked:", counts["checked_images"])
print("Masks checked:", counts["checked_masks"])
print("Never-polyp patients:", counts["never_polyp_patients"])
print("Errors:", len(errors))
print("Saved:", OUTPUT)
print("RESULT:", report["status"])
