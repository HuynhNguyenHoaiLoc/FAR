
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset"
OUTPUT = ROOT / "data"

OUTPUT.mkdir(exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png"}
records = []

patients = sorted(
    [
        p for p in DATASET.iterdir()
        if p.is_dir()
        and p.name.startswith("Patient")
        and p.name[7:].isdigit()
    ],
    key=lambda p: int(p.name[7:])
)

for patient in patients:
    for category, is_polyp in [("Healthy", 0), ("Polyp", 1)]:
        folder = patient / category

        for image_path in sorted(folder.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in EXTENSIONS:
                continue

            # ID duy nhất theo bệnh nhân, loại ảnh và tên file
            image_id = f"{patient.name}__{category}__{image_path.stem}"

            mask_path = ""

            if is_polyp == 1:
                mask_file = patient / "BinaryMask" / image_path.name

                if not mask_file.is_file():
                    raise FileNotFoundError(mask_file)

                mask_path = mask_file.relative_to(ROOT).as_posix()

            records.append({
                "image_id": image_id,
                "patient_id": patient.name,
                "is_polyp": is_polyp,
                "original_path": image_path.relative_to(ROOT).as_posix(),
                "mask_path": mask_path
            })

df = pd.DataFrame(records)

# Kiểm tra tính nhất quán
assert df["image_id"].is_unique, "Duplicate image_id!"
assert len(df) == 8181, "Unexpected image count!"
assert df["is_polyp"].sum() == 432, "Unexpected polyp count!"
assert df["patient_id"].nunique() == 455, "Unexpected patient count!"

output_file = OUTPUT / "dataset_manifest.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("=== DATASET MANIFEST ===")
print("Total images:", len(df))
print("Polyp images:", int(df["is_polyp"].sum()))
print("Healthy images:", int((df["is_polyp"] == 0).sum()))
print("Unique patients:", df["patient_id"].nunique())
print("Duplicate image IDs:", df["image_id"].duplicated().sum())
print("Saved:", output_file)
print("RESULT: PASS")
