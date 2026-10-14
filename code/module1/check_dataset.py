
from pathlib import Path
import pandas as pd

ROOT = Path("Dataset")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def get_files(folder):
    if not folder.is_dir():
        return set()
    return {
        f.name for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    }

patients = sorted(
    p for p in ROOT.iterdir()
    if p.is_dir() and p.name.startswith("Patient")
)

summary = pd.read_csv(ROOT / "patient_summary.csv")

counts = {
    "Healthy": 0,
    "Polyp": 0,
    "Annotated": 0,
    "BinaryMask": 0
}

missing_masks = 0
extra_masks = 0
mismatched_patients = []

for patient in patients:
    files = {
        name: get_files(patient / name)
        for name in counts
    }

    for name in counts:
        counts[name] += len(files[name])

    missing_masks += len(files["Polyp"] - files["BinaryMask"])
    extra_masks += len(files["BinaryMask"] - files["Polyp"])

    info = summary[summary["Patient ID"] == patient.name]

    if len(info) != 1:
        mismatched_patients.append(patient.name)
        continue

    row = info.iloc[0]

    if (
        len(files["Healthy"]) != row["Healthy Images"]
        or len(files["Polyp"]) != row["Polyp Images"]
    ):
        mismatched_patients.append(patient.name)

print("=== FAR-POLYP-SEG DATA CHECK ===")
print("Patient folders:", len(patients))
print("CSV patient rows:", len(summary))

for name, count in counts.items():
    print(f"{name} images: {count}")

print("Total original images:", counts["Healthy"] + counts["Polyp"])
print("Missing masks:", missing_masks)
print("Extra masks:", extra_masks)
print("Patients with count mismatch:", len(mismatched_patients))
