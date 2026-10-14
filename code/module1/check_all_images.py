
from pathlib import Path
from PIL import Image

ROOT = Path("Dataset")

stats = {
    "healthy": 0,
    "polyp": 0,
    "masks": 0,
}
errors = []

patients = sorted(
    [
        p for p in ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("Patient")
        and p.name[7:].isdigit()
    ],
    key=lambda p: int(p.name[7:])
)

for patient in patients:
    for category in ["Healthy", "Polyp"]:
        folder = patient / category

        for image_path in sorted(folder.glob("*.jpg")):
            try:
                with Image.open(image_path) as image:
                    image.load()
                    image_size = image.size

                if image_size[0] == 0 or image_size[1] == 0:
                    raise ValueError("Invalid image dimensions")

                stats[category.lower()] += 1

                if category == "Polyp":
                    mask_path = patient / "BinaryMask" / image_path.name

                    with Image.open(mask_path) as mask:
                        mask.load()

                        if mask.size != image_size:
                            raise ValueError("Image-mask size mismatch")

                        gray = mask.convert("L")
                        hist = gray.histogram()

                        foreground = sum(hist[128:])
                        total = gray.width * gray.height

                        if not (0 < foreground < total):
                            raise ValueError("Invalid foreground mask")

                    stats["masks"] += 1

            except Exception as e:
                errors.append(f"{image_path}: {e}")

    patient_number = int(patient.name[7:])

    if patient_number % 50 == 0:
        print(f"Checked through {patient.name}")

print("\n=== FULL DATASET CHECK ===")
print("Healthy images:", stats["healthy"])
print("Polyp images:", stats["polyp"])
print("Valid masks:", stats["masks"])
print("Total original images:", stats["healthy"] + stats["polyp"])
print("Errors:", len(errors))

if errors:
    print("\nError details:")
    for error in errors[:20]:
        print(error)

passed = (
    stats["healthy"] == 7749
    and stats["polyp"] == 432
    and stats["masks"] == 432
    and len(errors) == 0
)

print("\nRESULT:", "PASS" if passed else "FAIL")
