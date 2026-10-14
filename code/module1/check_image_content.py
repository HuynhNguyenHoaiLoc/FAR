
from pathlib import Path
from PIL import Image

ROOT = Path("Dataset")
checked = 0
errors = 0

print("=== CHECK 10 IMAGE-MASK PAIRS ===")

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
    polyp_dir = patient / "Polyp"
    mask_dir = patient / "BinaryMask"

    for image_path in sorted(polyp_dir.glob("*.jpg")):
        mask_path = mask_dir / image_path.name

        try:
            with Image.open(image_path) as img, Image.open(mask_path) as mask:
                img.load()
                mask.load()

                gray = mask.convert("L")
                hist = gray.histogram()

                foreground = sum(hist[128:])
                total = gray.width * gray.height
                unique_values = sum(v > 0 for v in hist)

                valid = (
                    img.size == mask.size
                    and 0 < foreground < total
                )

                print(f"\n[{checked + 1}] {patient.name}")
                print(f"Image: {image_path.name}")
                print(f"Image mode: {img.mode}")
                print(f"Image size: {img.size}")
                print(f"Mask size: {mask.size}")
                print(f"Mask unique values: {unique_values}")
                print(f"Polyp pixels (>=128): {foreground}")
                print(f"Status: {'PASS' if valid else 'FAIL'}")

                if not valid:
                    errors += 1

        except Exception as e:
            errors += 1
            print(f"ERROR: {image_path.name}: {e}")

        checked += 1

        if checked >= 10:
            break

    if checked >= 10:
        break

print("\n=== SUMMARY ===")
print("Pairs checked:", checked)
print("Errors:", errors)
print("Result:", "PASS" if checked == 10 and errors == 0 else "FAIL")
