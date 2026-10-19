"""LEGACY ONLY — historical uncleaned 432-image / patient splitting artifact.

Do not use as a source for Clean V2 or Inner CV5.  To reproduce a prior
experiment deliberately, pass --legacy-reproduce.  This program may write
historical output files; it does not modify approved Clean V2 CSVs.
"""
import sys
if "--legacy-reproduce" not in sys.argv:
    raise SystemExit(
        "LEGACY ONLY: this script creates historical data artifacts. "
        "Use --legacy-reproduce only for an intentional legacy reproduction. "
        "New experiments must use data/splits/inner_clean_v2_v1/."
    )


from pathlib import Path
import pandas as pd

# Xác định thư mục gốc của repository FAR
ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "dataset_manifest.csv"
OUTPUT = ROOT / "data" / "patient_cohorts.csv"

# Đọc manifest đã hoàn thành ở A01
df = pd.read_csv(INPUT)

# Kiểm tra dữ liệu đầu vào
assert len(df) == 8181
assert df["image_id"].is_unique
assert df["patient_id"].nunique() == 455
assert set(df["is_polyp"].unique()) == {0, 1}

# Thống kê theo từng bệnh nhân
patients = (
    df.groupby("patient_id")
    .agg(
        total_images=("image_id", "count"),
        polyp_images=("is_polyp", "sum")
    )
    .reset_index()
)

patients["healthy_images"] = (
    patients["total_images"] - patients["polyp_images"]
)

patients["has_polyp"] = (
    patients["polyp_images"] > 0
).astype(int)

# Chia thành 2 nhóm theo yêu cầu nhóm trưởng
patients["cohort"] = patients["has_polyp"].map({
    1: "POLYP_COHORT",
    0: "NEVER_POLYP_COHORT"
})

# Kiểm tra số liệu
positive = patients[patients["has_polyp"] == 1]
never_polyp = patients[patients["has_polyp"] == 0]

assert len(positive) == 187
assert len(never_polyp) == 268
assert positive["polyp_images"].sum() == 432
assert positive["healthy_images"].sum() == 2947
assert never_polyp["healthy_images"].sum() == 4802

# Xuất file
patients = patients[
    [
        "patient_id",
        "cohort",
        "has_polyp",
        "polyp_images",
        "healthy_images",
        "total_images"
    ]
]

patients.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("=== PATIENT COHORT SPLIT ===")
print("Total patients:", len(patients))
print("Polyp cohort:", len(positive))
print("Never-polyp cohort:", len(never_polyp))
print("Polyp images:", int(positive["polyp_images"].sum()))
print("Healthy images in polyp cohort:",
      int(positive["healthy_images"].sum()))
print("Healthy images in never-polyp cohort:",
      int(never_polyp["healthy_images"].sum()))
print("Saved:", OUTPUT)
print("RESULT: PASS")
