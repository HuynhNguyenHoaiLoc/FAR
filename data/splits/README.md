# Split protocols

## Current: Clean V2

Outer 5-fold:
- `clean_v2/patient_splits_clean_cv5_v2.csv`
- `clean_v2/image_splits_clean_cv5_v2.csv`
- `clean_v2/polyp_image_splits_clean_cv5_v2.csv`

Inner 5-fold:
- `inner_clean_v2_v1/patient_inner_splits_cv5_candidate.csv`
- `inner_clean_v2_v1/image_inner_splits_cv5_candidate.csv`

Despite the `_candidate` filename, the Inner split
protocol was approved by the group lead.

Use the Inner CSV `inner_split` column to separate
SEG_TRAIN, SEG_VAL, META_FIT and CALIBRATION.

## Legacy data

These files are kept for historical reproducibility:

- `../splits.csv`
- `../patient_splits_official_cv5.csv`
- `../official_polyp_image_splits_cv5.csv`

Do not silently mix these legacy files with Clean V2.
Existing consumers require a deliberate migration.
