# FAR Clean V2 - Outer CV5

This is the Clean V2 outer partition reconstructed
from the group-reviewed Inner CV5 input mappings.

- 455 patient IDs and 453 linked groups per fold
- 8073 RGB images per fold (430 Polyp, 7643 Healthy)
- Original outer split structure preserved
- Patient366 follows linked Patient386 in all five folds
- Patient10 retains POLYP_COHORT history

Use these Clean V2 paths for new experiments.
Do not use legacy official CSVs for Clean V2 training.

Historical reference files in data/ remain unchanged:
- data/patient_splits_official_cv5.csv
- data/official_polyp_image_splits_cv5.csv
- data/splits.csv

The actual cleaned images are stored in the
private Kaggle Clean V2 dataset, not in this repository.

Inner split version: inner_clean_v2_v1.
