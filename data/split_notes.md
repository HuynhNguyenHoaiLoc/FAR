# FAR patient split usage — current vs historical

## CURRENT — use this for new experiments

The team-approved data policy is **Clean V2 Outer CV5 + Inner CV5**:

- Patient-level outer split: `data/splits/clean_v2/patient_splits_clean_cv5_v2.csv`
- Image-level outer split: `data/splits/clean_v2/image_splits_clean_cv5_v2.csv`
- Patient-level inner split: `data/splits/inner_clean_v2_v1/patient_inner_splits_cv5_candidate.csv`
- **Image-level inner split** (training/inference authority): `data/splits/inner_clean_v2_v1/image_inner_splits_cv5_candidate.csv`

Despite the filename `_candidate`, the group lead accepted this Inner CV5
assignment. Some generated audit files still carry a historical pending string;
do not regenerate split assignments or silently edit provenance.

Select by both `fold` (0–4) and `inner_split`:

| Module | Purpose | Inner split |
|---|---|---|
| M1 | PraNet training | SEG_TRAIN (Polyp) |
| M1 | checkpoint selection | SEG_VAL (Polyp) |
| M2 | mining, fine-tuning | SEG_TRAIN only |
| M2 | checkpoint selection | SEG_VAL |
| M3 | memory building | META_FIT; exclude same patient/group at query time |
| M4 | fit reranker | META_FIT, using unseen M1/M2 predictions |
| M4 | choose threshold | CALIBRATION |
| All | locked final evaluation only | OUTER_TEST / NEVER_POLYP_TEST |

Do not infer patients from individual images or reshuffle them.  DataLoaders
may shuffle *within* SEG_TRAIN, but this must never change assignments.

Use `from far_splits import select_images` or M1 `FARPolypDataset`.
See `docs/CLEAN_V2_SPLIT_MIGRATION.md` and `docs/A02_CLEAN_V2_HANDOFF.md`.

## LEGACY — reproducibility only

- `data/splits.csv`
- `data/patient_splits_official_cv5.csv`
- `data/official_polyp_image_splits_cv5.csv`
- `code/module1/import_official_splits.py`
- `code/module1/build_official_image_splits.py`
- `code/module1/create_patient_cohorts.py`
- `code/module1/legacy_dataset.py`
- `results/module1/fold0/`

Those files describe an older 432-Polyp / 301-train Fold 0 protocol.
They are retained intentionally, but are **not valid defaults** for Clean V2.
Legacy split generator scripts require `--legacy-reproduce`; the old dataset
requires the environment variable `FAR_ALLOW_LEGACY_SPLITS=1`.
