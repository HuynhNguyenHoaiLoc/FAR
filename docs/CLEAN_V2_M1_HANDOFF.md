# Clean V2 / Inner CV5 / M1 Fold 0 - Handoff

## Current protocol

The group-approved data split protocols are:

- Outer: `data/splits/clean_v2/`
- Inner: `data/splits/inner_clean_v2_v1/`

The old official CSVs are historical references.
Do not use them for new Clean V2 experiments.

## Data

- 455 patient IDs across 453 linked patient groups
- 8073 clean RGB images per fold
- 7643 Healthy / 430 Polyp
- 5 outer folds and 5 inner partitions
- Patient366 is linked with Patient386
- Patient10 has no retained Polyp image after QC

## PraNet M1 Inner Fold 0

- Training: 215 SEG_TRAIN Polyp images
- Validation: 44 SEG_VAL Polyp images
- Best epoch: 71
- Best Validation Dice: 0.74869748
- Trained without loading the legacy checkpoint
- 2717 inference records across SEG_TRAIN, SEG_VAL,
  META_FIT, and CALIBRATION

## Kaggle checkpoint and actual predictions

https://www.kaggle.com/code/eoquiither/xulyanh?scriptVersionId=359730609

Kaggle output folders:

- `M1_CLEAN_V2_INNER_FOLD0/`
- `M1_INNER_V2_FOLD0_PREDICTIONS/`

The Kaggle output contains:

- `best_model.pth`
- `predictions/predictions/probability/` (2717 NPY)
- `predictions/predictions/binary_mask/` (2717 PNG)

Note: the paths above refer to the Kaggle prediction
output folder. The internal relative paths are also
recorded in `predictions.csv`.

Checkpoint SHA-256:

`8c75a4fb5675d0dc7985704dcbf2a44e2545f603645dec31ff38c61e9f91561a`

Kaggle access and file visibility must be confirmed
for each teammate before using the artifacts.

## For Module 2

Use predictions for SEG_TRAIN as defined by the
approved pipeline. Do not tune against held-out tests.

## For Module 3 / Module 4

META_FIT and CALIBRATION predictions are available.
Do not use CALIBRATION to fit the M3/M4 models.
Use CALIBRATION only for the approved calibration steps.

When retrieving examples, Module 3 must exclude
matches from the same linked patient group.

## Important restrictions

- OUTER_TEST has not been used.
- NEVER_POLYP_TEST has not been used.
- Fold 1-4 M1 training is not included in this handoff.
- The previous `results/module1/fold0/` represents
  an older experiment and must not be mixed with
  Clean V2 Inner Fold 0.

## Validation

- Outer / Inner split mapping: PASS
- Patient group leakage audit: PASS
- Checkpoint provenance: PASS
- 2717 output files of each type were found
- 40 probability/mask pairs were value-checked
- Final test evaluation: NOT PERFORMED
