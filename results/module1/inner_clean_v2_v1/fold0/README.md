# Module 1 - PraNet Inner Clean V2 - Fold 0

## Status
Clean V2 and Inner CV5 split protocols were approved by the group lead.
This new M1 Fold 0 training and inference output still requires review.
Candidate filenames are retained for traceability.

## Training
- Model: PraNet, matching FAR-POLYP-SEG upstream architecture
- Split protocol: inner_clean_v2_v1
- SEG_TRAIN: 215 Polyp images
- SEG_VAL: 44 Polyp images
- Best epoch: 71
- Best Validation Dice: 0.74869748
- Old checkpoint was NOT used for initialization.

## Checkpoint
SHA-256:
`8c75a4fb5675d0dc7985704dcbf2a44e2545f603645dec31ff38c61e9f91561a`

Checkpoint stored in Kaggle Notebook Output:
https://www.kaggle.com/code/eoquiither/xulyanh?scriptVersionId=359730609

This repository does not contain the new model weights.
Access to Kaggle output may require permission.

## Predictions
- 2717 images in SEG_TRAIN, SEG_VAL, META_FIT, CALIBRATION
- float32 probability maps (.npy), native resolution
- uint8 binary masks (.png), native resolution
- Predictions manifest: predictions.csv

The actual NPY/PNG files remain in Kaggle Notebook Output.
Paths in predictions.csv are relative to the Kaggle
prediction output directory, not this Git repository.

## Restrictions
- META_FIT and CALIBRATION were not used to train PraNet.
- OUTER_TEST and NEVER_POLYP_TEST were not used.
- Fixed binary masks use threshold 0.5; final calibration
  may select another decision threshold.
- Module 3 must implement same-group retrieval exclusion.

## Legacy results
results/module1/fold0/ is the earlier grouped-CV5 experiment.
Its old checkpoint, metrics and predictions must NOT be mixed
with these Clean V2 Inner Fold 0 artifacts.

## Not completed
- Fold 1-4 training
- Final outer-test evaluation
- End-to-end validation of M2-M4
