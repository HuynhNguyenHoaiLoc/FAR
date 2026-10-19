# Split path migration audit — commit 3f59d94 source snapshot

## Scope and findings

- **Migrated:** default `code/module1/dataset.py` now uses the approved Inner
  CV5 image split. New imports should no longer partition arbitrary patient IDs.
  The old implementation is preserved in `code/module1/legacy_dataset.py`.
- **Protected:** `code/module1/import_official_splits.py`,
  `code/module1/build_official_image_splits.py`, and
  `code/module1/create_patient_cohorts.py` are historical *generators*, not
  current consumers. They require `--legacy-reproduce` before execution;
  the old dataset requires `FAR_ALLOW_LEGACY_SPLITS=1`.
- **Already correct:** `code/module1/clean_v2_dataset.py`,
  `code/module1/a02_verify.py` and the approved Outer/Inner CSVs. The loader
  additionally rejects an incorrect `protocol_version` and requires an explicit
  `allow_final_test=True` for held-out data.
- **Updated documentation:** `data/split_notes.md` no longer recommends
  legacy `data/splits.csv` for the current Fold 0.
- **Preserved:** all source CSVs and model artifacts under `results/module1/fold0/`.
  Do not present legacy metrics/checkpoints as Clean V2 results.

## M2 / M3 / M4 integration boundary

At this source snapshot, **13 M2 files under `src/module2/` are empty (0 bytes)**
and no M3/M4 executable implementations are present. Therefore it would be
incorrect to claim their runtimes are migrated or tested. Do not fill another
member's files without coordinating ownership. They should import the approved
selector instead of hardcoding paths:

```python
from far_splits import select_for_module

# M2: only SEG_TRAIN for mining and gradient; SEG_VAL for model choice.
m2_train = select_for_module(0, 'M2', 'mine')
m2_val = select_for_module(0, 'M2', 'validate')

# M3: approved META_FIT reference pool, plus mandatory runtime group exclusion.
m3_memory = select_for_module(0, 'M3', 'memory')

# M4: fit on unseen-backbone predictions from META_FIT;
# CALIBRATION is separate and never used to fit the reranker.
m4_fit = select_for_module(0, 'M4', 'fit')
m4_calibration = select_for_module(0, 'M4', 'calibrate')

# Final test is intentionally restricted; never use it for tuning:
# final_test = select_for_module(0, 'M4', 'final_polyp_test', allow_final_test=True)
```

`select_images()` and `select_for_module()` check patient/group partition
integrity but **do not
implement** M3's per-query same-group memory exclusion. That is a retrieval
runtime invariant to test when M3 code exists. New M2/M3/M4 training and
retrieval consumers remain a separate integration/review task.

## Validate and publish

```shell
python code/data_validation/check_clean_v2_policy.py
python -m pytest -q tests/test_module1_a02_clean_v2.py tests/test_clean_v2_split_migration.py
```

Outputs and old split CSVs are unchanged. Review the patch, run both commands,
commit to `feature/clean-v2-inner-m1-handoff`, and push to update PR #7.
