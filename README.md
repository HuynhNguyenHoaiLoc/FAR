# FAR

---

## Clean V2 - Current Data Protocol

For new experiments, use the approved
**Outer Clean V2 + Inner CV5** split protocol.

- Outer: `data/splits/clean_v2/`
- Inner: `data/splits/inner_clean_v2_v1/`
- PraNet Fold 0: `results/module1/inner_clean_v2_v1/fold0/`
- Handoff guide: `docs/CLEAN_V2_M1_HANDOFF.md`
- Kaggle model and predictions: https://www.kaggle.com/code/eoquiither/xulyanh?scriptVersionId=359730609

The earlier official CSVs and
`results/module1/fold0/` are retained as **legacy**.
Do not mix old and new model/split artifacts.


### Split policy guard and new default loader

New M1 code should use `code/module1/dataset.py` (Clean V2 Inner CV5 by
default) or `code/module1/clean_v2_dataset.py`. Other modules can use
`from far_splits import select_images` to select approved fold and split rows.
The old M1 dataset API is archived in `code/module1/legacy_dataset.py`.

```bash
python code/data_validation/check_clean_v2_policy.py
python -m pytest -q tests/test_clean_v2_split_migration.py tests/test_module1_a02_clean_v2.py
```

See `docs/CLEAN_V2_SPLIT_MIGRATION.md`.  Historical split generators are
explicitly opt-in and no legacy CSVs/model weights were removed.
