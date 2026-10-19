"""Audit approved Clean V2 / Inner CV5 mapping and accidental legacy references.

Usage: python code/data_validation/check_clean_v2_policy.py
No actual RGB/mask files are necessary; only repository split CSVs are read.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from far_splits import (ROOT, INNER_IMAGE_CSV, INNER_PATIENT_CSV,
                        PROTOCOL_VERSION, INNER_TO_OUTER)

LEGACY_REFERENCE = re.compile(
    r'(?<![\w.])(?:data[/\\])?(?:splits\.csv|'
    r'patient_splits_official_cv5\.csv|official_polyp_image_splits_cv5\.csv)'
)
# Archived generation/import code is not a consumer of new split CSVs.
EXEMPT = {
    'code/module1/legacy_dataset.py',
    'code/module1/build_official_image_splits.py',
    'code/module1/import_official_splits.py',
    'code/data_validation/check_clean_v2_policy.py',
}


def check_active_code(repo: Path = ROOT) -> list[str]:
    """Static guard over tracked-style source locations; ignore legacy utilities."""
    problems = []
    for folder in ('code', 'src'):
        for path in sorted((repo / folder).rglob('*.py')):
            rel = path.relative_to(repo).as_posix()
            if rel in EXEMPT:
                continue
            for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if LEGACY_REFERENCE.search(line):
                    problems.append(f'{rel}:{line_no}: legacy split filename')
    return problems


def check_cv5(repo: Path = ROOT) -> dict:
    images_file = repo / INNER_IMAGE_CSV.relative_to(ROOT)
    patients_file = repo / INNER_PATIENT_CSV.relative_to(ROOT)
    images = pd.read_csv(images_file, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    patients = pd.read_csv(patients_file, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    assert len(images) == 40365 and len(patients) == 2275
    assert images['protocol_version'].eq(PROTOCOL_VERSION).all()
    assert patients['protocol_version'].eq(PROTOCOL_VERSION).all()
    assert not images.duplicated(['fold', 'image_id']).any()
    assert not patients.duplicated(['fold', 'patient_id']).any()
    assert set(images['fold']) == set(patients['fold']) == set('01234')
    merged = images.merge(
        patients[['fold', 'patient_id', 'split_group_id', 'inner_split']],
        on=['fold', 'patient_id'], how='left', validate='many_to_one',
        suffixes=('_img', '_patient'), indicator=True,
    )
    assert merged['_merge'].eq('both').all(), 'Missing patient assignments'
    assert merged['inner_split_img'].eq(merged['inner_split_patient']).all()
    assert merged['split_group_id_img'].eq(merged['split_group_id_patient']).all()
    summary = {}
    for fold in range(5):
        rows = images[images['fold'] == str(fold)]
        pat = patients[patients['fold'] == str(fold)]
        assert len(rows) == 8073 and len(pat) == 455
        assert rows['image_id'].nunique() == 8073
        assert pat['patient_id'].nunique() == 455
        assert rows['split_group_id'].nunique() == 453
        assert rows['category'].eq('Polyp').sum() == 430
        assert rows['category'].eq('Healthy').sum() == 7643
        assert not rows.loc[rows['sha256'].ne(''), 'sha256'].duplicated().any()
        assert rows.groupby('patient_id')['inner_split'].nunique().max() == 1
        assert rows.groupby('split_group_id')['inner_split'].nunique().max() == 1
        for split, outer in INNER_TO_OUTER.items():
            part = rows[rows['inner_split'] == split]
            assert not part.empty
            assert part['outer_split'].eq(outer).all(), f'{fold}: {split}'
        summary[str(fold)] = dict(images=len(rows), patients=len(pat),
                                  groups=453, polyp=430, healthy=7643)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=ROOT)
    args = ap.parse_args()
    bad = check_active_code(args.repo)
    if bad:
        raise SystemExit('Legacy references in active source:\n' + '\n'.join(bad))
    counts = check_cv5(args.repo)
    for fold, values in counts.items():
        print('Fold', fold, values)
    print('SPLIT POLICY: PASS (5 folds, no active legacy split references)')


if __name__ == '__main__':
    main()
