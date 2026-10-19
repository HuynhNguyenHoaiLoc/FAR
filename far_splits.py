"""Single-source Clean V2 split paths, policies, and pandas-only selectors.

Import from any repo-root Python process:
    from far_splits import select_images, INNER_IMAGE_CSV
    train = select_images(0, 'SEG_TRAIN', category='Polyp')

The loader never creates or changes splits and refuses final held-out data unless
explicitly requested for a *locked final evaluation*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

ROOT = Path(__file__).resolve().parent
INNER_IMAGE_CSV = ROOT / 'data/splits/inner_clean_v2_v1/image_inner_splits_cv5_candidate.csv'
INNER_PATIENT_CSV = ROOT / 'data/splits/inner_clean_v2_v1/patient_inner_splits_cv5_candidate.csv'
OUTER_IMAGE_CSV = ROOT / 'data/splits/clean_v2/image_splits_clean_cv5_v2.csv'
OUTER_PATIENT_CSV = ROOT / 'data/splits/clean_v2/patient_splits_clean_cv5_v2.csv'
PROTOCOL_VERSION = 'inner_clean_v2_v1'
INNER_TO_OUTER = {
    'SEG_TRAIN': 'TRAIN', 'SEG_VAL': 'VAL',
    'META_FIT': 'TRAIN', 'CALIBRATION': 'TRAIN',
    'OUTER_TEST': 'TEST', 'NEVER_POLYP_TEST': 'NEVER_POLYP_TEST',
}
HELD_OUT = frozenset(('OUTER_TEST', 'NEVER_POLYP_TEST'))
LEGACY_SPLITS = frozenset((
    'splits.csv', 'patient_splits_official_cv5.csv',
    'official_polyp_image_splits_cv5.csv',
))


def select_images(fold: Union[int, str], split: str, *,
                  csv_path: Union[str, Path, None] = None,
                  category: str | None = None,
                  allow_final_test: bool = False):
    """Load *approved Inner CV5 only*, with no random re-splitting.

    M1: SEG_TRAIN/SEG_VAL; M2: SEG_TRAIN for mining/gradient, SEG_VAL for selection;
    M3: META_FIT memory + same-group exclusion in retrieval; M4: META_FIT fit,
    CALIBRATION threshold. Final test is intentionally opt-in.
    """
    import pandas as pd

    if str(fold) not in {'0', '1', '2', '3', '4'}:
        raise ValueError('fold must be 0..4')
    if split not in INNER_TO_OUTER:
        raise ValueError(f'Unknown Clean V2 inner split: {split!r}')
    if split in HELD_OUT and not allow_final_test:
        raise PermissionError(
            f'{split} is held out; set allow_final_test=True only for locked final evaluation'
        )
    path = Path(csv_path) if csv_path is not None else INNER_IMAGE_CSV
    if path.name in LEGACY_SPLITS:
        raise ValueError('Legacy split CSV is forbidden for current experiments')
    if path.resolve() != INNER_IMAGE_CSV.resolve():
        raise ValueError(
            'Only the approved repository Inner CV5 CSV is supported by select_images; '
            'use the canonical data/splits/inner_clean_v2_v1/ file'
        )
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required = {
        'fold', 'image_id', 'patient_id', 'split_group_id', 'category',
        'inner_split', 'outer_split', 'protocol_version', 'sha256',
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Not an Inner CV5 image CSV; missing: {sorted(missing)}')
    rows = frame[frame['fold'] == str(fold)]
    if len(rows) != 8073 or rows['image_id'].nunique() != 8073:
        raise ValueError('Unapproved/partial fold: expected 8073 distinct images')
    if rows['patient_id'].nunique() != 455 or rows['split_group_id'].nunique() != 453:
        raise ValueError('Unexpected Clean V2 patient/group counts')
    if not rows['protocol_version'].eq(PROTOCOL_VERSION).all():
        raise ValueError('Unexpected Clean V2 version')
    if (rows.groupby('patient_id')['inner_split'].nunique() > 1).any():
        raise ValueError('Patient leakage')
    if (rows.groupby('split_group_id')['inner_split'].nunique() > 1).any():
        raise ValueError('Group leakage')
    for label, outer in INNER_TO_OUTER.items():
        block = rows[rows['inner_split'] == label]
        if not block['outer_split'].eq(outer).all():
            raise ValueError(f'Outer partition mismatch: {label}')
    result = rows[rows['inner_split'] == split]
    if category is not None:
        if category not in {'Healthy', 'Polyp'}:
            raise ValueError('category must be Healthy or Polyp')
        result = result[result['category'] == category]
    return result.copy().reset_index(drop=True)


# A role-specific interface for other team modules. Prevent using held-out or
# calibration data for a training/mining/memory operation by accident.
MODULE_PURPOSE_SPLITS = {
    'M1': {
        'train': 'SEG_TRAIN', 'validate': 'SEG_VAL',
        'export_meta_fit': 'META_FIT', 'export_calibration': 'CALIBRATION',
    },
    'M2': {'mine': 'SEG_TRAIN', 'train': 'SEG_TRAIN', 'validate': 'SEG_VAL'},
    'M3': {'memory': 'META_FIT', 'query_meta_fit': 'META_FIT',
           'query_calibration': 'CALIBRATION'},
    'M4': {'fit': 'META_FIT', 'calibrate': 'CALIBRATION',
           'final_polyp_test': 'OUTER_TEST',
           'final_healthy_test': 'NEVER_POLYP_TEST'},
}


def select_for_module(fold, module: str, purpose: str, *,
                      category: str | None = None, allow_final_test: bool = False):
    """Role-aware image selector; no module may silently reassign patients.

    Important: M3 must additionally exclude the query patient and linked group
    *during retrieval*; selecting rows is not sufficient to prevent self-retrieval.
    """
    if module not in MODULE_PURPOSE_SPLITS or purpose not in MODULE_PURPOSE_SPLITS[module]:
        raise ValueError(f'Forbidden/unknown module or purpose: {module}/{purpose}')
    split = MODULE_PURPOSE_SPLITS[module][purpose]
    return select_images(fold, split, category=category, allow_final_test=allow_final_test)
