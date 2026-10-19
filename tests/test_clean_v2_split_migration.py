"""Migration regression tests: current Clean V2 defaults, legacy opt-in, CV5."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'code/data_validation'))
sys.path.insert(0, str(ROOT / 'code/module1'))
from check_clean_v2_policy import check_active_code, check_cv5
from far_splits import select_images, INNER_IMAGE_CSV, INNER_PATIENT_CSV
from dataset import FARPolypDataset, DEFAULT_INNER_CSV


def test_current_paths_and_all_five_folds():
    assert DEFAULT_INNER_CSV.resolve() == INNER_IMAGE_CSV.resolve()
    assert INNER_PATIENT_CSV.is_file()
    stats = check_cv5()
    assert len(stats) == 5
    assert all(x['patients'] == 455 and x['polyp'] == 430 for x in stats.values())


def test_current_selector_only_approved_inner_train():
    train = select_images(0, 'SEG_TRAIN', category='Polyp')
    val = select_images(0, 'SEG_VAL', category='Polyp')
    assert len(train) == 215
    assert len(val) == 44
    assert set(train['patient_id']).isdisjoint(set(val['patient_id']))
    assert set(train['split_group_id']).isdisjoint(set(val['split_group_id']))


def test_held_out_requires_opt_in():
    with pytest.raises(PermissionError):
        select_images(0, 'OUTER_TEST')
    with pytest.raises(PermissionError):
        select_images(0, 'NEVER_POLYP_TEST')
    assert len(select_images(0, 'OUTER_TEST', allow_final_test=True)) > 0


def test_legacy_schema_is_rejected():
    with pytest.raises(ValueError, match='Legacy'):
        select_images(0, 'SEG_TRAIN', csv_path=ROOT / 'data/splits.csv')
    with pytest.raises((ValueError, TypeError)):
        FARPolypDataset('historical_manifest.csv', {'Patient1'}, 'SEG_TRAIN')


def test_no_legacy_references_in_active_code():
    assert check_active_code() == []


@pytest.mark.parametrize('filename', [
    'import_official_splits.py', 'build_official_image_splits.py',
])
def test_historical_generators_require_opt_in(filename):
    proc = subprocess.run(
        [sys.executable, str(ROOT / 'code/module1' / filename)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode != 0
    assert 'LEGACY ONLY' in (proc.stdout + proc.stderr)


def test_direct_m1_dataset_held_out_requires_opt_in():
    from clean_v2_dataset import CleanV2SegDataset
    with pytest.raises(PermissionError):
        CleanV2SegDataset('unused_data_root', INNER_IMAGE_CSV, 0, 'OUTER_TEST')


def test_archived_dataset_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv('FAR_ALLOW_LEGACY_SPLITS', raising=False)
    from legacy_dataset import FARPolypDataset as LegacyDataset
    with pytest.raises(RuntimeError, match='LEGACY ONLY'):
        LegacyDataset('unused.csv', ['Patient1'], 'SEG_TRAIN')


def test_module_purpose_policy():
    from far_splits import select_for_module
    assert len(select_for_module(0, 'M1', 'train', category='Polyp')) == 215
    assert len(select_for_module(0, 'M2', 'mine', category='Healthy')) == 1103
    assert len(select_for_module(0, 'M3', 'memory', category='Polyp')) == 57
    assert len(select_for_module(0, 'M4', 'fit', category='Polyp')) == 57
    assert len(select_for_module(0, 'M4', 'calibrate', category='Polyp')) == 27
    with pytest.raises(ValueError, match='Forbidden'):
        select_for_module(0, 'M2', 'mine_outer_test')
    with pytest.raises(PermissionError):
        select_for_module(0, 'M4', 'final_polyp_test')
