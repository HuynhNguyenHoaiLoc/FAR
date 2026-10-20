"""A03 audit regressions: never manufacture AMP or checkpoint proof."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/module1'))
from a03_verify import audit, read_history, digest, DEFAULT_RUN, DEFAULT_SPLITS, split_audit


def fixture(tmp_path, *, amp=False, val_loss=True, checkpoint=True):
    run = tmp_path / 'run'
    run.mkdir()
    m = json.loads((DEFAULT_RUN / 'run_metadata.json').read_text(encoding='utf-8'))
    fake_checkpoint = tmp_path / 'best_model.pth'
    fake_checkpoint.write_bytes(b'fake-test-checkpoint-only')
    m['best_checkpoint_sha256'] = digest(fake_checkpoint)
    m['amp_enabled'] = amp
    (run / 'run_metadata.json').write_text(json.dumps(m), encoding='utf-8')
    header = ['epoch', 'train_loss', 'val_dice', 'amp_enabled', 'grad_scaler_enabled']
    if val_loss:
        header.append('val_loss')
    rows = []
    for e in range(1, 102):
        record = {'epoch': e, 'train_loss': 2.0 - e / 500,
                  'val_dice': 0.748697476988594 if e == 71 else 0.5,
                  'amp_enabled': amp, 'grad_scaler_enabled': amp}
        if val_loss:
            record['val_loss'] = 1.5 - e / 1000
        rows.append(record)
    history = run / 'history.csv'
    with history.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return run, history, fake_checkpoint if checkpoint else None


def test_official_split_hash_and_no_leakage():
    metadata = json.loads((DEFAULT_RUN / 'run_metadata.json').read_text())
    checked = split_audit(DEFAULT_SPLITS, metadata)
    assert checked['fold_patients'] == 455
    assert checked['fold_images'] == 8073
    assert checked['linked_group_leakage'] == 0
    assert checked['train_polyp'] == 215
    assert checked['validation_polyp'] == 44


def test_real_historical_metadata_never_claims_amp():
    original = json.loads((DEFAULT_RUN / 'run_metadata.json').read_text())
    assert original.get('amp_enabled') is not True
    assert original['best_epoch'] == 71
    assert original['best_val_dice'] == pytest.approx(0.748697476988594)


def test_history_loss_and_dice(tmp_path):
    _, h, _ = fixture(tmp_path, amp=True)
    records, summary = read_history(h)
    assert len(records) == 101
    assert summary['train_loss_logged']
    assert summary['validation_loss_logged']
    assert summary['validation_dice_logged']


def test_amp_unknown_is_pending_even_if_checkpoint_matches(tmp_path, monkeypatch):
    from a03_verify import strict_checkpoint_reload
    import a03_verify
    monkeypatch.setattr(a03_verify, 'strict_checkpoint_reload', lambda *args: {'strict_load': True})
    run, h, ckpt = fixture(tmp_path, amp=False)
    evidence = audit(run=run, splits=DEFAULT_SPLITS, checkpoint=ckpt, history_path=h, upstream=tmp_path)
    assert evidence['status'] == 'PENDING'
    assert evidence['amp']['verified'] is False
    assert 'executed_amp_training_provenance' in evidence['missing_evidence']


def test_amp_proof_and_strict_reloading_can_pass(tmp_path, monkeypatch):
    import a03_verify
    monkeypatch.setattr(a03_verify, 'strict_checkpoint_reload', lambda *args: {'strict_load': True})
    run, h, ckpt = fixture(tmp_path, amp=True)
    evidence = audit(run=run, splits=DEFAULT_SPLITS, checkpoint=ckpt, history_path=h, upstream=tmp_path)
    assert evidence['status'] == 'PASS'
    assert evidence['amp']['verified']
    assert evidence['checkpoint']['strict_load']


def test_missing_val_loss_prevents_pass(tmp_path, monkeypatch):
    import a03_verify
    monkeypatch.setattr(a03_verify, 'strict_checkpoint_reload', lambda *args: {'strict_load': True})
    run, h, ckpt = fixture(tmp_path, amp=True, val_loss=False)
    evidence = audit(run=run, splits=DEFAULT_SPLITS, checkpoint=ckpt, history_path=h, upstream=tmp_path)
    assert evidence['status'] == 'PENDING'
    assert 'val_loss_per_epoch' in evidence['missing_evidence']


def test_wrong_checkpoint_hash_hard_fails(tmp_path):
    run, h, ckpt = fixture(tmp_path, amp=True)
    ckpt.write_bytes(b'corrupted-checkpoint')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        audit(run=run, splits=DEFAULT_SPLITS, checkpoint=ckpt, history_path=h, upstream=None)


def test_wrong_split_checksum_hard_fails(tmp_path):
    run, h, ckpt = fixture(tmp_path)
    m = json.loads((run / 'run_metadata.json').read_text())
    m['split_csv_sha256'] = '0' * 64
    (run / 'run_metadata.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='Split checksum'):
        audit(run=run, splits=DEFAULT_SPLITS, checkpoint=ckpt, history_path=h, upstream=None)


def test_no_checkpoint_cannot_pass(tmp_path):
    run, h, _ = fixture(tmp_path, amp=True)
    evidence = audit(run=run, splits=DEFAULT_SPLITS, checkpoint=None, history_path=h, upstream=None)
    assert evidence['status'] == 'PENDING'
    assert 'actual_checkpoint_file' in evidence['missing_evidence']


def test_train_script_requires_explicit_amp_calls():
    text = (ROOT/'code/module1/a03_train_amp.py').read_text(encoding='utf-8')
    assert 'torch.autocast(device_type="cuda", dtype=torch.float16)' in text
    assert 'torch.amp.GradScaler("cuda", enabled=True)' in text
    assert 'scaler.scale(loss).backward()' in text
    assert 'scaler.step(optimizer)' in text
    assert 'scaler.update()' in text
    assert '"SEG_TRAIN"' in text and '"SEG_VAL"' in text
