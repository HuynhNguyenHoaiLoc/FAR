"""Default FAR Module 1 dataset: approved Clean V2 / Inner CV5.

The historical manifest/patient-list implementation is now in
``code/module1/legacy_dataset.py`` and requires an explicit legacy opt-in.
New training must not manually assign patients or silently use an old CSV.
"""
from __future__ import annotations

from pathlib import Path

try:
    from .clean_v2_dataset import CleanV2SegDataset, make_loader, load_fold_index
except ImportError:  # Import as a standalone module from code/module1/
    from clean_v2_dataset import CleanV2SegDataset, make_loader, load_fold_index

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INNER_CSV = ROOT / "data/splits/inner_clean_v2_v1/image_inner_splits_cv5_candidate.csv"


class FARPolypDataset(CleanV2SegDataset):
    """M1 dataset with the approved Inner CV5 split as the only default.

    Example: FARPolypDataset(clean_dataset_root, fold=0, split="SEG_TRAIN")
    The previous (manifest_path, patient_ids, split) API is intentionally invalid.
    """

    def __init__(self, data_root, fold=0, split="SEG_TRAIN", *,
                 csv_path=None, size=256, include_healthy=False,
                 augment=None, return_metadata=False,
                 allow_final_test=False):
        if isinstance(fold, (set, list, tuple, dict)):
            raise ValueError(
                "Historical patient_ids API is forbidden for Clean V2. "
                "Use FARPolypDataset(data_root, fold=0, split='SEG_TRAIN'). "
                "Archived implementation: legacy_dataset.py."
            )
        if str(fold) not in {"0", "1", "2", "3", "4"}:
            raise ValueError("fold must be an integer 0..4")
        if split in {"OUTER_TEST", "NEVER_POLYP_TEST"} and not allow_final_test:
            raise PermissionError(
                f"{split} is held out. allow_final_test=True is for a locked final evaluation only."
            )
        super().__init__(
            data_root, csv_path if csv_path is not None else DEFAULT_INNER_CSV,
            fold, split, size=size, include_healthy=include_healthy,
            augment=augment, return_metadata=return_metadata,
            strict_clean_v2=True, allow_final_test=allow_final_test,
        )


__all__ = ["FARPolypDataset", "CleanV2SegDataset", "make_loader", "load_fold_index", "DEFAULT_INNER_CSV"]
