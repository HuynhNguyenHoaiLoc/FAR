"""Evaluation metrics package for FAR-Polyp-Seg (Module 4)."""

from .metrics import (
    MIN_COMPONENT_AREA,
    binarize_mask,
    compute_confusion_counts,
    compute_mask_metrics,
    dice_score,
    evaluate_dataset,
    extract_connected_components,
    filter_small_components,
    has_positive_prediction,
    image_level_fpr,
    image_level_recall,
    iou_score,
    pixel_fpr_score,
    precision_score,
    recall_score,
    run_hand_calculated_tests,
)

__all__ = [
    "MIN_COMPONENT_AREA",
    "binarize_mask",
    "compute_confusion_counts",
    "compute_mask_metrics",
    "dice_score",
    "evaluate_dataset",
    "extract_connected_components",
    "filter_small_components",
    "has_positive_prediction",
    "image_level_fpr",
    "image_level_recall",
    "iou_score",
    "pixel_fpr_score",
    "precision_score",
    "recall_score",
    "run_hand_calculated_tests",
]
