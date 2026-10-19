#!/usr/bin/env python3
"""Evaluation metrics for FAR-Polyp-Seg (Module 4: M4 Metrics).

Provides pixel-level and image/component-level evaluation metrics required by
Module 1 (Baseline evaluation), Module 2 (Hard-negative learning comparison),
Module 3 (Memory retrieval ablation), and Module 4 (Reranker & final evaluation):

1. Pixel-level metrics:
   - Dice coefficient (F1)
   - IoU (Intersection over Union / Jaccard)
   - Recall (Sensitivity)
   - Precision
   - Pixel-level FPR (False Positive Rate)
2. Connected-component rule (>= 100 pixels):
   - Extracts connected components and filters out noise blobs < 100 px.
   - Determines image-level False Positive alarms on Normal/Healthy images
     (an image is flagged as FP iff it contains >= 1 connected component with
     area >= 100 px).
   - Computes dataset-level Polyp Dice/IoU/Recall and Normal FPR, and categorizes
     images into TP / FP / TN / FN for error visualization.

Note: Supports NumPy arrays and PyTorch tensors out-of-the-box, and includes a
zero-dependency pure-Python 2D grid fallback so self-tests can also run in bare
Python environments where NumPy is not installed.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

try:
    from scipy import ndimage as _ndimage  # type: ignore
except ImportError:
    _ndimage = None

MIN_COMPONENT_AREA: int = 100
DEFAULT_THRESHOLD: float = 0.5


class _Grid2D:
    """Lightweight 2D array fallback when NumPy is not installed."""

    def __init__(self, data: List[List[Any]], dtype: type = int) -> None:
        self.dtype = dtype
        self.data: List[List[Any]] = [
            [dtype(v) for v in row] for row in data
        ]
        self.height = len(self.data)
        self.width = len(self.data[0]) if self.height > 0 else 0
        self.shape: Tuple[int, int] = (self.height, self.width)
        self.ndim: int = 2
        self.size: int = self.height * self.width

    @classmethod
    def zeros(cls, shape: Tuple[int, int], dtype: type = int) -> "_Grid2D":
        h, w = shape
        fill = dtype(0)
        return cls([[fill for _ in range(w)] for _ in range(h)], dtype=dtype)

    def copy(self) -> "_Grid2D":
        return _Grid2D(self.data, dtype=self.dtype)

    def __getitem__(self, key: Tuple[Any, Any]) -> Any:
        r_key, c_key = key
        if isinstance(r_key, int) and isinstance(c_key, int):
            return self.data[r_key][c_key]
        r_idx = range(*r_key.indices(self.height)) if isinstance(r_key, slice) else [r_key]
        c_idx = range(*c_key.indices(self.width)) if isinstance(c_key, slice) else [c_key]
        return _Grid2D(
            [[self.data[r][c] for c in c_idx] for r in r_idx],
            dtype=self.dtype,
        )

    def __setitem__(self, key: Tuple[Any, Any], value: Any) -> None:
        r_key, c_key = key
        if isinstance(r_key, int) and isinstance(c_key, int):
            self.data[r_key][c_key] = self.dtype(value)
            return
        r_idx = range(*r_key.indices(self.height)) if isinstance(r_key, slice) else [r_key]
        c_idx = range(*c_key.indices(self.width)) if isinstance(c_key, slice) else [c_key]
        cast_val = self.dtype(value)
        for r in r_idx:
            row = self.data[r]
            for c in c_idx:
                row[c] = cast_val

    def __and__(self, other: "_Grid2D") -> "_Grid2D":
        return _Grid2D(
            [
                [bool(a) and bool(b) for a, b in zip(r1, r2)]
                for r1, r2 in zip(self.data, other.data)
            ],
            dtype=bool,
        )

    def __or__(self, other: "_Grid2D") -> "_Grid2D":
        return _Grid2D(
            [
                [bool(a) or bool(b) for a, b in zip(r1, r2)]
                for r1, r2 in zip(self.data, other.data)
            ],
            dtype=bool,
        )

    def __ior__(self, other: "_Grid2D") -> "_Grid2D":
        for r in range(self.height):
            row_s = self.data[r]
            row_o = other.data[r]
            for c in range(self.width):
                row_s[c] = self.dtype(bool(row_s[c]) or bool(row_o[c]))
        return self

    def __invert__(self) -> "_Grid2D":
        return _Grid2D(
            [[not bool(v) for v in row] for row in self.data],
            dtype=bool,
        )

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        if isinstance(other, (int, float, bool)):
            return _Grid2D(
                [[v == other for v in row] for row in self.data],
                dtype=bool,
            )
        return False

    def __ge__(self, threshold: Union[int, float]) -> "_Grid2D":
        return _Grid2D(
            [[v >= threshold for v in row] for row in self.data],
            dtype=bool,
        )

    def max_val(self) -> Union[int, float]:
        if self.size == 0:
            return 0
        return max(max(row) for row in self.data)

    def any(self) -> bool:
        return any(any(bool(v) for v in row) for row in self.data)

    def count_nonzero(self) -> int:
        return sum(sum(1 for v in row if bool(v)) for row in self.data)

    def nonzero_coords(self) -> List[Tuple[int, int]]:
        coords: List[Tuple[int, int]] = []
        for r, row in enumerate(self.data):
            for c, val in enumerate(row):
                if bool(val):
                    coords.append((r, c))
        return coords


def _zeros(shape: Tuple[int, int], dtype: str = "uint8") -> Any:
    """Create a 2D zero mask using NumPy when available or _Grid2D fallback."""
    if np is not None:
        np_dtype = {
            "bool": np.bool_,
            "int32": np.int32,
            "uint8": np.uint8,
            "float32": np.float32,
        }[dtype]
        return np.zeros(shape, dtype=np_dtype)
    py_dtype = {"bool": bool, "int32": int, "uint8": int, "float32": float}[dtype]
    return _Grid2D.zeros(shape, dtype=py_dtype)


def _to_2d_array(array: Any) -> Any:
    """Convert input (NumPy array, PyTorch tensor, PIL Image, list) to 2D array."""
    if isinstance(array, _Grid2D):
        return array

    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    elif hasattr(array, "cpu") and hasattr(array, "numpy"):
        array = array.cpu().numpy()

    if np is not None:
        arr = np.asarray(array)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError(
                    f"Expected 2D mask or single-channel 3D mask, got shape {arr.shape}"
                )
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {arr.shape}")
        return arr

    # Pure-Python fallback
    if isinstance(array, (list, tuple)):
        if len(array) > 0 and isinstance(array[0], (list, tuple)):
            first = array[0][0] if len(array[0]) > 0 else 0
            dtype = bool if isinstance(first, bool) else (float if isinstance(first, float) else int)
            return _Grid2D([list(row) for row in array], dtype=dtype)
    raise ValueError("Expected a 2D array or nested list")


def _any(mask: Any) -> bool:
    if np is not None and isinstance(mask, np.ndarray):
        return bool(np.any(mask))
    return mask.any()


def _count_nonzero(mask: Any) -> int:
    if np is not None and isinstance(mask, np.ndarray):
        return int(np.count_nonzero(mask))
    return mask.count_nonzero()


def _nonzero_coords(mask: Any) -> List[Tuple[int, int]]:
    if np is not None and isinstance(mask, np.ndarray):
        ys, xs = np.nonzero(mask)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]
    return mask.nonzero_coords()


def binarize_mask(
    mask: Any,
    threshold: float = DEFAULT_THRESHOLD,
) -> Any:
    """Convert a probability map or integer mask into a 2D boolean mask.

    Conventions:
    - If `mask` is already boolean, returns a copy as `bool`.
    - If `mask` is integer with max > 1 (e.g. uint8 0..255 PNG mask), values >= 128
      (or >= threshold * 255 if threshold != 0.5) are treated as positive.
    - Otherwise (float probabilities in [0, 1] or binary {0, 1}), values >= threshold
      are treated as positive.
    """
    arr = _to_2d_array(mask)
    if np is not None and isinstance(arr, np.ndarray):
        if arr.dtype == np.bool_:
            return arr.copy()
        if np.issubdtype(arr.dtype, np.integer) and arr.max(initial=0) > 1:
            scaled_threshold = 128 if threshold == 0.5 else threshold * 255.0
            return arr >= scaled_threshold
        return arr >= threshold

    # _Grid2D fallback
    if arr.dtype is bool:
        return arr.copy()
    if arr.dtype is int and arr.max_val() > 1:
        scaled_threshold = 128 if threshold == 0.5 else threshold * 255.0
        return arr >= scaled_threshold
    return arr >= threshold


def _label_components_bfs(
    binary_mask: Any,
    connectivity: int = 8,
) -> Tuple[Any, int]:
    """BFS connected-component labeling (works for both NumPy and _Grid2D)."""
    height, width = binary_mask.shape
    labels = _zeros((height, width), dtype="int32")
    if not _any(binary_mask):
        return labels, 0

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    current_label = 0
    for start_y, start_x in _nonzero_coords(binary_mask):
        if labels[start_y, start_x] != 0:
            continue
        current_label += 1
        labels[start_y, start_x] = current_label
        queue = deque([(start_y, start_x)])
        while queue:
            cy, cx = queue.popleft()
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and bool(binary_mask[ny, nx])
                    and labels[ny, nx] == 0
                ):
                    labels[ny, nx] = current_label
                    queue.append((ny, nx))

    return labels, current_label


def label_connected_components(
    binary_mask: Any,
    connectivity: int = 8,
) -> Tuple[Any, int]:
    """Label connected components in a 2D boolean mask.

    Args:
        binary_mask: 2D boolean mask.
        connectivity: 4 (orthogonal) or 8 (orthogonal + diagonal, default).

    Returns:
        (labeled_array, num_components)
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    mask_bool = binarize_mask(binary_mask)
    if not _any(mask_bool):
        return _zeros(mask_bool.shape, dtype="int32"), 0

    if np is not None and _ndimage is not None and isinstance(mask_bool, np.ndarray):
        if connectivity == 8:
            structure = np.ones((3, 3), dtype=np.int32)
        else:
            structure = np.array(
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32
            )
        labeled, num_features = _ndimage.label(mask_bool, structure=structure)
        return labeled.astype(np.int32), int(num_features)

    return _label_components_bfs(mask_bool, connectivity=connectivity)


def extract_connected_components(
    mask: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = MIN_COMPONENT_AREA,
    connectivity: int = 8,
) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract connected components and apply the minimum pixel area rule.

    Args:
        mask: 2D probability map or binary mask.
        threshold: Binarization threshold (default 0.5).
        min_area: Minimum connected-component area in pixels (default 100).
        connectivity: 4 or 8 neighbor connectivity (default 8).

    Returns:
        filtered_mask: 2D boolean mask keeping only components with area >= min_area.
        valid_components: List of metadata dicts for components with area >= min_area.
        rejected_components: List of metadata dicts for components with area < min_area.
    """
    binary = binarize_mask(mask, threshold=threshold)
    labeled, num_components = label_connected_components(
        binary, connectivity=connectivity
    )

    if num_components == 0:
        return _zeros(binary.shape, dtype="bool"), [], []

    filtered_mask = _zeros(binary.shape, dtype="bool")
    valid_components: List[Dict[str, Any]] = []
    rejected_components: List[Dict[str, Any]] = []

    for comp_id in range(1, num_components + 1):
        comp_mask = labeled == comp_id
        coords = _nonzero_coords(comp_mask)
        area = len(coords)
        ys = [c[0] for c in coords]
        xs = [c[1] for c in coords]
        bbox = (
            min(ys),
            min(xs),
            max(ys) + 1,
            max(xs) + 1,
        )
        is_kept = area >= min_area
        info = {
            "component_id": comp_id,
            "area": area,
            "bbox": bbox,  # (y_min, x_min, y_max_exclusive, x_max_exclusive)
            "is_kept": is_kept,
        }
        if is_kept:
            filtered_mask |= comp_mask
            valid_components.append(info)
        else:
            rejected_components.append(info)

    return filtered_mask, valid_components, rejected_components


def filter_small_components(
    mask: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = MIN_COMPONENT_AREA,
    connectivity: int = 8,
) -> Any:
    """Return a 2D boolean mask keeping only connected components with >= min_area px."""
    if min_area <= 0:
        return binarize_mask(mask, threshold=threshold)
    filtered_mask, _, _ = extract_connected_components(
        mask,
        threshold=threshold,
        min_area=min_area,
        connectivity=connectivity,
    )
    return filtered_mask


def compute_confusion_counts(
    pred: Any,
    gt: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = 0,
    connectivity: int = 8,
) -> Dict[str, int]:
    """Compute pixel-level confusion matrix counts (TP, FP, FN, TN)."""
    pred_bin = filter_small_components(
        pred, threshold=threshold, min_area=min_area, connectivity=connectivity
    )
    gt_bin = binarize_mask(gt, threshold=threshold)

    if pred_bin.shape != gt_bin.shape:
        raise ValueError(
            f"Shape mismatch between pred {pred_bin.shape} and gt {gt_bin.shape}"
        )

    tp = _count_nonzero(pred_bin & gt_bin)
    fp = _count_nonzero(pred_bin & ~gt_bin)
    fn = _count_nonzero(~pred_bin & gt_bin)
    tn = _count_nonzero(~pred_bin & ~gt_bin)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pred_pos": tp + fp,
        "gt_pos": tp + fn,
        "total_pixels": int(pred_bin.size),
    }


def dice_score(
    pred: Any,
    gt: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = 0,
    connectivity: int = 8,
    empty_value: float = 1.0,
) -> float:
    """Compute Dice coefficient: 2*TP / (2*TP + FP + FN)."""
    counts = compute_confusion_counts(
        pred, gt, threshold=threshold, min_area=min_area, connectivity=connectivity
    )
    denom = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    if denom == 0:
        return float(empty_value)
    return float((2.0 * counts["tp"]) / denom)


def iou_score(
    pred: Any,
    gt: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = 0,
    connectivity: int = 8,
    empty_value: float = 1.0,
) -> float:
    """Compute Intersection over Union (Jaccard): TP / (TP + FP + FN)."""
    counts = compute_confusion_counts(
        pred, gt, threshold=threshold, min_area=min_area, connectivity=connectivity
    )
    denom = counts["tp"] + counts["fp"] + counts["fn"]
    if denom == 0:
        return float(empty_value)
    return float(counts["tp"] / denom)


def recall_score(
    pred: Any,
    gt: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = 0,
    connectivity: int = 8,
    empty_value: float = 1.0,
) -> float:
    """Compute pixel-level Recall (Sensitivity): TP / (TP + FN)."""
    counts = compute_confusion_counts(
        pred, gt, threshold=threshold, min_area=min_area, connectivity=connectivity
    )
    denom = counts["tp"] + counts["fn"]
    if denom == 0:
        return float(empty_value)
    return float(counts["tp"] / denom)


def precision_score(
    pred: Any,
    gt: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = 0,
    connectivity: int = 8,
    empty_value: float = 1.0,
) -> float:
    """Compute pixel-level Precision: TP / (TP + FP)."""
    counts = compute_confusion_counts(
        pred, gt, threshold=threshold, min_area=min_area, connectivity=connectivity
    )
    denom = counts["tp"] + counts["fp"]
    if denom == 0:
        return float(empty_value)
    return float(counts["tp"] / denom)


def pixel_fpr_score(
    pred: Any,
    gt: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = 0,
    connectivity: int = 8,
    empty_value: float = 0.0,
) -> float:
    """Compute pixel-level False Positive Rate: FP / (FP + TN)."""
    counts = compute_confusion_counts(
        pred, gt, threshold=threshold, min_area=min_area, connectivity=connectivity
    )
    denom = counts["fp"] + counts["tn"]
    if denom == 0:
        return float(empty_value)
    return float(counts["fp"] / denom)


def has_positive_prediction(
    pred: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = MIN_COMPONENT_AREA,
    connectivity: int = 8,
) -> bool:
    """Return True if prediction has >= 1 connected component with area >= min_area."""
    filtered = filter_small_components(
        pred, threshold=threshold, min_area=min_area, connectivity=connectivity
    )
    return _any(filtered)


def image_level_fpr(
    normal_preds: Sequence[Any],
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = MIN_COMPONENT_AREA,
    connectivity: int = 8,
) -> float:
    """Compute image-level False Positive Rate on Normal/Healthy images.

    According to the FAR protocol, a normal image is counted as a False Positive (FP)
    if and only if it contains at least one predicted connected component with
    area >= min_area (default 100 pixels).
    """
    if len(normal_preds) == 0:
        return 0.0
    fp_images = sum(
        1
        for pred in normal_preds
        if has_positive_prediction(
            pred,
            threshold=threshold,
            min_area=min_area,
            connectivity=connectivity,
        )
    )
    return float(fp_images / len(normal_preds))


def image_level_recall(
    polyp_preds: Sequence[Any],
    polyp_gts: Sequence[Any],
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = MIN_COMPONENT_AREA,
    connectivity: int = 8,
    min_iou: float = 0.0,
) -> float:
    """Compute image-level Detection Recall across Polyp images.

    A polyp image is considered detected (TP) if the component-filtered prediction
    (components >= min_area px) overlaps with the ground-truth mask (TP_pixels > 0,
    and IoU >= min_iou when min_iou > 0).
    """
    if len(polyp_preds) != len(polyp_gts):
        raise ValueError("polyp_preds and polyp_gts must have the same length")
    if len(polyp_preds) == 0:
        return 0.0

    detected = 0
    for pred, gt in zip(polyp_preds, polyp_gts):
        counts = compute_confusion_counts(
            pred,
            gt,
            threshold=threshold,
            min_area=min_area,
            connectivity=connectivity,
        )
        denom = counts["tp"] + counts["fp"] + counts["fn"]
        iou = (counts["tp"] / denom) if denom > 0 else 0.0
        if counts["tp"] > 0 and iou >= min_iou:
            detected += 1

    return float(detected / len(polyp_preds))


def compute_mask_metrics(
    pred: Any,
    gt: Any,
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = MIN_COMPONENT_AREA,
    connectivity: int = 8,
) -> Dict[str, Any]:
    """Compute complete metrics for a single (pred, gt) mask pair.

    Returns both raw pixel metrics (`raw_*`) and connected-component filtered metrics
    (`dice`, `iou`, `recall`, `precision`, `pixel_fpr`), along with component details
    and image-level outcome (`TP`, `FP`, `TN`, `FN`).
    """
    raw_counts = compute_confusion_counts(
        pred, gt, threshold=threshold, min_area=0, connectivity=connectivity
    )
    filtered_mask, valid_comps, rejected_comps = extract_connected_components(
        pred, threshold=threshold, min_area=min_area, connectivity=connectivity
    )
    cc_counts = compute_confusion_counts(
        filtered_mask, gt, threshold=0.5, min_area=0, connectivity=connectivity
    )

    is_polyp_gt = cc_counts["gt_pos"] > 0
    has_alarm = len(valid_comps) > 0

    if is_polyp_gt:
        image_outcome = "TP" if cc_counts["tp"] > 0 else "FN"
    else:
        image_outcome = "FP" if has_alarm else "TN"

    return {
        "is_polyp_gt": is_polyp_gt,
        "has_positive_alarm": has_alarm,
        "image_outcome": image_outcome,
        # Filtered metrics (with connected component >= min_area rule)
        "dice": dice_score(filtered_mask, gt, threshold=0.5, min_area=0),
        "iou": iou_score(filtered_mask, gt, threshold=0.5, min_area=0),
        "recall": recall_score(filtered_mask, gt, threshold=0.5, min_area=0),
        "precision": precision_score(filtered_mask, gt, threshold=0.5, min_area=0),
        "pixel_fpr": pixel_fpr_score(filtered_mask, gt, threshold=0.5, min_area=0),
        "tp": cc_counts["tp"],
        "fp": cc_counts["fp"],
        "fn": cc_counts["fn"],
        "tn": cc_counts["tn"],
        "pred_pos": cc_counts["pred_pos"],
        "gt_pos": cc_counts["gt_pos"],
        # Raw metrics (before connected component filtering)
        "raw_dice": dice_score(pred, gt, threshold=threshold, min_area=0),
        "raw_iou": iou_score(pred, gt, threshold=threshold, min_area=0),
        "raw_recall": recall_score(pred, gt, threshold=threshold, min_area=0),
        "raw_precision": precision_score(pred, gt, threshold=threshold, min_area=0),
        "raw_pixel_fpr": pixel_fpr_score(pred, gt, threshold=threshold, min_area=0),
        "raw_pred_pos": raw_counts["pred_pos"],
        # Connected component statistics
        "min_area": min_area,
        "num_valid_components": len(valid_comps),
        "num_rejected_components": len(rejected_comps),
        "valid_components": valid_comps,
        "rejected_components": rejected_comps,
    }


def evaluate_dataset(
    samples: Iterable[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    min_area: int = MIN_COMPONENT_AREA,
    connectivity: int = 8,
) -> Dict[str, Any]:
    """Evaluate a dataset of samples and return summary metrics + per-image records.

    Each item in `samples` should be a dict containing:
      - `pred`: 2D prediction array (probability map or binary mask)
      - `gt`: Optional 2D ground-truth mask (if omitted, an all-zero normal mask
        of `pred.shape` is assumed)
      - `image_id` (optional): Identifier for tracking TP/FP/TN/FN cases
      - `patient_id` (optional): Patient identifier
    """
    per_image: List[Dict[str, Any]] = []
    polyp_metrics: List[Dict[str, Any]] = []
    normal_metrics: List[Dict[str, Any]] = []

    outcomes: Dict[str, List[str]] = {"TP": [], "FP": [], "TN": [], "FN": []}

    for idx, item in enumerate(samples):
        image_id = str(item.get("image_id", f"sample_{idx}"))
        patient_id = str(item.get("patient_id", ""))
        pred = item["pred"]
        pred_arr = _to_2d_array(pred)

        if "gt" in item and item["gt"] is not None:
            gt = item["gt"]
        else:
            gt = _zeros(pred_arr.shape, dtype="uint8")

        res = compute_mask_metrics(
            pred_arr,
            gt,
            threshold=threshold,
            min_area=min_area,
            connectivity=connectivity,
        )
        res["image_id"] = image_id
        res["patient_id"] = patient_id

        per_image.append(res)
        outcomes[res["image_outcome"]].append(image_id)

        if res["is_polyp_gt"]:
            polyp_metrics.append(res)
        else:
            normal_metrics.append(res)

    def _mean(items: List[Dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return float(sum(float(x[key]) for x in items) / len(items))

    num_polyp = len(polyp_metrics)
    num_normal = len(normal_metrics)
    num_fp_normal = sum(1 for x in normal_metrics if x["has_positive_alarm"])
    num_raw_fp_normal = sum(1 for x in normal_metrics if x["raw_pred_pos"] > 0)
    num_tp_polyp = sum(1 for x in polyp_metrics if x["image_outcome"] == "TP")

    summary = {
        "protocol": {
            "threshold": threshold,
            "min_component_area": min_area,
            "connectivity": connectivity,
        },
        "counts": {
            "total_images": len(per_image),
            "polyp_images": num_polyp,
            "normal_images": num_normal,
            "TP_images": len(outcomes["TP"]),
            "FP_images": len(outcomes["FP"]),
            "TN_images": len(outcomes["TN"]),
            "FN_images": len(outcomes["FN"]),
        },
        # Primary metrics (with connected-component rule >= min_area px)
        "polyp_segmentation": {
            "dice": _mean(polyp_metrics, "dice"),
            "iou": _mean(polyp_metrics, "iou"),
            "recall": _mean(polyp_metrics, "recall"),
            "precision": _mean(polyp_metrics, "precision"),
            "image_detection_recall": (
                float(num_tp_polyp / num_polyp) if num_polyp > 0 else 0.0
            ),
        },
        "normal_false_alarm": {
            "fpr": float(num_fp_normal / num_normal) if num_normal > 0 else 0.0,
            "fp_images": num_fp_normal,
            "tn_images": num_normal - num_fp_normal,
            "pixel_fpr": _mean(normal_metrics, "pixel_fpr"),
            "raw_fpr_without_cc_filter": (
                float(num_raw_fp_normal / num_normal) if num_normal > 0 else 0.0
            ),
        },
        # Raw metrics before connected-component filtering
        "raw_polyp_segmentation": {
            "dice": _mean(polyp_metrics, "raw_dice"),
            "iou": _mean(polyp_metrics, "raw_iou"),
            "recall": _mean(polyp_metrics, "raw_recall"),
            "precision": _mean(polyp_metrics, "raw_precision"),
        },
        "outcome_image_ids": outcomes,
        "per_image": per_image,
    }
    return summary


def run_hand_calculated_tests(
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Run hand-calculated unit tests for Dice, IoU, Recall, FPR, and CC >= 100 px.

    Verifies exact numerical agreement between manual hand calculations and code
    outputs, and writes the full verification log to `results/M4/tests.txt`.
    """
    lines: List[str] = []

    def log(text: str = "") -> None:
        lines.append(text)

    log("======================================================================")
    log("FAR-POLYP-SEG | MODULE 4 (M4: METRICS) - HAND-CALCULATED VERIFICATION")
    log("======================================================================")
    log(f"Connected Component Minimum Area Rule: >= {MIN_COMPONENT_AREA} pixels")
    log("")

    # ------------------------------------------------------------------
    # TEST 1: Hand-calculated 10x10 partial overlap mask (Pixel metrics)
    # ------------------------------------------------------------------
    log("[TEST 1] 10x10 Partial Overlap Mask (Dice, IoU, Recall, Precision, Pixel FPR)")
    gt_1 = _zeros((10, 10), dtype="uint8")
    pred_1 = _zeros((10, 10), dtype="uint8")
    # GT: rows 2..5 (4 rows), cols 2..6 (5 cols) -> 4 * 5 = 20 pixels
    gt_1[2:6, 2:7] = 1
    # Pred: rows 4..7 (4 rows), cols 2..6 (5 cols) -> 4 * 5 = 20 pixels
    pred_1[4:8, 2:7] = 1
    # Overlap (TP): rows 4..5 (2 rows), cols 2..6 (5 cols) -> 2 * 5 = 10 pixels
    # FP = 20 - 10 = 10 pixels
    # FN = 20 - 10 = 10 pixels
    # TN = 100 - (10 + 10 + 10) = 70 pixels
    # Hand calculations:
    #   Dice      = 2*10 / (20 + 20) = 20 / 40 = 0.500000
    #   IoU       = 10 / (10 + 10 + 10) = 10 / 30 = 1/3 = 0.333333...
    #   Recall    = 10 / (10 + 10) = 10 / 20 = 0.500000
    #   Precision = 10 / (10 + 10) = 10 / 20 = 0.500000
    #   Pixel FPR = 10 / (10 + 70) = 10 / 80 = 0.125000
    c1 = compute_confusion_counts(pred_1, gt_1, min_area=0)
    d1 = dice_score(pred_1, gt_1, min_area=0)
    i1 = iou_score(pred_1, gt_1, min_area=0)
    r1 = recall_score(pred_1, gt_1, min_area=0)
    p1 = precision_score(pred_1, gt_1, min_area=0)
    f1 = pixel_fpr_score(pred_1, gt_1, min_area=0)

    assert c1["tp"] == 10 and c1["fp"] == 10 and c1["fn"] == 10 and c1["tn"] == 70
    assert math.isclose(d1, 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(i1, 1.0 / 3.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(r1, 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(p1, 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(f1, 0.125, rel_tol=1e-9, abs_tol=1e-9)

    log("  Hand calculation : TP=10, FP=10, FN=10, TN=70 (Total=100 px)")
    log(
        f"  Code counts      : TP={c1['tp']}, FP={c1['fp']}, FN={c1['fn']}, TN={c1['tn']}"
    )
    log(f"  Dice      : hand = 20/40 = 0.500000 | code = {d1:.6f}")
    log(f"  IoU       : hand = 10/30 = 0.333333 | code = {i1:.6f}")
    log(f"  Recall    : hand = 10/20 = 0.500000 | code = {r1:.6f}")
    log(f"  Precision : hand = 10/20 = 0.500000 | code = {p1:.6f}")
    log(f"  Pixel FPR : hand = 10/80 = 0.125000 | code = {f1:.6f}")
    log("  Status    : PASS")
    log("")

    # ------------------------------------------------------------------
    # TEST 2: Asymmetric Overlap (Different Recall vs Precision)
    # ------------------------------------------------------------------
    log("[TEST 2] Asymmetric Overlap on 20x20 Grid (Hand-calculated exact fractions)")
    gt_2 = _zeros((20, 20), dtype="uint8")
    pred_2 = _zeros((20, 20), dtype="uint8")
    # GT: 10x12 = 120 pixels (rows 0..9, cols 0..11)
    gt_2[0:10, 0:12] = 1
    # Pred: 10x9 = 90 pixels overlapping GT (rows 0..9, cols 0..8)
    #       + 6x10 = 60 pixels outside GT (rows 12..17, cols 0..9)
    # Total Pred = 150 pixels
    pred_2[0:10, 0:9] = 1
    pred_2[12:18, 0:10] = 1
    # Hand calculations:
    #   Total pixels = 400
    #   TP = 90, FP = 60, FN = 120 - 90 = 30, TN = 400 - (90 + 60 + 30) = 220
    #   Dice      = 2*90 / (150 + 120) = 180 / 270 = 2/3 = 0.666667
    #   IoU       = 90 / (90 + 60 + 30) = 90 / 180 = 0.500000
    #   Recall    = 90 / 120 = 0.750000
    #   Precision = 90 / 150 = 0.600000
    #   Pixel FPR = 60 / (60 + 220) = 60 / 280 = 3/14 = 0.214286
    c2 = compute_confusion_counts(pred_2, gt_2, min_area=0)
    d2 = dice_score(pred_2, gt_2, min_area=0)
    i2 = iou_score(pred_2, gt_2, min_area=0)
    r2 = recall_score(pred_2, gt_2, min_area=0)
    p2 = precision_score(pred_2, gt_2, min_area=0)
    f2 = pixel_fpr_score(pred_2, gt_2, min_area=0)

    assert c2["tp"] == 90 and c2["fp"] == 60 and c2["fn"] == 30 and c2["tn"] == 220
    assert math.isclose(d2, 2.0 / 3.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(i2, 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(r2, 0.75, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(p2, 0.60, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(f2, 3.0 / 14.0, rel_tol=1e-9, abs_tol=1e-9)

    log("  Hand calculation : TP=90, FP=60, FN=30, TN=220 (Total=400 px)")
    log(
        f"  Code counts      : TP={c2['tp']}, FP={c2['fp']}, FN={c2['fn']}, TN={c2['tn']}"
    )
    log(f"  Dice      : hand = 180/270 = 0.666667 | code = {d2:.6f}")
    log(f"  IoU       : hand =  90/180 = 0.500000 | code = {i2:.6f}")
    log(f"  Recall    : hand =  90/120 = 0.750000 | code = {r2:.6f}")
    log(f"  Precision : hand =  90/150 = 0.600000 | code = {p2:.6f}")
    log(f"  Pixel FPR : hand =  60/280 = 0.214286 | code = {f2:.6f}")
    log("  Status    : PASS")
    log("")

    # ------------------------------------------------------------------
    # TEST 3: Edge Cases (Perfect Match, Disjoint, Both Empty)
    # ------------------------------------------------------------------
    log("[TEST 3] Edge Cases (Perfect Match, Complete Miss, Empty-Empty Normal)")
    # 3a: Perfect match
    assert math.isclose(dice_score(gt_1, gt_1), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(iou_score(gt_1, gt_1), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(recall_score(gt_1, gt_1), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(pixel_fpr_score(gt_1, gt_1), 0.0, rel_tol=1e-9, abs_tol=1e-9)

    # 3b: Complete miss (disjoint)
    disjoint_pred = _zeros((10, 10), dtype="uint8")
    disjoint_pred[8:10, 8:10] = 1  # 4 pixels disjoint from gt_1[2:6, 2:7]
    assert math.isclose(dice_score(disjoint_pred, gt_1), 0.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(iou_score(disjoint_pred, gt_1), 0.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(recall_score(disjoint_pred, gt_1), 0.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(pixel_fpr_score(disjoint_pred, gt_1), 4.0 / 80.0, rel_tol=1e-9, abs_tol=1e-9)

    # 3c: Both empty (True Negative normal image)
    empty_mask = _zeros((10, 10), dtype="uint8")
    assert math.isclose(dice_score(empty_mask, empty_mask), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(iou_score(empty_mask, empty_mask), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(pixel_fpr_score(empty_mask, empty_mask), 0.0, rel_tol=1e-9, abs_tol=1e-9)

    log("  3a Perfect Match : Dice=1.000000, IoU=1.000000, Recall=1.000000, FPR=0.000000")
    log("  3b Disjoint Miss : Dice=0.000000, IoU=0.000000, Recall=0.000000, FPR=0.050000 (4/80)")
    log("  3c Empty Normal  : Dice=1.000000, IoU=1.000000, FPR=0.000000")
    log("  Status           : PASS")
    log("")

    # ------------------------------------------------------------------
    # TEST 4: Connected Component Rule >= 100 px (Boundary Test: 45, 99, 100, 150 px)
    # ------------------------------------------------------------------
    log("[TEST 4] Connected Component Rule >= 100 px (Boundary & Filtering Test)")
    cc_test_mask = _zeros((60, 60), dtype="uint8")
    # Component A: 5x9 = 45 px (< 100 px -> MUST BE REJECTED)
    cc_test_mask[2:7, 2:11] = 1
    # Component B: 9x11 = 99 px (< 100 px -> BOUNDARY JUST BELOW 100 -> MUST BE REJECTED)
    cc_test_mask[2:11, 20:31] = 1
    # Component C: 10x10 = 100 px (== 100 px -> BOUNDARY EXACTLY 100 -> MUST BE KEPT)
    cc_test_mask[20:30, 2:12] = 1
    # Component D: 10x15 = 150 px (> 100 px -> MUST BE KEPT)
    cc_test_mask[20:30, 25:40] = 1

    filtered_cc, valid_comps, rejected_comps = extract_connected_components(
        cc_test_mask, min_area=100, connectivity=8
    )
    raw_total_px = _count_nonzero(cc_test_mask)
    filtered_total_px = _count_nonzero(filtered_cc)
    valid_areas = sorted(c["area"] for c in valid_comps)
    rejected_areas = sorted(c["area"] for c in rejected_comps)

    assert raw_total_px == 45 + 99 + 100 + 150 == 394
    assert filtered_total_px == 100 + 150 == 250
    assert valid_areas == [100, 150]
    assert rejected_areas == [45, 99]

    log(f"  Input blobs      : 45 px, 99 px, 100 px, 150 px (Total raw = {raw_total_px} px)")
    log(f"  Rejected (<100)  : {rejected_areas} px (2 components removed)")
    log(f"  Kept (>=100)     : {valid_areas} px (2 components retained)")
    log(f"  Filtered pixels  : hand = 250 px | code = {filtered_total_px} px")
    log("  Status           : PASS")
    log("")

    # ------------------------------------------------------------------
    # TEST 5: Effect of CC >= 100 px on Polyp Segmentation (Removing FP noise blob)
    # ------------------------------------------------------------------
    log("[TEST 5] CC >= 100 px Effect on Polyp Mask with Spurious Noise Blob")
    gt_5 = _zeros((40, 40), dtype="uint8")
    pred_5 = _zeros((40, 40), dtype="uint8")
    gt_5[2:12, 2:14] = 1      # 10x12 = 120 px GT polyp
    pred_5[2:12, 2:12] = 1    # 10x10 = 100 px true detection overlapping GT
    pred_5[25:31, 20:30] = 1  # 6x10 = 60 px false positive noise blob (< 100 px)

    # Without CC filter (raw):
    #   TP = 100, FP = 60, FN = 20
    #   Raw Dice = 200 / (160 + 120) = 200 / 280 = 5/7 = 0.714286
    #   Raw IoU  = 100 / (100 + 60 + 20) = 100 / 180 = 5/9 = 0.555556
    # With CC >= 100 px filter:
    #   The 60 px FP blob is removed!
    #   TP = 100, FP = 0, FN = 20
    #   Filtered Dice = 200 / (100 + 120) = 200 / 220 = 10/11 = 0.909091
    #   Filtered IoU  = 100 / (100 + 0 + 20) = 100 / 120 = 5/6 = 0.833333
    #   Filtered Recall = 100 / 120 = 5/6 = 0.833333
    m5 = compute_mask_metrics(pred_5, gt_5, min_area=100)
    assert math.isclose(m5["raw_dice"], 5.0 / 7.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(m5["raw_iou"], 5.0 / 9.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(m5["dice"], 10.0 / 11.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(m5["iou"], 5.0 / 6.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(m5["recall"], 5.0 / 6.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(m5["precision"], 1.0, rel_tol=1e-9, abs_tol=1e-9)

    log(
        f"  Before CC filter : TP=100, FP=60, FN=20 -> Dice={m5['raw_dice']:.6f} (5/7), IoU={m5['raw_iou']:.6f} (5/9)"
    )
    log(
        f"  After CC >=100px : TP=100, FP=0,  FN=20 -> Dice={m5['dice']:.6f} (10/11), IoU={m5['iou']:.6f} (5/6)"
    )
    log("  Status           : PASS")
    log("")

    # ------------------------------------------------------------------
    # TEST 6: Image-level FPR on Normal Images with CC >= 100 px Rule
    # ------------------------------------------------------------------
    log("[TEST 6] Image-level FPR on 5 Normal (Healthy) Images under CC >= 100 px Rule")
    # Normal 1: Two small noise blobs (30 px + 85 px = 115 px total, but each < 100 px) -> TN
    norm_1 = _zeros((50, 50), dtype="uint8")
    norm_1[2:7, 2:8] = 1     # 5x6 = 30 px
    norm_1[20:25, 10:27] = 1 # 5x17 = 85 px
    # Normal 2: Single blob of 10x12 = 120 px (>= 100 px) -> FP alarm
    norm_2 = _zeros((50, 50), dtype="uint8")
    norm_2[10:20, 10:22] = 1
    # Normal 3: Completely clean (0 px) -> TN
    norm_3 = _zeros((50, 50), dtype="uint8")
    # Normal 4: Single blob of 9x11 = 99 px (< 100 px) -> TN
    norm_4 = _zeros((50, 50), dtype="uint8")
    norm_4[5:14, 5:16] = 1
    # Normal 5: Single blob of 10x10 = 100 px (>= 100 px) -> FP alarm
    norm_5 = _zeros((50, 50), dtype="uint8")
    norm_5[15:25, 15:25] = 1

    normal_list = [norm_1, norm_2, norm_3, norm_4, norm_5]
    fpr_cc = image_level_fpr(normal_list, min_area=100)
    fpr_raw = image_level_fpr(normal_list, min_area=1)

    assert math.isclose(fpr_cc, 2.0 / 5.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(fpr_raw, 4.0 / 5.0, rel_tol=1e-9, abs_tol=1e-9)

    log("  Normal #1 (30px + 85px disconnected) : max CC =  85 px < 100 -> TN")
    log("  Normal #2 (120px single component)   : max CC = 120 px >=100 -> FP")
    log("  Normal #3 (0px clean)                : max CC =   0 px < 100 -> TN")
    log("  Normal #4 (99px boundary component)  : max CC =  99 px < 100 -> TN")
    log("  Normal #5 (100px boundary component) : max CC = 100 px >=100 -> FP")
    log(f"  Image-level FPR (CC >= 100 px) : hand = 2/5 = 0.400000 | code = {fpr_cc:.6f}")
    log(f"  Raw FPR (no CC filter)         : hand = 4/5 = 0.800000 | code = {fpr_raw:.6f}")
    log("  Status                         : PASS")
    log("")

    # ------------------------------------------------------------------
    # TEST 7: Full Dataset Evaluation (Polyp + Normal images for Module 1/2/4)
    # ------------------------------------------------------------------
    log("[TEST 7] End-to-End Dataset Evaluation (Polyp + Normal Protocol Summary)")
    gt_b = _zeros((40, 40), dtype="uint8")
    pred_b = _zeros((40, 40), dtype="uint8")
    gt_b[5:15, 5:15] = 1   # 10x10 = 100 px GT
    pred_b[5:13, 5:15] = 1 # 8x10 = 80 px prediction (< 100 px -> filtered to 0!)

    dataset_samples = [
        {"image_id": "polyp_A", "patient_id": "P01", "pred": pred_5, "gt": gt_5},
        {"image_id": "polyp_B", "patient_id": "P02", "pred": pred_b, "gt": gt_b},
        {"image_id": "norm_1", "patient_id": "P03", "pred": norm_1},
        {"image_id": "norm_2", "patient_id": "P04", "pred": norm_2},
        {"image_id": "norm_3", "patient_id": "P05", "pred": norm_3},
        {"image_id": "norm_4", "patient_id": "P06", "pred": norm_4},
        {"image_id": "norm_5", "patient_id": "P07", "pred": norm_5},
    ]
    summary = evaluate_dataset(dataset_samples, threshold=0.5, min_area=100)

    assert math.isclose(summary["polyp_segmentation"]["dice"], 5.0 / 11.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(summary["polyp_segmentation"]["iou"], 5.0 / 12.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(summary["polyp_segmentation"]["recall"], 5.0 / 12.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(summary["polyp_segmentation"]["image_detection_recall"], 0.5, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(summary["normal_false_alarm"]["fpr"], 0.4, rel_tol=1e-9, abs_tol=1e-9)
    assert summary["outcome_image_ids"]["TP"] == ["polyp_A"]
    assert summary["outcome_image_ids"]["FN"] == ["polyp_B"]
    assert summary["outcome_image_ids"]["FP"] == ["norm_2", "norm_5"]
    assert summary["outcome_image_ids"]["TN"] == ["norm_1", "norm_3", "norm_4"]

    log(
        f"  Mean Polyp Dice        : hand = 5/11 = 0.454545 | code = {summary['polyp_segmentation']['dice']:.6f}"
    )
    log(
        f"  Mean Polyp IoU         : hand = 5/12 = 0.416667 | code = {summary['polyp_segmentation']['iou']:.6f}"
    )
    log(
        f"  Mean Polyp Recall      : hand = 5/12 = 0.416667 | code = {summary['polyp_segmentation']['recall']:.6f}"
    )
    log(
        f"  Polyp Image Recall     : hand =  1/2 = 0.500000 | code = {summary['polyp_segmentation']['image_detection_recall']:.6f}"
    )
    log(
        f"  Normal Image FPR       : hand =  2/5 = 0.400000 | code = {summary['normal_false_alarm']['fpr']:.6f}"
    )
    log(f"  Outcome Partitioning   : TP={summary['outcome_image_ids']['TP']}, "
        f"FN={summary['outcome_image_ids']['FN']}, "
        f"FP={summary['outcome_image_ids']['FP']}, "
        f"TN={summary['outcome_image_ids']['TN']}")
    log("  Status                 : PASS")
    log("")
    log("======================================================================")
    log("ALL 7 HAND-CALCULATED METRIC & CONNECTED-COMPONENT TESTS PASSED (100%)")
    log("======================================================================")

    report = "\n".join(lines) + "\n"
    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report, encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M4 evaluation metrics verification or evaluate predictions."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2] / "results" / "M4" / "tests.txt"
        ),
        help="Path to write the hand-calculated verification report.",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default=None,
        help="Optional path to save JSON test summary.",
    )
    args = parser.parse_args()

    report = run_hand_calculated_tests(output_path=args.output)
    print(report, end="")
    if args.json_summary:
        summary_path = Path(args.json_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"status": "PASS", "tests_run": 7}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
