from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Metrics for one binary target; masked labels are excluded."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        y_true = y_true[mask]
        y_prob = y_prob[mask]

    n = int(len(y_true))
    positives = int((y_true == 1).sum()) if n else 0
    negatives = int((y_true == 0).sum()) if n else 0

    if n == 0:
        return {
            "n": 0, "positives": 0, "negatives": 0,
            "threshold": float(threshold),
            "precision": None, "recall": None, "f1": None,
            "roc_auc": None, "pr_auc": None,
            "tn": None, "fp": None, "fn": None, "tp": None,
        }

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    # ROC-AUC requires both classes. Average precision is also not useful as a
    # discrimination metric when there are no positives.
    if positives > 0 and negatives > 0:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        result["pr_auc"] = float(average_precision_score(y_true, y_prob))

    return result


def anatomy_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    names = ["spine", "left_hip", "right_hip"]
    pred = np.argmax(logits, axis=1)

    return {
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(
            f1_score(y_true, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "classes": {
            name: {
                "support": int((y_true == idx).sum()),
                "recall": float(
                    recall_score(
                        (y_true == idx).astype(int),
                        (pred == idx).astype(int),
                        zero_division=0,
                    )
                ),
            }
            for idx, name in enumerate(names)
        },
    }


def summarize_quality_metrics(
    targets: np.ndarray,
    masks: np.ndarray,
    probabilities: np.ndarray,
    anatomy: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    names = [
        "spine_layout", "spine_axis", "spine_artifact",
        "hip_position_rotation", "hip_roi", "quality_class",
    ]
    thresholds = thresholds or {name: 0.5 for name in names}

    result: dict[str, Any] = {"targets": {}, "quality_by_anatomy": {}}
    f1_values = []

    for idx, name in enumerate(names):
        m = binary_metrics(
            targets[:, idx], probabilities[:, idx], masks[:, idx], thresholds[name]
        )
        result["targets"][name] = m
        if m["f1"] is not None:
            f1_values.append(m["f1"])

    result["detail_macro_f1"] = float(np.mean(f1_values)) if f1_values else None

    for region_idx, region_name in enumerate(["spine", "left_hip", "right_hip"]):
        region_mask = anatomy == region_idx
        result["quality_by_anatomy"][region_name] = binary_metrics(
            targets[:, 5],
            probabilities[:, 5],
            masks[:, 5] * region_mask.astype(float),
            thresholds["quality_class"],
        )

    return result
