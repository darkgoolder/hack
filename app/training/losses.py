from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE that ignores entries where mask == 0."""
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    masked = elementwise * mask
    return masked.sum() / mask.sum().clamp_min(1.0)


def multitask_loss(
    output,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    anatomy: torch.Tensor,
    spine_pos_weight: torch.Tensor | None = None,
    hip_pos_weight: torch.Tensor | None = None,
    quality_pos_weight: torch.Tensor | None = None,
    anatomy_class_weight: torch.Tensor | None = None,
    lambda_spine: float = 1.0,
    lambda_hip: float = 1.0,
    lambda_quality: float = 1.0,
):
    """
    targets columns:
        0 spine_layout
        1 spine_axis
        2 spine_artifact
        3 hip_position_rotation
        4 hip_roi
        5 quality_class
    """
    anatomy_loss = F.cross_entropy(
        output.anatomy_logits,
        anatomy,
        weight=anatomy_class_weight,
    )

    spine_loss = masked_bce_with_logits(
        output.spine_logits,
        targets[:, 0:3],
        target_mask[:, 0:3],
        pos_weight=spine_pos_weight,
    )

    hip_loss = masked_bce_with_logits(
        output.hip_logits,
        targets[:, 3:5],
        target_mask[:, 3:5],
        pos_weight=hip_pos_weight,
    )

    quality_loss = masked_bce_with_logits(
        output.quality_logits.squeeze(1),
        targets[:, 5],
        target_mask[:, 5],
        pos_weight=quality_pos_weight,
    )

    total = (
        anatomy_loss
        + lambda_spine * spine_loss
        + lambda_hip * hip_loss
        + lambda_quality * quality_loss
    )

    return {
        "loss": total,
        "anatomy_loss": anatomy_loss.detach(),
        "spine_loss": spine_loss.detach(),
        "hip_loss": hip_loss.detach(),
        "quality_loss": quality_loss.detach(),
    }
