from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits, targets, mask, pos_weight=None):
    elementwise = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    masked = elementwise * mask
    return masked.sum() / mask.sum().clamp_min(1.0)
