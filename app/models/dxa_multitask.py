from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class ModelOutput:
    anatomy_logits: torch.Tensor
    spine_logits: torch.Tensor
    hip_logits: torch.Tensor
    quality_logits: torch.Tensor
    features: torch.Tensor


class DXAMultiTaskModel(nn.Module):
    """
    Shared ResNet-18 backbone with four prediction heads.

    Anatomy:
        0 = spine
        1 = left_hip
        2 = right_hip

    Spine:
        0 = layout violation
        1 = axis violation
        2 = artifact violation

    Hip:
        0 = positioning/rotation violation
        1 = ROI violation

    Overall:
        0 = quality_class
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)

        self.anatomy_head = nn.Linear(in_features, 3)
        self.spine_head = nn.Linear(in_features, 3)
        self.hip_head = nn.Linear(in_features, 2)
        self.quality_head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        features = self.backbone(x)
        features = self.dropout(features)

        return ModelOutput(
            anatomy_logits=self.anatomy_head(features),
            spine_logits=self.spine_head(features),
            hip_logits=self.hip_head(features),
            quality_logits=self.quality_head(features),
            features=features,
        )

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def cam_target_layer(self):
        """Last convolutional layer for Grad-CAM."""
        return self.backbone.layer4[-1]


def build_model(pretrained: bool = True) -> DXAMultiTaskModel:
    return DXAMultiTaskModel(pretrained=pretrained)
