from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.data.dicom_dataset import DXAQualityDataset
from app.models.dxa_multitask import build_model
from app.training.losses import multitask_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--root", type=Path, default=Path("data/raw/dataset"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DXAQualityDataset(
        args.manifest, split=args.split, training=False,
        root=args.root,
    )

    model = build_model(pretrained=False).to(device)
    model.eval()

    n = min(args.count, len(dataset))
    batch = [dataset[i] for i in range(n)]

    images = torch.stack([x["image"] for x in batch]).to(device)
    anatomy = torch.stack([x["anatomy"] for x in batch]).to(device)
    targets = torch.stack([x["targets"] for x in batch]).to(device)
    masks = torch.stack([x["target_mask"] for x in batch]).to(device)

    with torch.no_grad():
        output = model(images)

    losses = multitask_loss(
        output,
        targets,
        masks,
        anatomy,
    )

    print(f"Device: {device}")
    print(f"Input: {tuple(images.shape)}")
    print(f"Anatomy logits: {tuple(output.anatomy_logits.shape)}")
    print(f"Spine logits: {tuple(output.spine_logits.shape)}")
    print(f"Hip logits: {tuple(output.hip_logits.shape)}")
    print(f"Quality logits: {tuple(output.quality_logits.shape)}")
    print(f"Total smoke loss: {float(losses['loss']):.6f}")
    print("MODEL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
