from __future__ import annotations

import argparse
from pathlib import Path

from app.training.trainer import fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DXA ResNet-18 multi-task baseline.")
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("val_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline_resnet18"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--root", type=Path, default=Path("data/raw/dataset"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretrained", dest="pretrained", action="store_true")
    group.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()

    summary = fit(
        args.train_csv, args.val_csv, args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        image_size=args.image_size, num_workers=args.num_workers,
        seed=args.seed, pretrained=args.pretrained,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        head_lr=args.head_lr, backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay, patience=args.patience,
        root=args.root,
    )

    print("\nTraining finished:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
