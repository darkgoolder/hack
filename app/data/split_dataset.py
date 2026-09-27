from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from app.data.schema import ANATOMY_VALUES, CORE_COLUMNS, LABEL_COLUMNS


REQUIRED_COLUMNS = set(CORE_COLUMNS) | set(LABEL_COLUMNS)


def _normalise_binary(value: object) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return 0.0
    number = float(value)
    if number not in (0.0, 1.0):
        raise ValueError(f"Expected 0/1/blank, got {value!r}")
    return number


def validate_manifest(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Manifest is missing required columns: " + ", ".join(missing)
        )

    if df.empty:
        raise ValueError("Manifest is empty.")

    regions = set(df["anatomical_region"].dropna().astype(str))
    unknown_regions = sorted(regions - ANATOMY_VALUES)
    if unknown_regions:
        raise ValueError(f"Unknown anatomical_region values: {unknown_regions}")

    for column in LABEL_COLUMNS:
        df[column].map(_normalise_binary)

    duplicate_paths = df["image_path"].astype(str).duplicated()
    if duplicate_paths.any():
        examples = df.loc[duplicate_paths, "image_path"].head(5).tolist()
        raise ValueError(f"Duplicate image_path values found: {examples}")


def _folder_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build one multi-label row per folder for group-level stratification."""
    rows: list[dict[str, object]] = []
    for folder_id, group in df.groupby("folder_id", sort=True):
        row: dict[str, object] = {"folder_id": folder_id}
        for anatomy in sorted(ANATOMY_VALUES):
            row[f"n_{anatomy}"] = int((group["anatomical_region"] == anatomy).sum())
        for label in LABEL_COLUMNS:
            row[label] = int(group[label].fillna(0).astype(float).max())
        # Useful extra signal for balancing the overall number of failed images.
        row["n_images"] = int(len(group))
        row["n_quality_failures"] = int(
            group.get("quality_class", pd.Series(index=group.index, dtype=float))
            .fillna(0)
            .astype(float)
            .sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _objective(
    current_counts: np.ndarray,
    desired_counts: np.ndarray,
    candidate_counts: np.ndarray,
    current_size: int,
    desired_size: int,
    candidate_size: int,
) -> float:
    """Lower is better. Normalised L1 distance over group-level targets."""
    new_counts = current_counts + candidate_counts
    # +1 avoids unstable division for labels that happen to be absent.
    count_error = np.abs(new_counts - desired_counts).sum() / (np.abs(desired_counts).sum() + 1.0)
    size_error = abs((current_size + candidate_size) - desired_size) / max(desired_size, 1)
    return float(count_error + 0.25 * size_error)


def _greedy_group_holdout(
    groups: pd.DataFrame,
    fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Deterministic greedy group split that approximately preserves multi-label rates."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    group_ids = groups["folder_id"].astype(str).to_numpy()
    feature_cols = [c for c in groups.columns if c not in {"folder_id", "n_images"}]
    matrix = groups[feature_cols].fillna(0).to_numpy(dtype=float)

    order = rng.permutation(len(groups))
    # Place rarer/high-signal groups first; the random permutation is only a tie-breaker.
    rarity = matrix.sum(axis=1)
    ordered_indices = sorted(order.tolist(), key=lambda i: (-rarity[i], str(group_ids[i])))

    target_n = max(1, int(round(len(groups) * fraction)))
    desired_counts = matrix.sum(axis=0) * fraction

    chosen: set[str] = set()
    current_counts = np.zeros(matrix.shape[1], dtype=float)
    current_size = 0

    remaining = list(ordered_indices)
    while remaining and current_size < target_n:
        best_idx = None
        best_score = float("inf")
        for idx in remaining:
            score = _objective(
                current_counts,
                desired_counts,
                matrix[idx],
                current_size,
                target_n,
                1,
            )
            # Never exceed target size unless this is the last option.
            if current_size + 1 > target_n and remaining:
                continue
            if score < best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            best_idx = remaining[0]
        remaining.remove(best_idx)
        chosen.add(str(group_ids[best_idx]))
        current_counts += matrix[best_idx]
        current_size += 1

    all_groups = set(group_ids.tolist())
    return chosen, all_groups - chosen


def make_splits(
    manifest_path: Path,
    output_dir: Path,
    test_fraction: float = 0.20,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest)

    manifest["folder_id"] = manifest["folder_id"].astype(str)
    groups = _folder_matrix(manifest)

    test_groups, remaining_groups = _greedy_group_holdout(groups, test_fraction, seed)
    remaining = groups[groups["folder_id"].isin(remaining_groups)].reset_index(drop=True)
    # val_fraction is relative to the original full dataset; convert to the remaining pool.
    relative_val_fraction = val_fraction / (1.0 - test_fraction)
    val_groups, train_groups = _greedy_group_holdout(remaining, relative_val_fraction, seed + 1)

    split_by_folder = {g: "test" for g in test_groups}
    split_by_folder.update({g: "val" for g in val_groups})
    split_by_folder.update({g: "train" for g in train_groups})

    if len(split_by_folder) != groups.shape[0]:
        raise RuntimeError("Every folder must belong to exactly one split.")

    manifest["split"] = manifest["folder_id"].map(split_by_folder)
    if manifest["split"].isna().any():
        raise RuntimeError("Some manifest rows have no split.")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "manifest_with_split.csv", index=False)

    stats: dict[str, object] = {
        "seed": seed,
        "fractions": {"train": 1 - test_fraction - val_fraction, "val": val_fraction, "test": test_fraction},
        "folders": {k: int((groups["folder_id"].isin(v)).sum()) for k, v in {
            "train": train_groups,
            "val": val_groups,
            "test": test_groups,
        }.items()},
        "images": {},
    }

    for split in ("train", "val", "test"):
        subset = manifest[manifest["split"] == split]
        stats["images"][split] = int(len(subset))
        anatomy_counts = subset["anatomical_region"].value_counts().to_dict()
        stats[f"anatomy_{split}"] = {str(k): int(v) for k, v in anatomy_counts.items()}
        for label in LABEL_COLUMNS:
            values = subset[label].dropna().astype(int)
            stats[f"{label}_{split}"] = {
                "0": int((values == 0).sum()),
                "1": int((values == 1).sum()),
                "blank": int(subset[label].isna().sum()),
            }
        subset[["folder_id", "image_path", "anatomical_region", "laterality", *LABEL_COLUMNS, "split"]].to_csv(
            output_dir / f"{split}.csv", index=False
        )

    # Explicit leakage check.
    folder_sets = {split: set(manifest.loc[manifest["split"] == split, "folder_id"]) for split in ("train", "val", "test")}
    if folder_sets["train"] & folder_sets["val"] or folder_sets["train"] & folder_sets["test"] or folder_sets["val"] & folder_sets["test"]:
        raise RuntimeError("Data leakage detected: a folder occurs in multiple splits.")

    (output_dir / "split_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe train/val/test splits by folder_id.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/splits"))
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = make_splits(
        args.manifest,
        args.output_dir,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
