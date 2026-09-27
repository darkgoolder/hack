from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

STRATIFY_COLUMNS = [
    "has_spine", "has_left_hip", "has_right_hip",
    "spine_layout_positive", "spine_axis_positive", "spine_artifact_positive",
    "hip_rotation_positive", "hip_roi_positive",
]


def build_folder_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for folder_id, g in df.groupby("folder_id", sort=True):
        def any_positive(col: str) -> int:
            vals = pd.to_numeric(g[col], errors="coerce")
            return int((vals == 1).any())

        rows.append({
            "folder_id": folder_id,
            "has_spine": int((g["anatomical_region"] == "spine").any()),
            "has_left_hip": int((g["anatomical_region"] == "left_hip").any()),
            "has_right_hip": int((g["anatomical_region"] == "right_hip").any()),
            "spine_layout_positive": any_positive("spine_layout"),
            "spine_axis_positive": any_positive("spine_axis"),
            "spine_artifact_positive": any_positive("spine_artifact"),
            "hip_rotation_positive": any_positive("hip_position_rotation"),
            "hip_roi_positive": any_positive("hip_roi"),
        })
    return pd.DataFrame(rows)


def evaluate_assignment(folder_table, fold_map, val_fold, test_fold):
    total = folder_table[STRATIFY_COLUMNS].mean().to_numpy()
    score = 0.0
    all_folds = set(range(5))

    for split_name, fold_ids in {
        "train": all_folds - {val_fold, test_fold},
        "val": {val_fold},
        "test": {test_fold},
    }.items():
        mask = folder_table["folder_id"].map(lambda x: fold_map[x] in fold_ids)
        subset = folder_table.loc[mask, STRATIFY_COLUMNS]
        if len(subset) == 0:
            return float("inf")
        weight = 1.0 if split_name == "train" else 2.0
        score += weight * float(np.abs(subset.mean().to_numpy() - total).sum())

    for col in STRATIFY_COLUMNS:
        positives = int(folder_table[col].sum())
        if positives >= 3:
            for fold in (val_fold, test_fold):
                mask = folder_table["folder_id"].map(lambda x: fold_map[x] == fold)
                if int(folder_table.loc[mask, col].sum()) == 0:
                    score += 100.0

    return score


def choose_assignment(folder_table, seed):
    X = folder_table.index.to_numpy().reshape(-1, 1)
    Y = folder_table[STRATIFY_COLUMNS].to_numpy(dtype=int)
    splitter = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_map = {}
    for fold_id, (_, test_idx) in enumerate(splitter.split(X, Y)):
        for idx in test_idx:
            fold_map[str(folder_table.iloc[idx]["folder_id"])] = fold_id

    best = None
    all_folds = set(range(5))
    for val_fold, test_fold in itertools.permutations(range(5), 2):
        train_folds = all_folds - {val_fold, test_fold}
        score = evaluate_assignment(folder_table, fold_map, val_fold, test_fold)
        if best is None or score < best[0]:
            best = (score, val_fold, test_fold)

    score, val_fold, test_fold = best
    assignment = {
        folder_id: (
            "val" if fold == val_fold else "test" if fold == test_fold else "train"
        )
        for folder_id, fold in fold_map.items()
    }
    return assignment, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/cv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    folder_table = build_folder_table(df)

    candidates = []
    for seed in range(args.seed, args.seed + 50):
        assignment, score = choose_assignment(folder_table, seed)
        candidates.append((score, seed, assignment))

    score, used_seed, assignment = min(candidates, key=lambda x: x[0])
    df = df.copy()
    df["split"] = df["folder_id"].map(assignment)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df[df["split"] == split].to_csv(args.output_dir / f"{split}.csv", index=False)

    folder_table["split"] = folder_table["folder_id"].map(assignment)
    folder_table.to_csv(args.output_dir / "folder_split.csv", index=False)

    stats = {
        "seed_requested": args.seed,
        "seed_selected": used_seed,
        "objective_score": float(score),
        "folders": folder_table["split"].value_counts().to_dict(),
        "images": df["split"].value_counts().to_dict(),
        "anatomy": {
            split: df.loc[df["split"] == split, "anatomical_region"].value_counts().to_dict()
            for split in ("train", "val", "test")
        },
    }
    (args.output_dir / "split_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
