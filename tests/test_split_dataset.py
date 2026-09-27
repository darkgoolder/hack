from __future__ import annotations

import pandas as pd

from app.data.split_dataset import validate_manifest, _folder_matrix


def test_manifest_validation_and_folder_aggregation() -> None:
    df = pd.DataFrame(
        [
            {"folder_id": "1", "image_path": "1/spine.dcm", "anatomical_region": "spine", "laterality": "", "spine_layout": 0, "spine_axis": 1, "spine_artifact": 0, "hip_position_rotation": None, "hip_roi": None},
            {"folder_id": "1", "image_path": "1/right_hip.dcm", "anatomical_region": "right_hip", "laterality": "right", "spine_layout": None, "spine_axis": None, "spine_artifact": None, "hip_position_rotation": 1, "hip_roi": 0},
            {"folder_id": "2", "image_path": "2/spine.dcm", "anatomical_region": "spine", "laterality": "", "spine_layout": 0, "spine_axis": 0, "spine_artifact": 0, "hip_position_rotation": None, "hip_roi": None},
        ]
    )
    validate_manifest(df)
    groups = _folder_matrix(df)
    row1 = groups.loc[groups["folder_id"] == "1"].iloc[0]
    assert row1["spine_axis"] == 1
    assert row1["hip_position_rotation"] == 1
