import pandas as pd
from app.data.split_dataset import build_folder_table


def test_folder_table_has_one_row_per_folder():
    df = pd.DataFrame([
        {"folder_id":"A","anatomical_region":"spine","spine_layout":1,"spine_axis":0,"spine_artifact":0,"hip_position_rotation":None,"hip_roi":None},
        {"folder_id":"A","anatomical_region":"right_hip","spine_layout":None,"spine_axis":None,"spine_artifact":None,"hip_position_rotation":0,"hip_roi":1},
        {"folder_id":"B","anatomical_region":"left_hip","spine_layout":None,"spine_axis":None,"spine_artifact":None,"hip_position_rotation":0,"hip_roi":0},
    ])
    out = build_folder_table(df)
    assert len(out) == 2
    a = out[out.folder_id == "A"].iloc[0]
    assert a.has_spine == 1
    assert a.has_right_hip == 1
    assert a.has_left_hip == 0
    assert a.spine_layout_positive == 1
    assert a.hip_roi_positive == 1
