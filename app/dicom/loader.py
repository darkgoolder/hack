from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


@dataclass(slots=True)
class DicomInfo:
    path: Path
    study_uid: str | None
    series_uid: str | None
    sop_instance_uid: str | None
    modality: str | None
    manufacturer: str | None
    manufacturer_model: str | None
    series_description: str | None
    study_description: str | None
    body_part_examined: str | None
    laterality: str | None
    rows: int | None
    columns: int | None
    pixel_spacing: tuple[float, float] | None
    projection: str | None


def _get_str(ds: Any, key: str) -> str | None:
    value = getattr(ds, key, None)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _get_spacing(ds: Any) -> tuple[float, float] | None:
    value = getattr(ds, "PixelSpacing", None)
    if value is None or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def infer_projection(ds: Any) -> str | None:
    text = " ".join(
        filter(
            None,
            [
                _get_str(ds, "SeriesDescription"),
                _get_str(ds, "StudyDescription"),
                _get_str(ds, "ProtocolName"),
                _get_str(ds, "ImageType"),
            ],
        )
    ).lower()
    if "lateral" in text or "lat" in text:
        return "lateral"
    if "ap" in text or "pa" in text:
        return "AP"
    return None


def read_dicom(path: Path, stop_before_pixels: bool = False) -> Any:
    return pydicom.dcmread(path, stop_before_pixels=stop_before_pixels, force=False)


def get_dicom_info(path: Path) -> DicomInfo:
    ds = read_dicom(path, stop_before_pixels=True)
    rows = getattr(ds, "Rows", None)
    cols = getattr(ds, "Columns", None)
    return DicomInfo(
        path=path,
        study_uid=_get_str(ds, "StudyInstanceUID"),
        series_uid=_get_str(ds, "SeriesInstanceUID"),
        sop_instance_uid=_get_str(ds, "SOPInstanceUID"),
        modality=_get_str(ds, "Modality"),
        manufacturer=_get_str(ds, "Manufacturer"),
        manufacturer_model=_get_str(ds, "ManufacturerModelName"),
        series_description=_get_str(ds, "SeriesDescription"),
        study_description=_get_str(ds, "StudyDescription"),
        body_part_examined=_get_str(ds, "BodyPartExamined"),
        laterality=_get_str(ds, "Laterality"),
        rows=int(rows) if rows is not None else None,
        columns=int(cols) if cols is not None else None,
        pixel_spacing=_get_spacing(ds),
        projection=infer_projection(ds),
    )


def load_pixel_array(path: Path) -> np.ndarray:
    ds = read_dicom(path, stop_before_pixels=False)
    pixel_array = ds.pixel_array
    return np.asarray(pixel_array)
