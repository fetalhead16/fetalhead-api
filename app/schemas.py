from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class MeasurementModel(BaseModel):
    label: str
    value: float
    unit: str


class CalibrationModel(BaseModel):
    pixel_spacing_mm: Optional[float] = None
    source: str
    absolute_measurements: bool


class QualityModel(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)
    contour_points: int
    fit_score: float = Field(..., ge=0.0, le=1.0)
    center_offset_px: float


class AssessmentModel(BaseModel):
    classifier_mode: str
    status: str
    summary: str
    notes: List[str]


class PreviewModel(BaseModel):
    original: str
    preprocessed: str
    mask: str
    overlay: str


class AnalysisResponse(BaseModel):
    filename: str
    image_size: List[int]
    calibration: CalibrationModel
    quality: QualityModel
    measurements: dict[str, MeasurementModel]
    assessment: AssessmentModel
    notes: List[str]
    previews: PreviewModel


class RegistrationRequest(BaseModel):
    name: str
    email: str
    college: Optional[str] = None
    role: Optional[str] = None
    message: Optional[str] = None


class RegistrationResponse(BaseModel):
    success: bool
    message: str
