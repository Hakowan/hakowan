"""Deterministic inspection, validation, framing, and observation workflows."""

from .framing import resolve_camera, turntable_cameras
from .inspection import AttributeSummary, DataSummary, inspect
from .observation import (
    AttributeVisibility,
    CameraState,
    LayerVisibility,
    Observation,
    ObservationError,
    OcclusionRecord,
    PixelHit,
    RegionSummary,
    SceneSummary,
    Snapshot,
    observe,
    snapshot,
)
from .validation import Diagnostic, ValidationError, ValidationReport, validate

__all__ = [
    "AttributeSummary",
    "AttributeVisibility",
    "CameraState",
    "DataSummary",
    "Diagnostic",
    "LayerVisibility",
    "Observation",
    "ObservationError",
    "OcclusionRecord",
    "PixelHit",
    "RegionSummary",
    "SceneSummary",
    "Snapshot",
    "ValidationError",
    "ValidationReport",
    "inspect",
    "observe",
    "resolve_camera",
    "snapshot",
    "turntable_cameras",
    "validate",
]
