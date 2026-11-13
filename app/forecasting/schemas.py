"""Pydantic response schemas for the forecast endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    timestamp: datetime = Field(..., description="Start of the target hour")
    p10: float = Field(..., description="10th percentile")
    p50: float = Field(..., description="Median forecast (bias-corrected when available)")
    p90: float = Field(..., description="90th percentile")
    known_price: Optional[float] = Field(None, description="Actual FI day-ahead price if already published")


class ForecastResponse(BaseModel):
    issued_at: datetime
    model_version: str
    unit: str = Field(..., description="c_kwh_vat (c/kWh incl. VAT) or eur_mwh (EUR/MWh excl. VAT)")
    timezone: str
    points: List[ForecastPoint]

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "issued_at": "2026-11-13T14:45:00+02:00",
                "model_version": "lgbm-20261108T010000Z",
                "unit": "c_kwh_vat",
                "timezone": "Europe/Helsinki",
                "points": [{"timestamp": "2026-11-14T00:00:00+02:00", "p10": 2.1,
                            "p50": 4.3, "p90": 7.9, "known_price": 4.05}],
            }
        },
    }


class HorizonAccuracy(BaseModel):
    horizon_day: int
    mae: Optional[float] = Field(None, description="MAE of the served (bias-corrected) p50, EUR/MWh")
    mae_raw: Optional[float] = Field(None, description="MAE of the raw model p50, EUR/MWh")
    n: int


class DriftStatus(BaseModel):
    drift: bool
    live_mae_14d: Optional[float] = None
    n: int
    min_samples: int
    holdout_mae: Optional[float] = None
    threshold: Optional[float] = None
    model_version: Optional[str] = None

    model_config = {"protected_namespaces": ()}


class AccuracyResponse(BaseModel):
    unit: str = "eur_mwh"
    source: Optional[str] = Field(None, description="'live', 'backtest' or null when nothing is scored yet")
    windows: Dict[str, List[HorizonAccuracy]] = Field(..., description="Keyed by window, e.g. '14d', '28d'")
    model_version: Optional[str] = None
    holdout_mae: Optional[float] = None
    drift: DriftStatus

    model_config = {"protected_namespaces": ()}


class DriverContribution(BaseModel):
    feature: str
    value: Optional[float] = Field(None, description="Feature value at the target hour")
    contribution: float = Field(..., description="Contribution to p50, EUR/MWh")


class DriversResponse(BaseModel):
    timestamp: datetime
    issued_at: datetime
    model_version: str
    unit: str = "eur_mwh"
    base_value: Optional[float] = None
    p50: Optional[float] = Field(None, description="Raw model p50 for the hour, EUR/MWh")
    contributions: List[DriverContribution]

    model_config = {"protected_namespaces": ()}


class ModelInfo(BaseModel):
    version: str
    created_at: datetime
    train_start: datetime
    train_end: datetime
    holdout_mae: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None


class CoverageRange(BaseModel):
    min: Optional[datetime] = None
    max: Optional[datetime] = None
    rows: int = 0


class StatusResponse(BaseModel):
    scheduler: Dict[str, Any]
    active_model: Optional[ModelInfo] = None
    coverage: Dict[str, CoverageRange]
    latest_issued_at: Optional[datetime] = None
