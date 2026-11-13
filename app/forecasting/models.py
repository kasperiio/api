"""
Database tables for price forecasting.

Tables are created by ``Base.metadata.create_all`` at startup (main.py imports
this module before that call).
"""

from sqlalchemy import Boolean, Column, Float, Integer, String, Text, UniqueConstraint

from app.database import Base
from app.models import TZDateTime


class SpotPriceRaw(Base):
    """Day-ahead prices per bidding zone, EUR/MWh without VAT."""

    __tablename__ = "spot_prices_raw"

    timestamp = Column(TZDateTime, primary_key=True)
    zone = Column(String(8), primary_key=True)
    price_eur_mwh = Column(Float, nullable=False)
    resolution_minutes = Column(Integer, nullable=False, default=60)


class DriverData(Base):
    """Latest known value of a driver series for an hour (long format)."""

    __tablename__ = "driver_data"

    timestamp = Column(TZDateTime, primary_key=True)
    series = Column(String(64), primary_key=True)
    value = Column(Float, nullable=False)
    source = Column(String(32), nullable=False)
    updated_at = Column(TZDateTime, nullable=False)


class ForecastSnapshot(Base):
    """What a forecast said for an hour at the moment it was captured."""

    __tablename__ = "forecast_snapshots"

    issued_at = Column(TZDateTime, primary_key=True)
    timestamp = Column(TZDateTime, primary_key=True)
    series = Column(String(64), primary_key=True)
    value = Column(Float, nullable=False)


class Prediction(Base):
    """A price prediction for one target hour, later scored against the actual."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("issued_at", "target_ts", "model_version", name="uq_prediction"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    issued_at = Column(TZDateTime, nullable=False, index=True)
    target_ts = Column(TZDateTime, nullable=False, index=True)
    horizon_hours = Column(Integer, nullable=False)
    model_version = Column(String(64), nullable=False, index=True)
    p10 = Column(Float, nullable=False)
    p50 = Column(Float, nullable=False)
    p90 = Column(Float, nullable=False)
    p10_corrected = Column(Float, nullable=True)
    p50_corrected = Column(Float, nullable=True)
    p90_corrected = Column(Float, nullable=True)
    actual = Column(Float, nullable=True)
    is_backtest = Column(Boolean, nullable=False, default=False)


class ModelRecord(Base):
    """Registry of trained models; exactly one row has is_active=True."""

    __tablename__ = "model_registry"

    version = Column(String(64), primary_key=True)
    created_at = Column(TZDateTime, nullable=False)
    train_start = Column(TZDateTime, nullable=False)
    train_end = Column(TZDateTime, nullable=False)
    path = Column(String(255), nullable=False)
    holdout_mae = Column(Float, nullable=True)
    is_active = Column(Boolean, nullable=False, default=False)
    metrics_json = Column(Text, nullable=True)
    features_json = Column(Text, nullable=True)
