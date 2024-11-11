from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel, ConfigDict, Field, RootModel, validator

from app.models import ElectricityPrice


class PriceData(BaseModel):
    """Schema for price data at a specific time."""

    price: float = Field(..., description="Electricity price")
    price_daily_average_ratio: float = Field(
        ...,
        description="Ratio of price to daily average"
    )


class ElectricityPriceResponse(RootModel[Dict[datetime, PriceData]]):
    """Response schema for electricity price endpoints."""

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        from_attributes=True,
        json_schema_extra={
            "example": {
                "2024-01-01T12:00:00+00:00": {
                    "price": 50.0,
                    "price_daily_average_ratio": 1.2,
                },
                "2024-01-01T13:00:00+00:00": {
                    "price": 45.0,
                    "price_daily_average_ratio": -0.9,
                }
            }
        },
    )

    @classmethod
    def from_db_model_list(
            cls,
            db_models: List['ElectricityPrice']
    ) -> 'ElectricityPriceResponse':
        """Convert database models to response schema.

        Args:
            db_models: List of database models

        Returns:
            ElectricityPriceResponse: Response schema instance
        """
        data_dict = {
            model.timestamp: PriceData(
                price=model.price,
                price_daily_average_ratio=model.price_daily_average_ratio,
            )
            for model in db_models
        }

        return cls(data_dict)


class ElectricityPriceCreate(BaseModel):
    """Schema for creating new price entries."""

    timestamp: datetime = Field(..., description="Timestamp for the price")
    price: float = Field(..., description="Electricity price")

    @validator('timestamp')
    def timestamp_must_be_aware(cls, v: datetime) -> datetime:
        """Ensure timestamp has timezone information."""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-01T12:00:00+00:00",
                "price": 50.0
            }
        }
    )