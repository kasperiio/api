from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, RootModel, validator
from zoneinfo import ZoneInfo


class PriceData(BaseModel):
    """Schema for price data at a specific time."""

    price: float = Field(..., description="Electricity price")
    price_daily_average_ratio: float = Field(
        ...,
        description="Ratio of price to daily average",
        ge=0,  # Must be non-negative
    )

    @validator('price_daily_average_ratio')
    def validate_ratio(cls, v: float) -> float:
        """Ensure ratio is reasonable."""
        if v < 0:
            raise ValueError("Price ratio cannot be negative")
        if v > 10:  # Arbitrary reasonable maximum
            raise ValueError("Price ratio seems unreasonably high")
        return v


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
                    "price_daily_average_ratio": 1.1,
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

    @validator('price')
    def price_must_be_reasonable(cls, v: float) -> float:
        """Validate price is within reasonable bounds."""
        if v < -1000:  # Adjust these bounds based on your needs
            raise ValueError("Price seems unreasonably low")
        if v > 10000:
            raise ValueError("Price seems unreasonably high")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2024-01-01T12:00:00+00:00",
                "price": 50.0
            }
        }
    )