from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, RootModel, validator

from app.models import ElectricityPrice


class PricePoint(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp for the price")
    price: Optional[float] = Field(..., description="Electricity price (null if unavailable from all providers)")


class ElectricityPriceResponse(RootModel[List[PricePoint]]):
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        from_attributes=True,
        json_schema_extra={
            "example": [
                {"timestamp": "2024-01-01T12:00:00+00:00", "price": 50.0},
                {"timestamp": "2024-01-01T12:15:00+00:00", "price": 45.0}
            ]
        },
    )

    @classmethod
    def from_db_model_list(
            cls,
            db_models: List['ElectricityPrice']
    ) -> 'ElectricityPriceResponse':
        return cls([
            PricePoint(timestamp=model.timestamp, price=model.price)
            for model in db_models
        ])


class ElectricityLatestResponse(BaseModel):
    current: PricePoint
    prices: List[PricePoint]

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        from_attributes=True,
        json_schema_extra={
            "example": {
                "current": {"timestamp": "2024-01-01T12:15:00+00:00", "price": 47.5},
                "prices": [
                    {"timestamp": "2024-01-01T12:00:00+00:00", "price": 50.0},
                    {"timestamp": "2024-01-01T12:15:00+00:00", "price": 47.5}
                ]
            }
        },
    )

    @classmethod
    def from_db_models(
        cls,
        db_models: List['ElectricityPrice'],
        current_model: 'ElectricityPrice'
    ) -> 'ElectricityLatestResponse':
        return cls(
            current=PricePoint(timestamp=current_model.timestamp, price=current_model.price),
            prices=[PricePoint(timestamp=m.timestamp, price=m.price) for m in db_models]
        )
