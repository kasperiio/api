from datetime import datetime
from typing import Dict, List
from pydantic import ConfigDict, RootModel, BaseModel

from app import models


class PriceData(BaseModel):
    price: float
    price_daily_average_ratio: float


class ElectricityPrice(RootModel[Dict[datetime, PriceData]]):
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        from_attributes=True,
        json_schema_extra={
            "example": {
                "2023-10-01T12:00:00": {
                    "price": 50.0,
                    "price_daily_average_ratio": 1.2,
                },
                "2023-10-01T13:00:00": {
                    "price": 45.0,
                    "price_daily_average_ratio": 1.1,
                },
                "2023-10-01T14:00:00": {
                    "price": 55.0,
                    "price_daily_average_ratio": 1.3,
                },
            }
        },
    )

    @classmethod
    def from_db_model_list(
        cls, db_models: List[models.ElectricityPrice]
    ) -> "ElectricityPrice":
        data_dict = {
            db_model.timestamp: PriceData(
                price=db_model.price,
                price_daily_average_ratio=db_model.price_daily_average_ratio,
            )
            for db_model in db_models
        }
        return cls(data_dict)
