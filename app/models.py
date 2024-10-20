from datetime import timedelta, timezone
import os
import pytz
from sqlalchemy import types
from sqlalchemy import Column, Float
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property

from .database import Base


class UTCDateTime(types.TypeDecorator):

    impl = types.DateTime
    cache_ok = True

    def process_bind_param(self, value, _):
        if value is None:
            return
        if value.utcoffset() is None:
            raise ValueError("Got naive datetime while timezone-aware is expected")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, _):
        if value is not None:
            return value.replace(tzinfo=pytz.utc)


class ElectricityPrice(Base):
    __tablename__ = "electricity_prices"

    timestamp = Column(UTCDateTime, index=True, primary_key=True)
    price = Column(Float)

    @hybrid_property
    def price_daily_average_ratio(self):
        day_start = self.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(hours=23)

        session = Session.object_session(self)
        if session is None:
            return 1

        day_prices = session.query(ElectricityPrice).filter(
            ElectricityPrice.timestamp >= day_start.astimezone(pytz.UTC),
            ElectricityPrice.timestamp < day_end.astimezone(pytz.UTC)
        ).all()

        if not day_prices:
            return 1

        negative = all(p.price < 0 for p in day_prices)
        day_average = sum(p.price for p in day_prices) / len(day_prices)
        ratio = self.price / day_average if day_average else 1
        return -ratio if negative else ratio

    @price_daily_average_ratio.expression
    def price_daily_average_ratio(cls):
        tz = pytz.timezone(os.getenv("TZ", "UTC"))
        return (
            cls.price / 
            func.avg(cls.price).over(
                partition_by=func.date_trunc('day', func.timezone(tz.zone, cls.timestamp))
            )
        )

    def __eq__(self, other):
        return self.timestamp == other.timestamp

    def __repr__(self):
        return f"<ElectricityPrice(timestamp={self.timestamp}, price={self.price})>"
