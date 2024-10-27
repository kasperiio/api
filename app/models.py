from datetime import timedelta, timezone
import os
import pytz
from sqlalchemy import types
from sqlalchemy import Column, Float, case
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

        day_average = sum(p.price for p in day_prices) / len(day_prices)
        if day_average == 0:
            return 1

        if day_average < 0:
            return self.price / abs(day_average)
        return self.price / day_average

    @price_daily_average_ratio.expression
    def price_daily_average_ratio(cls):
        tz = pytz.timezone(os.getenv("TZ", "UTC"))
        
        daily_avg = func.avg(cls.price).over(
            partition_by=func.date_trunc('day', func.timezone(tz.zone, cls.timestamp))
        )
        
        return case(
            # Handle zero division
            (daily_avg == 0, 1),
            # For negative daily average, divide by absolute value of average
            (daily_avg < 0, cls.price / func.abs(daily_avg)),
            # For positive daily average, normal ratio
            else_=cls.price / daily_avg
        )

    def __eq__(self, other):
        return self.timestamp == other.timestamp

    def __repr__(self):
        return f"<ElectricityPrice(timestamp={self.timestamp}, price={self.price})>"