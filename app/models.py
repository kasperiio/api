from datetime import datetime, timedelta, timezone
from typing import Optional
from sqlalchemy import Column, Float, case, types
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property

from app.database import Base


class UTCDateTime(types.TypeDecorator):
    """Custom DateTime type that enforces UTC timezone."""

    impl = types.DateTime
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect) -> Optional[datetime]:
        if value is None:
            return None

        if value.tzinfo is None:
            raise ValueError("Naive datetime not allowed. Please provide timezone-aware datetime.")

        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Optional[datetime], dialect) -> Optional[datetime]:
        if value is not None:
            return value.replace(tzinfo=timezone.utc)
        return None


class ElectricityPrice(Base):
    """Model for storing electricity prices."""

    __tablename__ = "electricity_prices"

    timestamp = Column(UTCDateTime, index=True, primary_key=True)
    price = Column(Float, nullable=False)

    @hybrid_property
    def price_daily_average_ratio(self) -> float:
        """Calculate ratio of price to daily average.

        Returns:
            float: Price divided by daily average. Returns 1 if no data or zero average.
        """
        day_start = self.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        session = Session.object_session(self)
        if session is None:
            return 1.0

        day_prices = (
            session.query(ElectricityPrice)
            .filter(
                ElectricityPrice.timestamp >= day_start.astimezone(timezone.utc),
                ElectricityPrice.timestamp < day_end.astimezone(timezone.utc)
            )
            .all()
        )

        if not day_prices:
            return 1.0

        day_average = sum(p.price for p in day_prices) / len(day_prices)

        if day_average == 0:
            return 1.0
        elif day_average < 0:
            return self.price / abs(day_average)
        else:
            return self.price / day_average

    @price_daily_average_ratio.expression
    def price_daily_average_ratio(cls):
        """SQL expression for calculating price_daily_average_ratio."""
        daily_avg = func.avg(cls.price).over(
            partition_by=func.date(cls.timestamp)
        )

        return case(
            (daily_avg == 0, 1.0),
            (daily_avg < 0, cls.price / func.abs(daily_avg)),
            else_=cls.price / daily_avg
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ElectricityPrice):
            return NotImplemented
        return self.timestamp == other.timestamp

    def __repr__(self) -> str:
        return (
            f"ElectricityPrice(timestamp={self.timestamp.isoformat()}, "
            f"price={self.price:.2f})"
        )