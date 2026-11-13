from datetime import timezone
from sqlalchemy import Column, Float
from sqlalchemy.types import TypeDecorator, DateTime

from app.database import Base


class TZDateTime(TypeDecorator):
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            if not value.tzinfo:
                raise TypeError("tzinfo is required")
            value = value.astimezone(timezone.utc).replace(
                tzinfo=None
            )
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = value.replace(tzinfo=timezone.utc)
        return value


class ElectricityPrice(Base):
    """Model for storing electricity prices."""

    __tablename__ = "electricity_prices"

    timestamp = Column(TZDateTime, index=True, primary_key=True)
    price = Column(Float, nullable=True)  # NULL indicates data unavailable from all providers


    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ElectricityPrice):
            return NotImplemented
        return self.timestamp == other.timestamp

    def __hash__(self) -> int:
        return hash(self.timestamp)

    def __repr__(self) -> str:
        return (
            f"ElectricityPrice(timestamp={self.timestamp.isoformat()}, "
            f"price={self.price:.2f})"
        )