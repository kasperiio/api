from datetime import datetime, timedelta
from typing import List
from sqlalchemy.orm import Session

from app.utils import external
from app import models

def get_electricity_prices(
    db: Session, start_date: datetime, end_date: datetime
) -> List[models.ElectricityPrice]:
    """
    Retrieves electricity prices from the database within the specified time range, will use external data if not found.
    Args:
        db (Session): The database session.
        start_date (datetime): The start date of the time range.
        end_date (datetime): The end date of the time range.
    Returns:
        List[models.ElectricityPrice]: A list of electricity prices within the specified time range.
    """
    # Remove minutes, seconds and microseconds from the dates
    start_date = start_date.replace(minute=0, second=0, microsecond=0)
    end_date = end_date.replace(minute=0, second=0, microsecond=0)

    # Query existing prices from database
    prices = (
        db.query(models.ElectricityPrice)
        .filter(
            models.ElectricityPrice.timestamp >= start_date,
            models.ElectricityPrice.timestamp <= end_date,
        )
        .all()
    )

    # Find missing hours
    existing_timestamps = set(price.timestamp for price in prices)
    all_hours = {
        start_date + timedelta(hours=i)
        for i in range(int((end_date - start_date).total_seconds() // 3600) + 1)
    }
    missing_hours = all_hours - existing_timestamps

    if not missing_hours:
        all_prices = prices
    else:
        try:
            # Fetch missing data from ENTSO-E API
            new_prices = external.get_electricity_price(
                min(missing_hours), max(missing_hours) + timedelta(hours=1)
            )

            # Filter out prices that were outside wanted range
            new_prices = [price for price in new_prices if price.timestamp in missing_hours]

            # Add new prices to the database
            if new_prices:
                for new_price in new_prices:
                    db.add(new_price)
                db.commit()

            # Merge existing and new prices
            all_prices = prices + new_prices
            all_prices.sort(key=lambda x: x.timestamp)
        except Exception as e:
            db.rollback()
            # If external API fails, return what we have from the database
            all_prices = prices
            all_prices.sort(key=lambda x: x.timestamp)

    return all_prices

# Optional: Keep the service class for future use
class ElectricityPriceService:
    def __init__(self, db: Session):
        self.db = db

    def get_electricity_prices(
        self, start_date: datetime, end_date: datetime
    ) -> List[models.ElectricityPrice]:
        return get_electricity_prices(self.db, start_date, end_date)