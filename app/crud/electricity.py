from datetime import datetime, timedelta
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.providers import get_provider_manager, ProviderError
from app import models

async def get_electricity_prices(
    db: Session, start_date: datetime, end_date: datetime
) -> List[models.ElectricityPrice]:
    """
    Get electricity prices for the specified time range.

    Args:
        db: The database session.
        start_date: The start date of the time range.
        end_date: The end date of the time range.
    Returns:
        List of electricity prices within the specified time range.
    """
    # Remove minutes, seconds and microseconds from the dates
    start_date = start_date.replace(minute=0, second=0, microsecond=0)
    end_date = end_date.replace(minute=0, second=0, microsecond=0)

    # Query existing prices from database using SQLAlchemy 2.0 select
    stmt = select(models.ElectricityPrice).where(
        models.ElectricityPrice.timestamp >= start_date,
        models.ElectricityPrice.timestamp <= end_date,
    ).order_by(models.ElectricityPrice.timestamp)

    result = db.execute(stmt)
    prices = result.scalars().all()

    # Check if we have all the hours we need
    # Calculate the number of hours between start and end (inclusive)
    hours_diff = int((end_date - start_date).total_seconds() // 3600)
    expected_hours = {
        start_date + timedelta(hours=i)
        for i in range(hours_diff + 1)
    }

    existing_hours = {price.timestamp for price in prices}
    missing_hours = expected_hours - existing_hours

    if missing_hours:
        try:
            # Use async provider manager
            provider_manager = get_provider_manager()
            new_prices = await provider_manager.get_electricity_price(
                min(missing_hours), max(missing_hours)
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
        except (ProviderError, Exception):
            db.rollback()
            # If all providers fail, return what we have from the database
            all_prices = prices
            all_prices.sort(key=lambda x: x.timestamp)
    else:
        all_prices = prices

    return all_prices