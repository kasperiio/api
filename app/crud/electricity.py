from datetime import datetime, timedelta, timezone
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
    # Normalize to 15-minute boundaries
    def floor_to_quarter(dt: datetime) -> datetime:
        minute = (dt.minute // 15) * 15
        return dt.replace(minute=minute, second=0, microsecond=0)

    start_date = floor_to_quarter(start_date)
    end_date = floor_to_quarter(end_date)

    # Query existing prices from database using SQLAlchemy 2.0 select
    stmt = select(models.ElectricityPrice).where(
        models.ElectricityPrice.timestamp >= start_date,
        models.ElectricityPrice.timestamp <= end_date,
    ).order_by(models.ElectricityPrice.timestamp)

    result = db.execute(stmt)
    prices = result.scalars().all()

    # Check if we have all the 15-minute intervals we need (inclusive)
    quarters_diff = int((end_date - start_date).total_seconds() // 900)
    expected_intervals = {
        start_date + timedelta(minutes=15 * i)
        for i in range(quarters_diff + 1)
    }

    existing_intervals = {price.timestamp for price in prices}
    missing_intervals = expected_intervals - existing_intervals

    # Filter out future hours that providers won't have yet
    # Electricity price data availability:
    # - Before 13:00 UTC: Only current day data available
    # - After 13:00 UTC: Current day + next day data available (until ~21:00 UTC next day)
    current_time = datetime.now(timezone.utc)
    current_hour = current_time.hour

    if current_hour >= 13:
        # After 13:00 UTC: next day data is available until 21:00 UTC
        next_day = current_time.date() + timedelta(days=1)
        max_available_time = datetime.combine(
            next_day, datetime.min.time()
        ).replace(tzinfo=timezone.utc) + timedelta(hours=21)
    else:
        # Before 13:00 UTC: only current day data is available
        max_available_time = current_time.replace(
            hour=23, minute=59, second=59, microsecond=999999
        )

    fetchable_missing_intervals = {
        ts for ts in missing_intervals
        if ts <= max_available_time
    }

    if fetchable_missing_intervals:
        try:
            # Use async provider manager
            provider_manager = get_provider_manager()
            provider_prices = await provider_manager.get_electricity_price(
                min(fetchable_missing_intervals), max(fetchable_missing_intervals)
            )

            # Expand hourly provider data into 15-minute intervals when needed
            minutes_set = {p.timestamp.minute for p in provider_prices}
            is_hourly = len(provider_prices) > 0 and minutes_set == {0}

            expanded_prices = []
            if is_hourly:
                for p in provider_prices:
                    base = p.timestamp.replace(minute=0, second=0, microsecond=0)
                    for offset in (0, 15, 30, 45):
                        ts = base.replace(minute=offset)
                        expanded_prices.append(models.ElectricityPrice(timestamp=ts, price=p.price))
            else:
                expanded_prices = provider_prices

            # Filter out prices that were outside wanted range
            new_prices = [
                price for price in expanded_prices
                if price.timestamp in fetchable_missing_intervals
            ]

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