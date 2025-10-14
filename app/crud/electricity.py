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

    if missing_intervals:
        try:
            provider_manager = get_provider_manager()

            import logging
            logger = logging.getLogger("app")

            # Fetch the entire missing range in one request
            # The provider manager handles chunking internally (30-day chunks)
            min_ts = min(missing_intervals)
            max_ts = max(missing_intervals)

            logger.info(
                "Fetching %d missing intervals from %s to %s",
                len(missing_intervals), min_ts, max_ts
            )

            # Fetch data from provider
            # If the provider fails (network error, API down, etc.), this will raise ProviderError
            # and we won't insert NULL values - the error will propagate to the caller
            all_provider_prices = await provider_manager.get_electricity_price(min_ts, max_ts)

            # Expand hourly provider data into 15-minute intervals when needed
            minutes_set = {p.timestamp.minute for p in all_provider_prices}
            is_hourly = len(all_provider_prices) > 0 and minutes_set == {0}

            expanded_prices = []
            if is_hourly:
                for p in all_provider_prices:
                    base = p.timestamp.replace(minute=0, second=0, microsecond=0)
                    for offset in (0, 15, 30, 45):
                        ts = base.replace(minute=offset)
                        expanded_price = models.ElectricityPrice()
                        expanded_price.timestamp = ts
                        expanded_price.price = p.price
                        expanded_prices.append(expanded_price)
            else:
                expanded_prices = all_provider_prices

            # Filter to only include missing timestamps
            new_prices = [
                price for price in expanded_prices
                if price.timestamp in missing_intervals
            ]

            logger.info("Provider returned %d prices, %d are new", len(all_provider_prices), len(new_prices))

            # SQLite has a limit on SQL variables (default 999-32766)
            # With 2 params per row (timestamp, price), batch at 500 rows = 1000 params
            batch_size = 500

            # Bulk insert with SQLite's INSERT OR IGNORE to handle duplicates
            if new_prices:
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                total_inserted = 0

                for i in range(0, len(new_prices), batch_size):
                    batch = new_prices[i:i + batch_size]

                    # Convert to dict for bulk insert
                    price_dicts = [
                        {"timestamp": p.timestamp, "price": p.price}
                        for p in batch
                    ]

                    # Use INSERT OR IGNORE for SQLite
                    stmt = sqlite_insert(models.ElectricityPrice).values(price_dicts)
                    stmt = stmt.on_conflict_do_nothing(index_elements=['timestamp'])

                    db.execute(stmt)
                    total_inserted += len(batch)

                db.commit()
                logger.info("Inserted %d new prices into database", total_inserted)

            # Check if there are still missing intervals after successful fetch
            # This happens when the provider successfully returns data, but some timestamps
            # are genuinely missing from their database (e.g., ENTSO-E has incomplete data).
            # We insert NULL placeholders ONLY when the provider succeeded but returned incomplete data.
            # If the provider failed (network error, API down), the exception above will be raised
            # and we won't reach this code - no NULL values will be inserted.
            fetched_timestamps = {p.timestamp for p in new_prices}
            still_missing = missing_intervals - fetched_timestamps

            if still_missing:
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                logger.warning(
                    "Data unavailable for %d intervals from all providers, inserting NULL placeholders",
                    len(still_missing)
                )

                # Insert NULL prices for unavailable data to prevent repeated fetching
                null_prices = []
                for ts in still_missing:
                    null_price = models.ElectricityPrice()
                    null_price.timestamp = ts
                    null_price.price = None  # NULL indicates data unavailable
                    null_prices.append(null_price)

                # Insert NULL prices in batches
                if null_prices:
                    for i in range(0, len(null_prices), batch_size):
                        batch = null_prices[i:i + batch_size]
                        price_dicts = [
                            {"timestamp": p.timestamp, "price": p.price}
                            for p in batch
                        ]
                        stmt = sqlite_insert(models.ElectricityPrice).values(price_dicts)
                        stmt = stmt.on_conflict_do_nothing(index_elements=['timestamp'])
                        db.execute(stmt)

                    db.commit()
                    logger.info("Inserted %d NULL placeholders for unavailable data", len(null_prices))

                    # Add to new_prices for the final merge
                    new_prices.extend(null_prices)

            # Merge existing and new prices, using set to deduplicate
            all_prices = list(set(prices) | set(new_prices))
            all_prices.sort(key=lambda x: x.timestamp)

        except (ProviderError, Exception) as e:
            logger.error("Provider error: %s", str(e))
            db.rollback()
            # If all providers fail, return what we have from the database
            all_prices = list(prices)
            all_prices.sort(key=lambda x: x.timestamp)
    else:
        # Deduplicate prices from database (in case of DST transitions)
        all_prices = list(set(prices))
        all_prices.sort(key=lambda x: x.timestamp)

    return all_prices