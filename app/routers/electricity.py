from datetime import datetime, timezone, timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for Python < 3.9 or Windows
    from dateutil.tz import gettz as ZoneInfo

from app import schemas, models
from app.crud import electricity as crud
from app.database import get_db

router = APIRouter(
    prefix="",
    tags=["electricity"]
)

from app.logging_config import logger

def convert_to_timezone(prices: List[models.ElectricityPrice], target_tz: str = "UTC") -> List[models.ElectricityPrice]:
    """Convert UTC prices to target timezone."""
    if target_tz == "UTC":
        return prices

    try:
        tz = ZoneInfo(target_tz)

        # Modify the timestamps in place (since we're returning the response, not saving to DB)
        for price in prices:
            # Ensure timestamp is UTC-aware first
            utc_timestamp = price.timestamp
            if utc_timestamp.tzinfo is None:
                utc_timestamp = utc_timestamp.replace(tzinfo=timezone.utc)

            # Convert to target timezone
            price.timestamp = utc_timestamp.astimezone(tz)

        return prices
    except Exception as e:
        # If timezone conversion fails, return original prices
        logger.error("Timezone conversion failed: %s", e)
        return prices


def ensure_utc(dt: datetime, source_timezone: str = "UTC") -> datetime:
    """
    Ensure datetime is in UTC.

    Args:
        dt: The datetime to convert
        source_timezone: The timezone to assume for naive datetimes
    """
    if dt.tzinfo is None:
        # If naive datetime and source timezone is not UTC, treat as source timezone
        if source_timezone != "UTC":
            try:
                tz = ZoneInfo(source_timezone)
                # Make datetime aware in source timezone, then convert to UTC
                return dt.replace(tzinfo=tz).astimezone(timezone.utc)
            except Exception as e:
                logger.warning("Invalid timezone %s, treating naive datetime as UTC: %s", source_timezone, e)
                return dt.replace(tzinfo=timezone.utc)
        else:
            # Assume naive datetime is UTC
            return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@router.get("/prices", response_model=schemas.ElectricityPriceResponse)
async def get_electricity_prices(
        start_date: datetime = Query(
            example=datetime.now(),
        ),
        end_date: datetime = Query(
            example=datetime.now(),
        ),
        timezone_str: str = Query("UTC", description="Timezone for input dates (if naive) and response (e.g., 'Europe/Helsinki', 'UTC')"),
        db: Session = Depends(get_db),
):
    """
    Fetches electricity prices for a given time period.

    If start_date/end_date are naive (no timezone), they will be interpreted
    as being in the timezone_str timezone before conversion to UTC for database query.
    """
    # Convert input dates to UTC, treating naive datetimes as being in timezone_str
    start_date_utc = ensure_utc(start_date, timezone_str)
    end_date_utc = ensure_utc(end_date, timezone_str)

    # Query the database for prices (all in UTC) using async CRUD
    prices = await crud.get_electricity_prices(db, start_date_utc, end_date_utc)

    # Convert timestamps to requested timezone
    prices = convert_to_timezone(prices, timezone_str)

    return schemas.ElectricityPriceResponse.from_db_model_list(prices)


@router.get("/cheapest", response_model=schemas.ElectricityPriceResponse)
async def get_cheapest_intervals_today(
        amount: int = Query(6, description="Number of cheapest 15-minute intervals to find (1-96)"),
        consecutive: bool = Query(False, description="Should the intervals be consecutive"),
        timezone_str: str = Query("UTC", description="Timezone for 'today' calculation and response (e.g., 'Europe/Helsinki', 'UTC')"),
        db: Session = Depends(get_db),
):
    """Fetches the cheapest electricity prices for a given amount of 15-minute intervals."""
    if amount > 96 or amount < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Amount must be between 1 and 96"
        )

    # Calculate "today" in the requested timezone, then convert to UTC for database query
    try:
        if timezone_str == "UTC":
            tz = timezone.utc
        else:
            tz = ZoneInfo(timezone_str)

        # Get current date in the specified timezone
        current_date = datetime.now(tz).date()

        # Define start and end of day in the specified timezone
        start_of_day = datetime.combine(current_date, datetime.min.time()).replace(tzinfo=tz)
        end_of_day = datetime.combine(current_date, datetime.max.time()).replace(tzinfo=tz)

        # Convert to UTC for database query
        start_of_day_utc = start_of_day.astimezone(timezone.utc)
        end_of_day_utc = end_of_day.astimezone(timezone.utc)

    except Exception:
        # Fallback to UTC if timezone is invalid
        current_date = datetime.now(timezone.utc).date()
        start_of_day_utc = datetime.combine(current_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_of_day_utc = datetime.combine(current_date, datetime.max.time()).replace(tzinfo=timezone.utc)

    db_prices = await crud.get_electricity_prices(db, start_of_day_utc, end_of_day_utc)

    if not db_prices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No electricity prices found for today"
        )

    if consecutive:
        # Find the cheapest consecutive 15-minute intervals using extracted data
        cheapest = find_consecutive_cheapest_intervals(db_prices, amount)
    else:
        # Get the cheapest non-consecutive intervals
        cheapest = sorted(db_prices, key=lambda x: x.price)[:amount]
        # cheapest_hours.sort(key=lambda x: x.timestamp)

    # Convert timestamps to requested timezone
    cheapest = convert_to_timezone(cheapest, timezone_str)

    return schemas.ElectricityPriceResponse.from_db_model_list(cheapest)

@router.get("/latest", response_model=schemas.ElectricityLatestResponse)
async def get_latest_electricity_prices(
        timezone_str: str = Query("UTC", description="Response timezone (e.g., 'Europe/Helsinki', 'UTC')"),
        db: Session = Depends(get_db),
):
    """Fetch todays and tomorrows prices and include the current 15-minute price point."""
    now_utc = datetime.now(timezone.utc)
    normalized_time = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    # Query the database for the latest price (UTC) using async CRUD
    end_time = normalized_time + timedelta(days=2) - timedelta(minutes=15)
    db_prices = await crud.get_electricity_prices(db, normalized_time, end_time)

    if not db_prices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Latest price not available"
        )

    # Determine current 15-minute slot (floor) in UTC
    current_slot = now_utc.replace(minute=(now_utc.minute // 15) * 15, second=0, microsecond=0)

    # Find index of current item (UTC) or fallback to latest past
    current_idx = None
    for i, p in enumerate(db_prices):
        if p.timestamp == current_slot:
            current_idx = i
            break
    if current_idx is None:
        # Fallback: index of most recent past timestamp
        last_idx = None
        for i, p in enumerate(db_prices):
            if p.timestamp <= now_utc:
                last_idx = i
            else:
                break
        current_idx = last_idx if last_idx is not None else 0

    # Convert timestamps to requested timezone (in-place)
    db_prices = convert_to_timezone(db_prices, timezone_str)
    current_converted = db_prices[current_idx]

    return schemas.ElectricityLatestResponse.from_db_models(
        db_prices,
        current_converted,
    )

def find_consecutive_cheapest_intervals(
        prices: List[models.ElectricityPrice],
        count: int
) -> List[models.ElectricityPrice]:
    """Find the cheapest consecutive 15-minute intervals from a list of prices."""
    if len(prices) < count:
        return prices

    # Ensure prices are sorted by timestamp
    prices.sort(key=lambda x: x.timestamp)

    min_total = float('inf')
    cheapest_block = None

    # Find the consecutive block with lowest total price
    for i in range(len(prices) - count + 1):
        block = prices[i:i + count]
        total = sum(p.price for p in block)

        if total < min_total:
            min_total = total
            cheapest_block = block

    return cheapest_block or []


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
