from datetime import datetime, timezone, timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for Python < 3.9 or Windows
    from dateutil.tz import gettz as ZoneInfo

from app import schemas, models
from app.crud import electricity as crud
from app.database import get_db

router = APIRouter(
    prefix="/electricity",
    tags=["electricity"]
)


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
        print(f"Timezone conversion failed: {e}")
        return prices


def ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is in UTC."""
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@router.get("/price_at", response_model=schemas.ElectricityPriceResponse)
def get_electricity_price(
        start_date: datetime = Query(
            example=datetime.now(),
        ),
        end_date: datetime = Query(
            example=datetime.now(),
        ),
        timezone_str: str = Query("UTC", description="Response timezone (e.g., 'Europe/Helsinki', 'UTC')"),
        db: Session = Depends(get_db),
):
    """Fetches electricity prices for a given time period."""
    # Convert input dates to UTC
    start_date_utc = ensure_utc(start_date)
    end_date_utc = ensure_utc(end_date)

    if start_date_utc == end_date_utc:
        end_date_utc = end_date_utc.replace(hour=23)

    # Query the database for prices (all in UTC)
    prices = crud.get_electricity_prices(db, start_date_utc, end_date_utc)

    # Convert timestamps to requested timezone
    prices = convert_to_timezone(prices, timezone_str)

    return schemas.ElectricityPriceResponse.from_db_model_list(prices)


@router.get("/current_price", response_model=schemas.ElectricityPriceResponse)
def get_current_electricity_price(
        timezone_str: str = Query("UTC", description="Response timezone (e.g., 'Europe/Helsinki', 'UTC')"),
        db: Session = Depends(get_db),
):
    """Fetches the current electricity price."""
    current_datetime = datetime.now(timezone.utc)
    normalized_time = current_datetime.replace(minute=0, second=0, microsecond=0)

    # Query the database for the current price (UTC)
    prices = crud.get_electricity_prices(db, normalized_time, normalized_time)

    if not prices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current price not available"
        )

    # Convert timestamps to requested timezone
    prices = convert_to_timezone(prices, timezone_str)

    return schemas.ElectricityPriceResponse.from_db_model_list(prices)


@router.get("/cheapest_hours", response_model=schemas.ElectricityPriceResponse)
def get_cheapest_hours_today(
        amount: int = Query(6, description="Number of cheapest hours to find"),
        consecutive: bool = Query(False, description="Should the hours be consecutive"),
        timezone_str: str = Query("UTC", description="Timezone for 'today' calculation and response (e.g., 'Europe/Helsinki', 'UTC')"),
        db: Session = Depends(get_db),
):
    """Fetches the cheapest electricity prices for a given amount."""
    if amount > 24 or amount < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Amount must be between 1 and 24"
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

    db_prices = crud.get_electricity_prices(db, start_of_day_utc, end_of_day_utc)

    if not db_prices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No electricity prices found for today"
        )

    if consecutive:
        # Find the cheapest consecutive hours using extracted data
        cheapest_hours = find_consecutive_cheapest_hours(db_prices, amount)
    else:
        # Get the cheapest non-consecutive hours
        cheapest_hours = sorted(db_prices, key=lambda x: x.price)[:amount]
        cheapest_hours.sort(key=lambda x: x.timestamp)

    # Convert timestamps to requested timezone
    cheapest_hours = convert_to_timezone(cheapest_hours, timezone_str)

    return schemas.ElectricityPriceResponse.from_db_model_list(cheapest_hours)


def find_consecutive_cheapest_hours(
        prices: List[models.ElectricityPrice],
        count: int
) -> List[models.ElectricityPrice]:
    """Find the cheapest consecutive hours from a list of prices."""
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