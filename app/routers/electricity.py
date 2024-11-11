from datetime import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
import pytz
import os

from app import schemas, models
from app.crud import electricity as crud
from app.database import get_db
from app.utils.date_utils import convert_date_tz

router = APIRouter(
    prefix="/electricity",
    tags=["electricity"]
)


def convert_to_timezone(prices: List[models.ElectricityPrice]) -> List[models.ElectricityPrice]:
    """Convert prices timestamps to configured timezone."""
    if os.getenv("TZ") != "UTC":
        tz = pytz.timezone(os.getenv("TZ", "UTC"))
        for price in prices:
            price.timestamp = price.timestamp.astimezone(tz)
    return prices


@router.get("/price_at", response_model=schemas.ElectricityPriceResponse)
def get_electricity_price(
        start_date: datetime = Query(
            example=datetime.now(),
        ),
        end_date: datetime = Query(
            example=datetime.now(),
        ),
        db: Session = Depends(get_db),
):
    """Fetches electricity prices for a given time period."""
    # Convert the input dates timezones to UTC
    start_date_dt = convert_date_tz(start_date)
    end_date_dt = convert_date_tz(end_date)

    if start_date_dt == end_date_dt:
        end_date_dt = end_date_dt.replace(hour=23)

    if (end_date_dt - start_date_dt).days > 7:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Date range cannot exceed 7 days"
        )

    # Query the database for prices
    prices = crud.get_electricity_prices(db, start_date_dt, end_date_dt)

    # Convert timestamps to the specified timezone if needed
    prices = convert_to_timezone(prices)

    return schemas.ElectricityPriceResponse.from_db_model_list(prices)


@router.get("/current_price", response_model=schemas.ElectricityPriceResponse)
def get_current_electricity_price(
        db: Session = Depends(get_db),
):
    """Fetches the current electricity price."""
    current_datetime = datetime.now(pytz.utc)
    normalized_time = current_datetime.replace(minute=0, second=0, microsecond=0)

    # Query the database for the current price
    prices = crud.get_electricity_prices(db, normalized_time, normalized_time)

    if not prices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current price not available"
        )

    # Convert timestamps to the specified timezone if needed
    prices = convert_to_timezone(prices)

    return schemas.ElectricityPriceResponse.from_db_model_list(prices)


@router.get("/cheapest_hours", response_model=schemas.ElectricityPriceResponse)
def get_cheapest_hours_today(
        amount: int = Query(6, description="Number of cheapest hours to find"),
        consecutive: bool = Query(False, description="Should the hours be consecutive"),
        db: Session = Depends(get_db),
):
    """Fetches the cheapest electricity prices for a given amount."""
    if amount > 24 or amount < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Amount must be between 1 and 24"
        )

    # Define the timezone
    tz = pytz.timezone(os.getenv("TZ", "UTC"))

    # Fetch the current date in the specified timezone
    current_date = datetime.now(tz).date()

    # Define the start and end of the current day in the specified timezone
    start_of_day = tz.localize(datetime.combine(current_date, datetime.min.time()))
    end_of_day = tz.localize(datetime.combine(current_date, datetime.max.time()))

    # Convert the start and end of the day to UTC
    start_of_day_utc = start_of_day.astimezone(pytz.utc)
    end_of_day_utc = end_of_day.astimezone(pytz.utc)

    # Query the database for prices within the current day in UTC
    db_prices = crud.get_electricity_prices(db, start_of_day_utc, end_of_day_utc)

    if not db_prices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No electricity prices found for today"
        )

    if consecutive:
        # Find the cheapest consecutive hours
        cheapest_hours = find_consecutive_cheapest_hours(db_prices, amount)
    else:
        # Get the cheapest non-consecutive hours
        cheapest_hours = sorted(db_prices, key=lambda x: x.price)[:amount]
        cheapest_hours.sort(key=lambda x: x.timestamp)

    # Convert timestamps to the specified timezone if needed
    cheapest_hours = convert_to_timezone(cheapest_hours)

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