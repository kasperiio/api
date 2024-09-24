from datetime import datetime
import os

import pytz
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from .. import schemas
from ..crud import electricity as crud
from ..database import get_db
from ..utils.date_utils import convert_date_tz

router = APIRouter()


def convert_to_timezone(prices):
    """
    Convert a list of electricity prices to a specified timezone.

    Args:
        prices (List[schemas.ElectricityPrice]): The list of electricity prices to be converted.
        tz_str (str): The timezone string to convert the prices to.

    Returns:
        List[schemas.ElectricityPrice]: The list of electricity prices converted to the specified timezone.
    """
    if os.getenv("TZ") != "UTC":
        tz = pytz.timezone(os.getenv("TZ"))
        for price in prices:
            price.timestamp = price.timestamp.astimezone(tz)
    return prices


@router.get("/price_at", response_model=schemas.ElectricityPrice)
def get_electricity_price(
    start_date: datetime = Query(
        example=datetime.now(),
    ),
    end_date: datetime = Query(
        example=datetime.now(),
    ),
    db: Session = Depends(get_db),
):
    """
        Fetches electricity prices for a given time period.
    """
    # Convert the input dates timezones to UTC
    start_date_dt = convert_date_tz(start_date)
    end_date_dt = convert_date_tz(end_date)
    if start_date_dt == end_date_dt:
        end_date_dt.replace(hour=23)

    # Query the database for prices within the given date range
    prices = crud.get_electricity_prices(db, start_date_dt, end_date_dt)

    # Convert timestamps to the specified timezone if needed
    prices = convert_to_timezone(prices)

    return schemas.ElectricityPrice.from_db_model_list(prices)


@router.get("/current_price", response_model=schemas.ElectricityPrice)
def get_current_electricity_price(
    db: Session = Depends(get_db),
):
    """
        Fetches the current electricity price.
    """
    # Fetch the current date and hour
    current_datetime = datetime.now(pytz.utc)

    # Query the database for prices within the given date range
    prices = crud.get_electricity_prices(db, current_datetime, current_datetime)

    # Convert timestamps to the specified timezone if needed
    prices = convert_to_timezone(prices)

    return schemas.ElectricityPrice.from_db_model_list(prices)


@router.get("/cheapest_hours", response_model=schemas.ElectricityPrice)
def get_cheapest_hours_today(
    amount: int = Query(6, description="Number of cheapest hours to find"),
    consecutive: bool = Query(False, description="Should the hours be consecutive"),
    db: Session = Depends(get_db),
):
    """
        Fetches the cheapest electricity prices for a given amount.
    """
    if amount > 24 or amount < 1:
        raise HTTPException(status_code=400, detail="Amount must be between 1 and 24")

    # Define the timezone
    tz = pytz.timezone(os.getenv("TZ"))

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
            status_code=404, detail="No electricity prices found for today"
        )

    if consecutive:
        # Ensure db_prices is sorted by timestamp
        db_prices.sort(key=lambda x: x.timestamp)

        # Find the cheapest consecutive hours
        cheapest_hours = []
        for i in range(len(db_prices) - amount + 1):
            consecutive_hours = db_prices[i : i + amount]
            if len(consecutive_hours) == amount:
                cheapest_hours.append(consecutive_hours)

        # Sort by the sum of prices in each consecutive block
        if cheapest_hours:
            cheapest_hours.sort(key=lambda x: sum(hour.price for hour in x))
            cheapest_hours = cheapest_hours[0]
        else:
            cheapest_hours = []
    else:
        # Ensure db_prices is sorted by price
        db_prices.sort(key=lambda x: x.price)

        # Get the cheapest non-consecutive hours
        cheapest_hours = db_prices[:amount] if len(db_prices) >= amount else db_prices

    # Sort the cheapest hours by timestamp
    cheapest_hours.sort(key=lambda x: x.timestamp)

    # Convert timestamps to the specified timezone if needed
    cheapest_hours = convert_to_timezone(cheapest_hours)

    return schemas.ElectricityPrice.from_db_model_list(cheapest_hours)
