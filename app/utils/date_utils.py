from datetime import datetime
import os
import pytz


def convert_date_tz(date: datetime) -> datetime:
    """
    Converts a given datetime object to UTC timezone.
    Args:
        date (datetime): The datetime object to be converted.
    Returns:
        datetime: The converted datetime object in UTC timezone.
    """
    tz_str = os.getenv("TZ", "UTC")
    tz = pytz.timezone(tz_str)

    if not date.tzinfo:
        date.replace(tzinfo=tz)
    return date.astimezone(pytz.utc)
