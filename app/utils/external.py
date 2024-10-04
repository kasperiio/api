from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import os
import pytz
import requests

from app.models import ElectricityPrice


def get_electricity_price(
    start_date: datetime, end_date: datetime
) -> list[ElectricityPrice]:
    """
    Fetches electricity prices from the ENTSO-E API for a given time period.
    Args:
        start_date (datetime): The start date and time of the period.
        end_date (datetime): The end date and time of the period.
    Returns:
        list: A list of ElectricityPrice objects containing the timestamp, price, and price_daily_average_ratio.
    Raises:
        Exception: If the API request fails or returns a non-200 status code.
    """
    api_key = os.getenv("ENTSOE_API_KEY")
    url = "https://web-api.tp.entsoe.eu/api"
    params = {
        "documentType": "A44",
        "in_Domain": "10YFI-1--------U",
        "out_Domain": "10YFI-1--------U",
        "periodStart": start_date.strftime("%Y%m%d%H00"),
        "periodEnd": end_date.strftime("%Y%m%d%H00"),
        "securityToken": api_key,
    }

    response = requests.get(url, params=params, timeout=60)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from ENTSO-E API: {response.text}")

    prices = []
    root = ET.fromstring(response.text)
    namespace = {'ns': root.tag.split('}')[0].strip('{')}

    daily_prices = {}

    for timeseries in root.findall("ns:TimeSeries", namespace):
        for period in timeseries.findall("ns:Period", namespace):
            start = period.find("ns:timeInterval/ns:start", namespace).text
            start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%MZ").replace(
                tzinfo=pytz.utc
            )

            for point in period.findall("ns:Point", namespace):
                position = point.find("ns:position", namespace).text
                price_mwh = float(point.find("ns:price.amount", namespace).text)
                price_kwh = price_mwh / 1000
                price_cents = price_kwh * 100

                # Value added tax 25.5%
                price_cents = price_cents * 1.255
                hour = start_dt + timedelta(hours=int(position) - 1)

                day_str = hour.strftime("%Y-%m-%d")

                if day_str not in daily_prices:
                    daily_prices[day_str] = []

                daily_prices[day_str].append(price_cents)
                prices.append(
                    {
                        "timestamp": hour,
                        "price": price_cents,
                    }
                )

    # for day, day_prices in daily_prices.items():
    #     average_price = sum(day_prices) / len(day_prices)
    #     for price_entry in prices:
    #         if price_entry["timestamp"].strftime("%Y-%m-%d") == day:
    #             if average_price == 0:
    #                 price_entry["price_daily_average_ratio"] = 0
    #             else:
    #                 price_entry["price_daily_average_ratio"] = (
    #                     price_entry["price"] / average_price
    #                 )

    return [ElectricityPrice(**price) for price in prices]
