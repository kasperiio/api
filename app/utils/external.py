from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import os
import pytz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import HTTPException

from app.models import ElectricityPrice


class EntsoeAPIError(Exception):
    """Custom exception for ENTSO-E API errors."""
    pass


class EntsoeClient:
    """Client for interacting with the ENTSO-E API."""

    BASE_URL = "https://web-api.tp.entsoe.eu/api"
    FINLAND_DOMAIN = "10YFI-1--------U"
    VAT_RATE = 1.255  # 25.5% VAT

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ENTSOE_API_KEY")
        if not self.api_key:
            raise EntsoeAPIError("ENTSOE_API_KEY not found in environment variables")

        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

    def _format_date(self, dt: datetime) -> str:
        """Format datetime for ENTSO-E API.

        Args:
            dt: datetime object with timezone info

        Returns:
            Formatted string in YYYYMMDDHHMM format
        """
        # Convert to UTC if not already
        utc_dt = dt.astimezone(pytz.UTC)
        return utc_dt.strftime("%Y%m%d%H%M")

    def _get_request_params(self, start_date: datetime, end_date: datetime) -> Dict:
        """Prepare request parameters."""
        return {
            "documentType": "A44",
            "in_Domain": self.FINLAND_DOMAIN,
            "out_Domain": self.FINLAND_DOMAIN,
            "periodStart": self._format_date(start_date),
            "periodEnd": self._format_date(end_date),
            "securityToken": self.api_key,
        }

    def _parse_xml_response(self, xml_content: str) -> List[Dict]:
        """Parse XML response from ENTSO-E API."""
        try:
            root = ET.fromstring(xml_content)
            namespace = {'ns': root.tag.split('}')[0].strip('{')}
            prices = []

            for timeseries in root.findall(".//ns:TimeSeries", namespace):
                for period in timeseries.findall(".//ns:Period", namespace):
                    prices.extend(self._parse_period(period, namespace))

            return prices
        except ET.ParseError as e:
            raise EntsoeAPIError(f"Failed to parse XML response: {str(e)}")

    def _parse_period(self, period: ET.Element, namespace: Dict) -> List[Dict]:
        """Parse a single period from the XML response."""
        try:
            start = period.find(".//ns:start", namespace).text
            start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%MZ").replace(tzinfo=pytz.utc)

            prices = []
            for point in period.findall(".//ns:Point", namespace):
                price_data = self._parse_point(point, start_dt, namespace)
                if price_data:
                    prices.append(price_data)

            return prices
        except (AttributeError, ValueError) as e:
            return []

    def _parse_point(self, point: ET.Element, start_dt: datetime, namespace: Dict) -> Optional[Dict]:
        """Parse a single price point from the XML response."""
        try:
            position = int(point.find("ns:position", namespace).text)
            price_amount = point.find(".//ns:price.amount", namespace)
            if price_amount is None:
                return None

            price_mwh = float(price_amount.text)

            # Convert MWh to cents/kWh and apply VAT
            price_kwh = price_mwh / 1000
            price_cents = price_kwh * 100 * self.VAT_RATE

            timestamp = start_dt + timedelta(hours=position - 1)

            return {
                "timestamp": timestamp,
                "price": round(price_cents, 2),  # Round to 2 decimal places
            }
        except (ValueError, TypeError, AttributeError) as e:
            return None

    def get_electricity_price(
            self,
            start_date: datetime,
            end_date: datetime,
    ) -> List[ElectricityPrice]:
        """
        Fetch electricity prices from ENTSO-E API.

        Args:
            start_date: Start datetime (timezone-aware)
            end_date: End datetime (timezone-aware)

        Returns:
            List of ElectricityPrice objects

        Raises:
            EntsoeAPIError: If API request fails
            HTTPException: If input validation fails
        """
        # Validate input
        if not start_date.tzinfo or not end_date.tzinfo:
            raise HTTPException(status_code=400, detail="Dates must be timezone-aware")

        if start_date > end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")

        params = self._get_request_params(start_date, end_date)

        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=60
            )

            if response.status_code == 401:
                raise EntsoeAPIError("Invalid API key")
            elif response.status_code == 400:
                raise EntsoeAPIError(f"Bad request: {response.text}")

            response.raise_for_status()

            prices = self._parse_xml_response(response.text)
            return [ElectricityPrice(**price) for price in prices]

        except requests.exceptions.RequestException as e:
            raise EntsoeAPIError(f"API request failed: {str(e)}")


# Create a global client instance
entsoe_client = EntsoeClient()


def get_electricity_price(
        start_date: datetime,
        end_date: datetime
) -> List[ElectricityPrice]:
    """Wrapper function for backward compatibility."""
    return entsoe_client.get_electricity_price(start_date, end_date)