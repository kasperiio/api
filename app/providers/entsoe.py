"""
ENTSO-E electricity price provider.

This module provides integration with the ENTSO-E API for fetching
Finnish electricity spot prices as a fallback provider.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.models import ElectricityPrice
from .base import ElectricityPriceProvider, ProviderAPIError, ProviderDataError

_LOGGER = logging.getLogger(__name__)


class EntsoeClient(ElectricityPriceProvider):
    """Client for interacting with the ENTSO-E API."""

    BASE_URL = "https://web-api.tp.entsoe.eu/api"
    FINLAND_DOMAIN = "10YFI-1--------U"
    VAT_RATE = 1.255  # 25.5% VAT

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("ENTSO-E")
        self.api_key = api_key or os.getenv("ENTSOE_API_KEY")
        
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
        utc_dt = dt.astimezone(timezone.utc)
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
            raise ProviderDataError(f"Failed to parse XML response: {str(e)}")

    def _parse_period(self, period: ET.Element, namespace: Dict) -> List[Dict]:
        """Parse a single period from the XML response."""
        try:
            start = period.find(".//ns:start", namespace).text
            start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)

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

    async def get_electricity_price(
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
            ProviderAPIError: If API request fails
            ProviderDataError: If data parsing fails
        """
        # Validate input
        if not start_date.tzinfo or not end_date.tzinfo:
            raise ValueError("Dates must be timezone-aware")

        if start_date > end_date:
            raise ValueError("Start date must be before end date")

        params = self._get_request_params(start_date, end_date)

        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=60
            )

            if response.status_code == 401:
                raise ProviderAPIError("Invalid ENTSO-E API key")
            elif response.status_code == 400:
                raise ProviderAPIError(f"Bad request: {response.text}")

            response.raise_for_status()

            prices = self._parse_xml_response(response.text)
            return [ElectricityPrice(**price) for price in prices]

        except requests.exceptions.RequestException as e:
            raise ProviderAPIError(f"ENTSO-E API request failed: {str(e)}")

    def is_available(self) -> bool:
        """
        Check if the ENTSO-E provider is available.
        
        Returns:
            True if API key is configured, False otherwise
        """
        return self.api_key is not None

    def get_priority(self) -> int:
        """
        Get the priority of this provider.
        
        Returns:
            10 (lower priority than Nordpool)
        """
        return 10

    # Synchronous method for backward compatibility
    def get_electricity_price_sync(
            self,
            start_date: datetime,
            end_date: datetime,
    ) -> List[ElectricityPrice]:
        """
        Synchronous version for backward compatibility.
        
        This method maintains the original synchronous interface.
        """
        import asyncio
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use run_until_complete
                # This is a limitation - in this case, the caller should use the async version
                raise RuntimeError("Cannot call synchronous method from async context. Use get_electricity_price() instead.")
            else:
                return loop.run_until_complete(self.get_electricity_price(start_date, end_date))
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(self.get_electricity_price(start_date, end_date))
