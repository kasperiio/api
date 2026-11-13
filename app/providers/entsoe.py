"""
ENTSO-E electricity price provider.

This module provides integration with the ENTSO-E API for fetching
Finnish electricity spot prices as a fallback provider.
"""

import xml.etree.ElementTree as ET
import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import os
import httpx

from app.logging_config import logger
from app.models import ElectricityPrice
from .base import ElectricityPriceProvider, ProviderAPIError, ProviderDataError


class EntsoeClient(ElectricityPriceProvider):
    """Client for interacting with the ENTSO-E API."""

    BASE_URL = "https://web-api.tp.entsoe.eu/api"
    FINLAND_DOMAIN = "10YFI-1--------U"
    VAT_RATE = 1.255  # 25.5% VAT

    def __init__(self, api_key: Optional[str] = None, client: Optional[httpx.AsyncClient] = None):
        super().__init__("ENTSO-E")
        self.api_key = api_key or os.getenv("ENTSOE_API_KEY")
        self._client = client
        self._owned_client = client is None

        # Simple rate limiting: track last request time
        # 10 requests per second - ENTSO-E allows more but we'll be conservative
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # seconds (10 requests per second)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                follow_redirects=True
            )
        return self._client

    async def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limiting."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time

        if time_since_last_request < self._min_request_interval:
            wait_time = self._min_request_interval - time_since_last_request
            logger.debug("Rate limiting: waiting %.2f seconds", wait_time)
            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()

    async def _close_client(self):
        """Close the httpx client if we own it."""
        if self._owned_client and self._client:
            await self._client.aclose()
            self._client = None

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
                # Check for curveType (A03 = variable sized blocks)
                curve_type_elem = timeseries.find(".//ns:curveType", namespace)
                curve_type = curve_type_elem.text if curve_type_elem is not None else None

                for period in timeseries.findall(".//ns:Period", namespace):
                    prices.extend(self._parse_period(period, namespace, curve_type))

            return prices
        except ET.ParseError as e:
            raise ProviderDataError(f"Failed to parse XML response: {str(e)}")

    def _parse_period(self, period: ET.Element, namespace: Dict, curve_type: Optional[str] = None) -> List[Dict]:
        """Parse a single period from the XML response.

        Args:
            period: XML element containing the period data
            namespace: XML namespace dictionary
            curve_type: Curve type (A03 = variable sized blocks, A01 = sequential)
        """
        try:
            start = period.find(".//ns:start", namespace).text
            start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)

            end = period.find(".//ns:end", namespace).text
            end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)

            # Parse resolution (e.g., "PT60M" for hourly, "PT15M" for 15-minute)
            resolution_elem = period.find(".//ns:resolution", namespace)
            if resolution_elem is not None:
                resolution = resolution_elem.text
                # Extract minutes from resolution string (e.g., "PT15M" -> 15)
                if resolution.startswith("PT") and resolution.endswith("M"):
                    interval_minutes = int(resolution[2:-1])
                else:
                    interval_minutes = 60  # Default to hourly
            else:
                interval_minutes = 60  # Default to hourly

            # Parse all points into a dictionary keyed by position
            points_by_position = {}
            for point in period.findall(".//ns:Point", namespace):
                price_data = self._parse_point(point, start_dt, interval_minutes, namespace)
                if price_data:
                    position = int(point.find("ns:position", namespace).text)
                    points_by_position[position] = price_data

            # If curveType is A03 (variable sized blocks), fill in missing positions
            # with the price from the previous position
            if curve_type == "A03" and points_by_position:
                # Calculate expected number of intervals
                total_minutes = int((end_dt - start_dt).total_seconds() / 60)
                expected_intervals = total_minutes // interval_minutes

                prices = []
                last_price = None

                for position in range(1, expected_intervals + 1):
                    if position in points_by_position:
                        # Use the actual price from this position
                        price_data = points_by_position[position]
                        last_price = price_data["price"]
                        prices.append(price_data)
                    elif last_price is not None:
                        # Fill with the previous price (variable sized block)
                        timestamp = start_dt + timedelta(minutes=interval_minutes * (position - 1))
                        prices.append({
                            "timestamp": timestamp,
                            "price": last_price,
                        })

                return prices
            else:
                # For other curve types (or if no curve type), just return the points we have
                return list(points_by_position.values())

        except (AttributeError, ValueError):
            return []

    def _parse_point(self, point: ET.Element, start_dt: datetime, interval_minutes: int, namespace: Dict) -> Optional[Dict]:
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

            # Calculate timestamp based on position and interval
            timestamp = start_dt + timedelta(minutes=interval_minutes * (position - 1))

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

        # ENTSO-E API has a 1-year limit per request
        # Warn if the range is too large (should be handled by chunking in manager)
        days_diff = (end_date - start_date).days
        if days_diff > 365:
            logger.warning(
                "ENTSO-E request spans %d days (>365). API may truncate results. "
                "Consider using smaller chunks in the manager.",
                days_diff
            )

        params = self._get_request_params(start_date, end_date)
        client = await self._get_client()

        # Apply rate limiting - wait if needed
        await self._wait_for_rate_limit()

        try:
            logger.info("Fetching data from ENTSO-E API for %s to %s", start_date, end_date)

            response = await client.get(self.BASE_URL, params=params)
            logger.debug("ENTSO-E API request: %s", response.url)

            if response.status_code == 401:
                raise ProviderAPIError("Invalid ENTSO-E API key")
            elif response.status_code == 400:
                raise ProviderAPIError(f"Bad request: {response.text}")
            elif response.status_code == 429:
                raise ProviderAPIError("ENTSO-E API rate limit exceeded")

            if response.status_code != 200:
                raise ProviderAPIError(f"ENTSO-E API returned status {response.status_code}: {response.text}")

            xml_content = response.text
            prices = self._parse_xml_response(xml_content)

            logger.info("Successfully fetched %d prices from ENTSO-E", len(prices))
            return [ElectricityPrice(**price) for price in prices]

        except httpx.HTTPError as e:
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
        return 0

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_client()


