"""
Nordpool electricity price provider.

This module provides integration with the Nordpool API for fetching
Finnish electricity spot prices.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import aiohttp
import pytz
from dateutil.parser import parse as parse_dt

from app.models import ElectricityPrice
from .base import ElectricityPriceProvider, ProviderAPIError, ProviderDataError

_LOGGER = logging.getLogger(__name__)


class NordpoolClient(ElectricityPriceProvider):
    """Client for interacting with the Nordpool API."""
    
    BASE_URL = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
    FINLAND_AREA = "FI"
    CURRENCY = "EUR"
    VAT_RATE = 1.255  # 25.5% VAT for Finland
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        super().__init__("Nordpool")
        self._session = session
        self._owned_session = session is None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _close_session(self):
        """Close the aiohttp session if we own it."""
        if self._owned_session and self._session:
            await self._session.close()
            self._session = None
    
    def _parse_dt(self, time_str: str) -> datetime:
        """Parse datetimes to UTC from Stockholm time, which Nord Pool uses."""
        stockholm_tz = pytz.timezone("Europe/Stockholm")
        time = parse_dt(time_str)
        if time.tzinfo is None:
            time = stockholm_tz.localize(time)
        return time.astimezone(pytz.UTC)
    
    def _calculate_price(self, raw_price: float) -> float:
        """
        Calculate final price with VAT.
        
        Args:
            raw_price: Raw price in EUR/MWh
            
        Returns:
            Price in cents/kWh including VAT
        """
        # Convert EUR/MWh to EUR/kWh
        price_kwh = raw_price / 1000
        
        # Apply VAT
        price_with_vat = price_kwh * self.VAT_RATE
        
        # Convert to cents
        price_cents = price_with_vat * 100
        
        return round(price_cents, 2)
    
    async def _fetch_data(self, date: datetime) -> Optional[dict]:
        """
        Fetch data from Nordpool API for a specific date.
        
        Args:
            date: Date to fetch data for
            
        Returns:
            API response data or None if no data available
        """
        session = await self._get_session()
        
        params = {
            "market": "DayAhead",
            "currency": self.CURRENCY,
            "deliveryArea": self.FINLAND_AREA,
            "date": date.strftime("%Y-%m-%d")
        }
        
        try:
            async with session.get(self.BASE_URL, params=params) as response:
                _LOGGER.debug(f"Nordpool API request: {response.url}")
                
                if response.status == 204:
                    # No data available for this date
                    return None
                
                if response.status != 200:
                    raise ProviderAPIError(
                        f"Nordpool API returned status {response.status}: {await response.text()}"
                    )
                
                data = await response.json()
                return data
                
        except aiohttp.ClientError as e:
            raise ProviderAPIError(f"Nordpool API request failed: {str(e)}")
    
    def _parse_response(self, data: dict) -> List[ElectricityPrice]:
        """
        Parse Nordpool API response into ElectricityPrice objects.
        
        Args:
            data: API response data
            
        Returns:
            List of ElectricityPrice objects
        """
        if not data or "multiAreaEntries" not in data:
            return []
        
        prices = []
        
        try:
            for entry in data["multiAreaEntries"]:
                start_time = self._parse_dt(entry["deliveryStart"])
                
                # Get price for Finland area
                area_prices = entry.get("entryPerArea", {})
                if self.FINLAND_AREA not in area_prices:
                    continue
                
                raw_price = float(area_prices[self.FINLAND_AREA])
                
                # Skip invalid prices
                if raw_price == float('inf') or raw_price == float('-inf'):
                    continue
                
                final_price = self._calculate_price(raw_price)
                
                prices.append(ElectricityPrice(
                    timestamp=start_time,
                    price=final_price
                ))
                
        except (KeyError, ValueError, TypeError) as e:
            raise ProviderDataError(f"Failed to parse Nordpool response: {str(e)}")
        
        return prices
    
    async def get_electricity_price(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[ElectricityPrice]:
        """
        Fetch electricity prices from Nordpool API.
        
        Args:
            start_date: Start datetime (timezone-aware)
            end_date: End datetime (timezone-aware)
            
        Returns:
            List of ElectricityPrice objects
            
        Raises:
            ProviderAPIError: If API request fails
            ProviderDataError: If data parsing fails
        """
        if not start_date.tzinfo or not end_date.tzinfo:
            raise ValueError("Dates must be timezone-aware")
        
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        # Convert to Finnish timezone for date calculation
        finnish_tz = pytz.timezone("Europe/Helsinki")
        start_local = start_date.astimezone(finnish_tz)
        end_local = end_date.astimezone(finnish_tz)
        
        # Get all dates we need to fetch
        current_date = start_local.date()
        end_date_local = end_local.date()
        dates_to_fetch = []
        
        while current_date <= end_date_local:
            dates_to_fetch.append(datetime.combine(current_date, datetime.min.time()))
            current_date += timedelta(days=1)
        
        # Fetch data for all dates concurrently
        tasks = [self._fetch_data(date) for date in dates_to_fetch]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_prices = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                _LOGGER.warning(f"Failed to fetch data for {dates_to_fetch[i]}: {response}")
                continue
            
            if response is not None:
                prices = self._parse_response(response)
                all_prices.extend(prices)
        
        # Filter prices to the requested time range
        filtered_prices = [
            price for price in all_prices
            if start_date <= price.timestamp <= end_date
        ]
        
        # Sort by timestamp
        filtered_prices.sort(key=lambda x: x.timestamp)
        
        return filtered_prices
    
    def is_available(self) -> bool:
        """
        Check if the Nordpool provider is available.
        
        Returns:
            True (Nordpool doesn't require API keys)
        """
        return True
    
    def get_priority(self) -> int:
        """
        Get the priority of this provider.
        
        Returns:
            0 (highest priority)
        """
        return 0
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()
