"""
Provider manager for electricity price providers.

This module manages multiple electricity price providers and implements
fallback logic to ensure reliable data fetching.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional
import aiohttp

from app.models import ElectricityPrice
from .base import ElectricityPriceProvider, ProviderError
from .nordpool import NordpoolClient
from .entsoe import EntsoeClient

_LOGGER = logging.getLogger(__name__)


class ProviderManager:
    """
    Manages multiple electricity price providers with fallback logic.
    
    Tries providers in priority order (Nordpool first, then ENTSO-E).
    """
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self._owned_session = session is None
        self._providers: List[ElectricityPriceProvider] = []
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers in priority order."""
        # Add Nordpool provider (highest priority)
        nordpool = NordpoolClient(session=self._session)
        if nordpool.is_available():
            self._providers.append(nordpool)
            _LOGGER.info("Nordpool provider initialized")
        else:
            _LOGGER.warning("Nordpool provider not available")
        
        # Add ENTSO-E provider (fallback)
        entsoe = EntsoeClient()
        if entsoe.is_available():
            self._providers.append(entsoe)
            _LOGGER.info("ENTSO-E provider initialized")
        else:
            _LOGGER.warning("ENTSO-E provider not available (missing API key)")
        
        # Sort providers by priority
        self._providers.sort(key=lambda p: p.get_priority())
        
        if not self._providers:
            _LOGGER.error("No electricity price providers available!")
        else:
            provider_names = [p.name for p in self._providers]
            _LOGGER.info(f"Initialized providers in order: {provider_names}")
    
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
    
    async def get_electricity_price(
        self,
        start_date: datetime,
        end_date: datetime,
        max_retries: int = 2
    ) -> List[ElectricityPrice]:
        """
        Fetch electricity prices using available providers with fallback logic.
        
        Args:
            start_date: Start datetime (timezone-aware)
            end_date: End datetime (timezone-aware)
            max_retries: Maximum number of providers to try
            
        Returns:
            List of ElectricityPrice objects
            
        Raises:
            ProviderError: If all providers fail
        """
        if not self._providers:
            raise ProviderError("No electricity price providers available")
        
        # Validate input
        if not start_date.tzinfo or not end_date.tzinfo:
            raise ValueError("Dates must be timezone-aware")
        
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        last_error = None
        providers_tried = 0
        
        for provider in self._providers:
            if providers_tried >= max_retries:
                break
                
            try:
                _LOGGER.info(f"Trying provider: {provider.name}")
                prices = await provider.get_electricity_price(start_date, end_date)
                
                if prices:
                    _LOGGER.info(f"Successfully fetched {len(prices)} prices from {provider.name}")
                    return prices
                else:
                    _LOGGER.warning(f"Provider {provider.name} returned no data")
                    
            except Exception as e:
                last_error = e
                _LOGGER.warning(f"Provider {provider.name} failed: {str(e)}")
                
            providers_tried += 1
        
        # All providers failed
        error_msg = f"All {providers_tried} providers failed"
        if last_error:
            error_msg += f". Last error: {str(last_error)}"
        
        raise ProviderError(error_msg)
    
    def get_electricity_price_sync(
        self,
        start_date: datetime,
        end_date: datetime,
        max_retries: int = 2
    ) -> List[ElectricityPrice]:
        """
        Synchronous version for backward compatibility.

        Args:
            start_date: Start datetime (timezone-aware)
            end_date: End datetime (timezone-aware)
            max_retries: Maximum number of providers to try

        Returns:
            List of ElectricityPrice objects

        Raises:
            ProviderError: If all providers fail
        """
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # If we get here, we're in an async context - this is not allowed
            raise RuntimeError("Cannot call synchronous method from async context. Use get_electricity_price() instead.")
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                # No event loop exists, create a new one - this is the normal case
                return asyncio.run(self.get_electricity_price(start_date, end_date, max_retries))
            else:
                # We're in an async context
                raise e
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available provider names.
        
        Returns:
            List of provider names in priority order
        """
        return [provider.name for provider in self._providers]
    
    def get_provider_count(self) -> int:
        """
        Get the number of available providers.
        
        Returns:
            Number of available providers
        """
        return len(self._providers)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()


# Global instance for backward compatibility
_global_manager: Optional[ProviderManager] = None


def get_provider_manager() -> ProviderManager:
    """
    Get the global provider manager instance.
    
    Returns:
        Global ProviderManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = ProviderManager()
    return _global_manager


async def get_electricity_price(
    start_date: datetime,
    end_date: datetime
) -> List[ElectricityPrice]:
    """
    Convenience function for fetching electricity prices.
    
    Args:
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)
        
    Returns:
        List of ElectricityPrice objects
        
    Raises:
        ProviderError: If all providers fail
    """
    manager = get_provider_manager()
    return await manager.get_electricity_price(start_date, end_date)


def get_electricity_price_sync(
    start_date: datetime,
    end_date: datetime
) -> List[ElectricityPrice]:
    """
    Synchronous convenience function for fetching electricity prices.

    Args:
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)

    Returns:
        List of ElectricityPrice objects

    Raises:
        ProviderError: If all providers fail
    """
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # If we get here, we're in an async context - this is not allowed
        raise RuntimeError("Cannot call synchronous method from async context. Use get_electricity_price() instead.")
    except RuntimeError as e:
        if "no running event loop" in str(e).lower():
            # No event loop exists, create a new one - this is the normal case
            manager = get_provider_manager()
            return asyncio.run(manager.get_electricity_price(start_date, end_date))
        else:
            # We're in an async context
            raise e
