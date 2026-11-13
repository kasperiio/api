"""
Provider manager for electricity price providers.

This module manages multiple electricity price providers and implements
fallback logic to ensure reliable data fetching.
"""

from datetime import datetime, timedelta
from typing import List, Optional
import httpx

from app.logging_config import logger
from app.models import ElectricityPrice
from .base import ElectricityPriceProvider, ProviderError
from .entsoe import EntsoeClient


class ProviderManager:
    """
    Manages multiple electricity price providers with fallback logic.
    
    Tries providers in priority order (Nordpool first, then ENTSO-E).
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._client = client
        self._owned_client = client is None
        self._providers: List[ElectricityPriceProvider] = []
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available providers in priority order."""
        entsoe = EntsoeClient(client=None)
        if entsoe.is_available():
            self._providers.append(entsoe)
            logger.info("ENTSO-E provider initialized")
        else:
            logger.warning("ENTSO-E provider not available (missing API key)")

        # Sort providers by priority (only one provider now, but keeping for future extensibility)
        self._providers.sort(key=lambda p: p.get_priority())

        if not self._providers:
            logger.error("No electricity price providers available!")
        else:
            provider_names = [p.name for p in self._providers]
            logger.info("Initialized providers in order: %s", provider_names)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                follow_redirects=True
            )
        return self._client

    async def _close_client(self):
        """Close the httpx client if we own it."""
        if self._owned_client and self._client:
            await self._client.aclose()
            self._client = None

    async def get_electricity_price(
        self,
        start_date: datetime,
        end_date: datetime,
        max_retries: int = 2,
        chunk_days: int = 30
    ) -> List[ElectricityPrice]:
        """
        Fetch electricity prices using available providers with fallback logic.

        Splits large date ranges into smaller chunks to avoid timeouts.

        Args:
            start_date: Start datetime (timezone-aware)
            end_date: End datetime (timezone-aware)
            max_retries: Maximum number of providers to try
            chunk_days: Maximum number of days per request chunk (default: 30)

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

        # Split the date range into chunks
        date_chunks = self._split_date_range(start_date, end_date, chunk_days)
        logger.info("Split request into %d chunks of max %d days each", len(date_chunks), chunk_days)

        last_error = None
        providers_tried = 0

        # Track which chunks we still need to fetch
        chunks_to_fetch = list(enumerate(date_chunks, 1))  # [(index, (start, end)), ...]
        all_prices = []

        for provider in self._providers:
            if providers_tried >= max_retries:
                break

            if not chunks_to_fetch:
                # All chunks successfully fetched
                break

            try:
                logger.info("Trying provider: %s for %d remaining chunks", provider.name, len(chunks_to_fetch))
                failed_chunks = []

                # Fetch each remaining chunk
                for i, (chunk_start, chunk_end) in chunks_to_fetch:
                    logger.info("Fetching chunk %d/%d: %s to %s",
                               i, len(date_chunks), chunk_start, chunk_end)

                    try:
                        prices = await provider.get_electricity_price(chunk_start, chunk_end)

                        if prices:
                            all_prices.extend(prices)
                            logger.info("Chunk %d/%d returned %d prices", i, len(date_chunks), len(prices))
                        else:
                            logger.warning("Chunk %d/%d returned no data", i, len(date_chunks))
                            # Track this chunk as failed so we can retry with next provider
                            failed_chunks.append((i, (chunk_start, chunk_end)))
                    except Exception as chunk_error:
                        logger.error("Chunk %d/%d failed: %s", i, len(date_chunks), str(chunk_error))
                        # Track this chunk as failed so we can retry with next provider
                        failed_chunks.append((i, (chunk_start, chunk_end)))

                # Update chunks_to_fetch to only include failed chunks for next provider
                chunks_to_fetch = failed_chunks

            except Exception as e:
                last_error = e
                logger.warning("Provider %s failed: %s", provider.name, str(e))

            providers_tried += 1

        # Return whatever we managed to fetch
        if all_prices:
            if chunks_to_fetch:
                logger.warning(
                    "Successfully fetched %d prices, but %d chunks failed across all providers",
                    len(all_prices), len(chunks_to_fetch)
                )
            else:
                logger.info("Successfully fetched %d prices from providers", len(all_prices))
            return all_prices

        # All providers failed to fetch any data
        error_msg = f"All {providers_tried} providers failed to fetch any data"
        if last_error:
            error_msg += f". Last error: {str(last_error)}"

        raise ProviderError(error_msg)

    def _split_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        chunk_days: int
    ) -> List[tuple[datetime, datetime]]:
        """
        Split a date range into smaller chunks.

        Args:
            start_date: Start datetime
            end_date: End datetime
            chunk_days: Maximum number of days per chunk

        Returns:
            List of (start, end) datetime tuples
        """
        chunks = []
        current_start = start_date

        while current_start < end_date:
            # Calculate chunk end (either chunk_days later or end_date, whichever is earlier)
            chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
            chunks.append((current_start, chunk_end))
            current_start = chunk_end

        return chunks

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
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_client()


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
