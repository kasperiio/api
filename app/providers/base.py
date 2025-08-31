"""
Base classes and interfaces for electricity price providers.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List
from app.models import ElectricityPrice


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class ProviderAPIError(ProviderError):
    """Exception raised when API request fails."""
    pass


class ProviderDataError(ProviderError):
    """Exception raised when data parsing fails."""
    pass


class ElectricityPriceProvider(ABC):
    """Abstract base class for electricity price providers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def get_electricity_price(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[ElectricityPrice]:
        """
        Fetch electricity prices for the given time range.
        
        Args:
            start_date: Start datetime (timezone-aware)
            end_date: End datetime (timezone-aware)
            
        Returns:
            List of ElectricityPrice objects
            
        Raises:
            ProviderError: If the provider fails to fetch data
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available and properly configured.
        
        Returns:
            True if provider is available, False otherwise
        """
        pass
    
    def get_priority(self) -> int:
        """
        Get the priority of this provider (lower number = higher priority).
        
        Returns:
            Priority number (0 = highest priority)
        """
        return 100  # Default low priority
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()
