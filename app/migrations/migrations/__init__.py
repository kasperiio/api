"""
Migration definitions.
"""

from typing import List
from ..runner import Migration
from .m002_nullable_price import MakepriceNullable


def get_all_migrations() -> List[Migration]:
    """
    Get all migrations in chronological order.

    Returns:
        List of migrations to be applied
    """
    return [
        MakepriceNullable(),
    ]
