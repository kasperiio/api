"""
Migration manager and registry.
"""

import logging
from typing import List
from .runner import MigrationRunner, Migration
from .migrations import get_all_migrations

_LOGGER = logging.getLogger(__name__)


def run_migrations(database_url: str) -> int:
    """
    Run all pending migrations.
    
    Args:
        database_url: Database connection URL
        
    Returns:
        Number of migrations applied
    """
    try:
        runner = MigrationRunner(database_url)
        migrations = get_all_migrations()
        
        _LOGGER.info(f"Found {len(migrations)} total migrations")
        applied_count = runner.run_migrations(migrations)
        
        return applied_count
        
    except Exception as e:
        _LOGGER.error(f"Migration process failed: {e}")
        raise
