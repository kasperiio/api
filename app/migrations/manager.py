"""
Migration manager and registry.
"""

from app.logging_config import logger
from .runner import MigrationRunner
from .migrations import get_all_migrations


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
        
        logger.info("Found %s total migrations", len(migrations))
        applied_count = runner.run_migrations(migrations)

        return applied_count

    except Exception as e:
        logger.error("Migration process failed: %s", e)
        raise
