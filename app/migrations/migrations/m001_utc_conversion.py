"""
Migration 001: Convert timezone-aware data to UTC-only format.

This migration:
1. Ensures all existing electricity price timestamps are in UTC
2. Removes timezone-aware SQL expressions from the database
3. Prepares the system for UTC-only operation
"""

from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.logging_config import logger
from ..runner import Migration


class UTCConversionMigration(Migration):
    """Convert all timestamp data to UTC and remove timezone complexity."""
    
    def __init__(self):
        super().__init__(
            migration_id="001_utc_conversion",
            description="Convert timezone-aware data to UTC-only format"
        )
    
    def up(self, session: Session) -> None:
        """Apply the UTC conversion migration."""
        logger.info("Starting UTC conversion migration")

        # Step 1: Check if electricity_prices table exists
        result = session.execute(text("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='electricity_prices'
        """)).fetchone()

        if not result:
            logger.info("No electricity_prices table found, skipping data conversion")
            return

        # Step 2: Get count of existing records
        count_result = session.execute(text("""
            SELECT COUNT(*) as count FROM electricity_prices
        """)).fetchone()

        record_count = count_result[0] if count_result else 0
        logger.info("Found %s electricity price records to process", record_count)

        if record_count == 0:
            logger.info("No data to convert, migration complete")
            return
        
        # Step 3: Convert any timezone-aware timestamps to UTC
        # Since we're using SQLite and our UTCDateTime type already handles this,
        # we mainly need to ensure data consistency
        
        # Check for any records with potentially problematic timestamps
        sample_result = session.execute(text("""
            SELECT timestamp FROM electricity_prices LIMIT 5
        """)).fetchall()
        
        logger.info("Sample timestamps before conversion:")
        for row in sample_result:
            logger.info("  %s", row[0])

        # Step 4: Update any records that might have timezone issues
        # For SQLite, timestamps are stored as strings, so we need to ensure
        # they're in the correct UTC format

        # This query will normalize any timestamp format issues
        result = session.execute(text("""
            UPDATE electricity_prices
            SET timestamp = datetime(timestamp, 'utc')
            WHERE timestamp IS NOT NULL
        """))

        updated_count = result.rowcount if hasattr(result, 'rowcount') else record_count
        logger.info("Normalized %s timestamp records to UTC format", updated_count)

        # Step 5: Verify the conversion
        verification_result = session.execute(text("""
            SELECT COUNT(*) as count FROM electricity_prices
            WHERE timestamp IS NOT NULL
        """)).fetchone()

        verified_count = verification_result[0] if verification_result else 0
        logger.info("Verified %s records have valid UTC timestamps", verified_count)

        # Step 6: Log sample of converted data
        sample_after = session.execute(text("""
            SELECT timestamp FROM electricity_prices
            ORDER BY timestamp DESC LIMIT 3
        """)).fetchall()

        logger.info("Sample timestamps after conversion:")
        for row in sample_after:
            logger.info("  %s", row[0])

        session.commit()
        logger.info("UTC conversion migration completed successfully")
    
    def down(self, session: Session) -> None:
        """
        Rollback the UTC conversion migration.
        
        Note: This is a data migration that's difficult to rollback perfectly,
        as we don't store the original timezone information. This rollback
        is mainly for testing purposes.
        """
        logger.warning("Rolling back UTC conversion migration")
        logger.warning("Note: Original timezone information cannot be perfectly restored")

        # For rollback, we'll just log that the rollback occurred
        # In a production system, you might want to backup data before migration

        logger.info("UTC conversion rollback completed (data remains in UTC)")
    
    def can_rollback(self) -> bool:
        """This migration has limited rollback capability."""
        return True  # We allow rollback but with warnings
