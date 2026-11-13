"""
Migration to make price column nullable.

This allows storing NULL values for timestamps where data is unavailable
from all providers, preventing repeated fetch attempts.
"""

from sqlalchemy import text
from sqlalchemy.orm import Session
from app.migrations.runner import Migration


class MakepriceNullable(Migration):
    """Make the price column nullable to support unavailable data markers."""
    
    def __init__(self):
        super().__init__(
            migration_id="m002_nullable_price",
            description="Make price column nullable for unavailable data"
        )
    
    def up(self, session: Session) -> None:
        """Apply the migration - make price column nullable."""
        # SQLite doesn't support ALTER COLUMN directly, so we need to:
        # 1. Create a new table with the correct schema
        # 2. Copy data from old table
        # 3. Drop old table
        # 4. Rename new table

        session.execute(text("""
            CREATE TABLE electricity_prices_new (
                timestamp TIMESTAMP NOT NULL,
                price FLOAT,
                PRIMARY KEY (timestamp)
            )
        """))

        session.execute(text("""
            INSERT INTO electricity_prices_new (timestamp, price)
            SELECT timestamp, price FROM electricity_prices
        """))

        session.execute(text("DROP TABLE electricity_prices"))

        session.execute(text("ALTER TABLE electricity_prices_new RENAME TO electricity_prices"))

        # Recreate the index
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_electricity_prices_timestamp
            ON electricity_prices (timestamp)
        """))

        session.commit()
    
    def down(self, session: Session) -> None:
        """Rollback the migration - make price column NOT NULL again."""
        # This will fail if there are NULL values in the price column
        session.execute(text("""
            CREATE TABLE electricity_prices_new (
                timestamp TIMESTAMP NOT NULL,
                price FLOAT NOT NULL,
                PRIMARY KEY (timestamp)
            )
        """))

        # Only copy rows with non-NULL prices
        session.execute(text("""
            INSERT INTO electricity_prices_new (timestamp, price)
            SELECT timestamp, price FROM electricity_prices
            WHERE price IS NOT NULL
        """))

        session.execute(text("DROP TABLE electricity_prices"))

        session.execute(text("ALTER TABLE electricity_prices_new RENAME TO electricity_prices"))

        # Recreate the index
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_electricity_prices_timestamp
            ON electricity_prices (timestamp)
        """))

        session.commit()

