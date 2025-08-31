#!/usr/bin/env python3
"""
Standalone migration script.

This script can be run independently to apply database migrations
without starting the full application.

Usage:
    python migrate.py                    # Run all pending migrations
    python migrate.py --check           # Check migration status
    python migrate.py --rollback ID     # Rollback specific migration
"""

import argparse
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.database import SQLALCHEMY_DATABASE_URL
from app.migrations import run_migrations  # noqa: E402
from app.migrations.runner import MigrationRunner
from app.migrations.migrations import get_all_migrations

# Configure unified logging for standalone script
from app.logging_config import setup_logging, logger
setup_logging()


def check_migration_status():
    """Check and display migration status."""
    try:
        runner = MigrationRunner(SQLALCHEMY_DATABASE_URL)
        all_migrations = get_all_migrations()
        applied_migrations = runner.get_applied_migrations()
        
        logger.info("=== MIGRATION STATUS ===")
        logger.info("Database: %s", SQLALCHEMY_DATABASE_URL)
        logger.info("Total migrations: %s", len(all_migrations))
        logger.info("Applied migrations: %s", len(applied_migrations))

        logger.info("Migration Details:")
        for migration in all_migrations:
            status = "✅ APPLIED" if migration.migration_id in applied_migrations else "⏳ PENDING"
            logger.info("  %s: %s", migration.migration_id, status)
            logger.info("    Description: %s", migration.description)

        pending_count = len(all_migrations) - len(applied_migrations)
        if pending_count > 0:
            logger.warning("%s migrations pending", pending_count)
            return False
        else:
            logger.info("✅ Database is up to date")
            return True
            
    except Exception as e:
        logger.error("Failed to check migration status: %s", e)
        return False


def run_all_migrations():
    """Run all pending migrations."""
    try:
        logger.info("=== RUNNING MIGRATIONS ===")
        migration_count = run_migrations(SQLALCHEMY_DATABASE_URL)

        if migration_count > 0:
            logger.info("✅ Successfully applied %s migrations", migration_count)
        else:
            logger.info("✅ No migrations needed - database is up to date")

        return True

    except Exception as e:
        logger.error("Migration failed: %s", e)
        return False


def rollback_migration(migration_id: str):
    """Rollback a specific migration."""
    try:
        runner = MigrationRunner(SQLALCHEMY_DATABASE_URL)
        all_migrations = get_all_migrations()
        
        # Find the migration to rollback
        target_migration = None
        for migration in all_migrations:
            if migration.migration_id == migration_id:
                target_migration = migration
                break
        
        if not target_migration:
            logger.info("❌ Migration '%s' not found", migration_id)
            return False
        
        logger.info("\n=== ROLLING BACK MIGRATION ===")
        logger.info("Migration: %s", migration_id)
        logger.info("Description: %s", target_migration.description)
        
        # Confirm rollback
        confirm = input("\nAre you sure you want to rollback this migration? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Rollback cancelled")
            return False
        
        success = runner.rollback_migration(target_migration)
        if success:
            logger.info("✅ Migration rolled back successfully")
        else:
            logger.info("ℹ️  Migration was not applied, nothing to rollback")
        
        return True
        
    except Exception as e:
        logger.error("Rollback failed: %s", e)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Check migration status without applying"
    )
    parser.add_argument(
        "--rollback", 
        type=str, 
        help="Rollback specific migration by ID"
    )
    
    args = parser.parse_args()
    
    logger.info("🔧 Database Migration Tool")
    logger.info("=" * 50)
    
    if args.check:
        success = check_migration_status()
    elif args.rollback:
        success = rollback_migration(args.rollback)
    else:
        # Default: run migrations
        success = run_all_migrations()
    
    if success:
        logger.info("\n✅ Operation completed successfully")
        sys.exit(0)
    else:
        logger.error("\n❌ Operation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
