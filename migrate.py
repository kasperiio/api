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
import logging
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.database import SQLALCHEMY_DATABASE_URL
from app.migrations import run_migrations
from app.migrations.runner import MigrationRunner
from app.migrations.migrations import get_all_migrations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_migration_status():
    """Check and display migration status."""
    try:
        runner = MigrationRunner(SQLALCHEMY_DATABASE_URL)
        all_migrations = get_all_migrations()
        applied_migrations = runner.get_applied_migrations()
        
        print("\n=== MIGRATION STATUS ===")
        print(f"Database: {SQLALCHEMY_DATABASE_URL}")
        print(f"Total migrations: {len(all_migrations)}")
        print(f"Applied migrations: {len(applied_migrations)}")
        
        print("\nMigration Details:")
        for migration in all_migrations:
            status = "✅ APPLIED" if migration.migration_id in applied_migrations else "⏳ PENDING"
            print(f"  {migration.migration_id}: {status}")
            print(f"    Description: {migration.description}")
        
        pending_count = len(all_migrations) - len(applied_migrations)
        if pending_count > 0:
            print(f"\n⚠️  {pending_count} migrations pending")
            return False
        else:
            print("\n✅ Database is up to date")
            return True
            
    except Exception as e:
        logger.error(f"Failed to check migration status: {e}")
        return False


def run_all_migrations():
    """Run all pending migrations."""
    try:
        print("\n=== RUNNING MIGRATIONS ===")
        migration_count = run_migrations(SQLALCHEMY_DATABASE_URL)
        
        if migration_count > 0:
            print(f"✅ Successfully applied {migration_count} migrations")
        else:
            print("✅ No migrations needed - database is up to date")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print(f"❌ Migration failed: {e}")
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
            print(f"❌ Migration '{migration_id}' not found")
            return False
        
        print(f"\n=== ROLLING BACK MIGRATION ===")
        print(f"Migration: {migration_id}")
        print(f"Description: {target_migration.description}")
        
        # Confirm rollback
        confirm = input("\nAre you sure you want to rollback this migration? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Rollback cancelled")
            return False
        
        success = runner.rollback_migration(target_migration)
        if success:
            print("✅ Migration rolled back successfully")
        else:
            print("ℹ️  Migration was not applied, nothing to rollback")
        
        return True
        
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        print(f"❌ Rollback failed: {e}")
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
    
    print("🔧 Database Migration Tool")
    print("=" * 50)
    
    if args.check:
        success = check_migration_status()
    elif args.rollback:
        success = rollback_migration(args.rollback)
    else:
        # Default: run migrations
        success = run_all_migrations()
    
    if success:
        print("\n✅ Operation completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Operation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
