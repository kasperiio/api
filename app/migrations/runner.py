"""
Migration runner and base classes.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, Integer, String, DateTime, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

_LOGGER = logging.getLogger(__name__)

# Migration tracking table
MigrationBase = declarative_base()


class MigrationRecord(MigrationBase):
    """Table to track applied migrations."""
    __tablename__ = 'migration_history'
    
    id = Column(Integer, primary_key=True)
    migration_id = Column(String(255), unique=True, nullable=False)
    applied_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    description = Column(String(500))


class Migration(ABC):
    """Base class for database migrations."""
    
    def __init__(self, migration_id: str, description: str):
        self.migration_id = migration_id
        self.description = description
    
    @abstractmethod
    def up(self, session: Session) -> None:
        """Apply the migration."""
        pass
    
    @abstractmethod
    def down(self, session: Session) -> None:
        """Rollback the migration (optional implementation)."""
        pass
    
    def can_rollback(self) -> bool:
        """Whether this migration supports rollback."""
        return True


class MigrationRunner:
    """Handles running database migrations."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """Create the migration tracking table if it doesn't exist."""
        try:
            MigrationBase.metadata.create_all(self.engine)
            _LOGGER.info("Migration tracking table ready")
        except SQLAlchemyError as e:
            _LOGGER.error(f"Failed to create migration table: {e}")
            raise
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migration IDs."""
        session = self.SessionLocal()
        try:
            records = session.query(MigrationRecord).all()
            return [record.migration_id for record in records]
        except SQLAlchemyError as e:
            _LOGGER.error(f"Failed to get applied migrations: {e}")
            return []
        finally:
            session.close()
    
    def is_migration_applied(self, migration_id: str) -> bool:
        """Check if a specific migration has been applied."""
        return migration_id in self.get_applied_migrations()
    
    def apply_migration(self, migration: Migration) -> bool:
        """
        Apply a single migration.
        
        Returns:
            True if migration was applied, False if already applied
        """
        if self.is_migration_applied(migration.migration_id):
            _LOGGER.info(f"Migration {migration.migration_id} already applied, skipping")
            return False
        
        session = self.SessionLocal()
        try:
            _LOGGER.info(f"Applying migration {migration.migration_id}: {migration.description}")
            
            # Apply the migration
            migration.up(session)
            
            # Record the migration
            record = MigrationRecord(
                migration_id=migration.migration_id,
                description=migration.description,
                applied_at=datetime.utcnow()
            )
            session.add(record)
            session.commit()
            
            _LOGGER.info(f"Successfully applied migration {migration.migration_id}")
            return True
            
        except Exception as e:
            session.rollback()
            _LOGGER.error(f"Failed to apply migration {migration.migration_id}: {e}")
            raise
        finally:
            session.close()
    
    def rollback_migration(self, migration: Migration) -> bool:
        """
        Rollback a single migration.
        
        Returns:
            True if migration was rolled back, False if not applied
        """
        if not self.is_migration_applied(migration.migration_id):
            _LOGGER.info(f"Migration {migration.migration_id} not applied, nothing to rollback")
            return False
        
        if not migration.can_rollback():
            raise ValueError(f"Migration {migration.migration_id} does not support rollback")
        
        session = self.SessionLocal()
        try:
            _LOGGER.info(f"Rolling back migration {migration.migration_id}")
            
            # Rollback the migration
            migration.down(session)
            
            # Remove the migration record
            session.query(MigrationRecord).filter(
                MigrationRecord.migration_id == migration.migration_id
            ).delete()
            session.commit()
            
            _LOGGER.info(f"Successfully rolled back migration {migration.migration_id}")
            return True
            
        except Exception as e:
            session.rollback()
            _LOGGER.error(f"Failed to rollback migration {migration.migration_id}: {e}")
            raise
        finally:
            session.close()
    
    def run_migrations(self, migrations: List[Migration]) -> int:
        """
        Run a list of migrations in order.
        
        Returns:
            Number of migrations applied
        """
        applied_count = 0
        
        for migration in migrations:
            try:
                if self.apply_migration(migration):
                    applied_count += 1
            except Exception as e:
                _LOGGER.error(f"Migration failed, stopping: {e}")
                break
        
        if applied_count > 0:
            _LOGGER.info(f"Applied {applied_count} migrations successfully")
        else:
            _LOGGER.info("No new migrations to apply")
        
        return applied_count
