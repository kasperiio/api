"""
Database configuration and session management.

This module provides SQLAlchemy database setup with modern 2.0 patterns
for the electricity price API.
"""

from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Ensure data directory exists
Path("./data").mkdir(exist_ok=True)

SQLALCHEMY_DATABASE_URL = "sqlite:///./data/electricity_prices.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite specific
    pool_pre_ping=True,  # Enable connection health checks
    pool_recycle=3600,   # Recycle connections every hour
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.commit()
        db.close()

def init_db() -> None:
    """Initialize database and create all tables."""
    Base.metadata.create_all(bind=engine)