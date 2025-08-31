from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.logging_config import logger
from app.database import engine, SQLALCHEMY_DATABASE_URL
from app.models import Base
from app.routers import electricity
from app.migrations import run_migrations

# Load environment variables first
load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

# Run database migrations
try:
    migration_count = run_migrations(SQLALCHEMY_DATABASE_URL)
    if migration_count > 0:
        logger.info("Applied %s database migrations", migration_count)
    else:
        logger.info("Database is up to date")
except Exception as e:
    logger.error("Migration failed: %s", e)
    raise


tags_metadata = [
    {
        "name": "electricity",
        "description": "Get electricity spot prices for finnish region.",
    },
]

app = FastAPI(openapi_tags=tags_metadata)

app.include_router(electricity.router)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to API documentation."""
    return RedirectResponse(url="/docs")