from dotenv import load_dotenv
load_dotenv()
import logging
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.database import engine, SQLALCHEMY_DATABASE_URL
from app.models import Base
from app.routers import electricity
from app.migrations import run_migrations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Run database migrations
try:
    migration_count = run_migrations(SQLALCHEMY_DATABASE_URL)
    if migration_count > 0:
        logger.info(f"Applied {migration_count} database migrations")
    else:
        logger.info("Database is up to date")
except Exception as e:
    logger.error(f"Migration failed: {e}")
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