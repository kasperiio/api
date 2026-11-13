from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.logging_config import logger
from app.database import engine, SQLALCHEMY_DATABASE_URL
from app.models import Base
import app.forecasting.models  # noqa: F401  (registers forecasting tables)
from app.routers import electricity, forecast
from app.forecasting import scheduler as forecast_scheduler
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
    {
        "name": "forecast",
        "description": "Self-learning 7-day forecast of finnish day-ahead prices with accuracy and drivers.",
    },
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Forecast jobs run in a background thread; bootstrap never blocks startup.
    forecast_scheduler.start_scheduler()
    try:
        yield
    finally:
        forecast_scheduler.stop_scheduler()


app = FastAPI(openapi_tags=tags_metadata, lifespan=lifespan)

app.include_router(electricity.router)
app.include_router(forecast.router)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to API documentation."""
    return RedirectResponse(url="/docs")