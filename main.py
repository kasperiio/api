from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.database import engine
from app.models import Base
from app.routers import electricity

Base.metadata.create_all(bind=engine)


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