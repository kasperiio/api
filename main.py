from dotenv import load_dotenv
from fastapi import FastAPI
from app.database import engine
from app.models import Base
from app.routers import electricity

Base.metadata.create_all(bind=engine)

load_dotenv()

tags_metadata = [
    {
        "name": "electricity",
        "description": "Get electricity spot prices for finnish region.",
    },
]

app = FastAPI(openapi_tags=tags_metadata)

app.include_router(electricity.router, prefix="/electricity", tags=["electricity"])
