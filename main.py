import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from routers.skin import router as skin_router
from routers.mental import router as mental_router

load_dotenv()

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-Dermind-Key")

app = FastAPI()

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Akses ditolak. API Key salah.")

app.include_router(skin_router, dependencies=[Depends(get_api_key)])
app.include_router(mental_router, dependencies=[Depends(get_api_key)])
