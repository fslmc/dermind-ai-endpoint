import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from wrappers.mental_wrapper import predict_mental

router = APIRouter(prefix="/mental", tags=["mental"])


class MentalRequest(BaseModel):
    text: str


@router.post("/infer")
async def infer_mental(request: MentalRequest):
    try:
        result = await asyncio.to_thread(predict_mental, request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result
