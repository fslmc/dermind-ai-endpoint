import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile
from wrappers.skin_wrapper import predict_skin

router = APIRouter(prefix="/skin", tags=["skin"])


@router.post("/infer")
async def infer_skin(image: UploadFile = File(...)):
    image_bytes = await image.read()
    try:
        result = await asyncio.to_thread(predict_skin, image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result
