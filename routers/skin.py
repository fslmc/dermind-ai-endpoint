import asyncio
import json
import os                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
import urllib.error
import urllib.request

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/skin", tags=["skin"])

HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_API_TOKEN")
HF_SKIN_MODEL = os.getenv("HF_SKIN_MODEL", "google/vit-base-patch16-224")
HF_BASE_URL = "https://api-inference.huggingface.co/models"


def _huggingface_request(model: str, body: bytes, content_type: str):
    if not HF_API_TOKEN:
        raise HTTPException(
            status_code=500,
            detail=(
                "Hugging Face API token is not configured. "
                "Set HUGGINGFACE_API_TOKEN or HF_API_TOKEN in your environment."
            ),
        )

    url = f"{HF_BASE_URL}/{model}"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": content_type,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(raw.decode(charset))
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=error.code,
            detail=json.loads(payload) if payload else error.reason,
        )
    except urllib.error.URLError as error:
        raise HTTPException(status_code=503, detail=str(error))


@router.post("/infer")
async def infer_skin(image: UploadFile = File(...)):
    image_bytes = await image.read()
    content_type = image.content_type or "application/octet-stream"
    result = await asyncio.to_thread(
        _huggingface_request,
        HF_SKIN_MODEL,
        image_bytes,
        content_type,
    )
    return {"model": HF_SKIN_MODEL, "inference": result}
