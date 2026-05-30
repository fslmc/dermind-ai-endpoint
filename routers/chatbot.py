from pydantic import BaseModel
from fastapi import APIRouter
from wrappers.chat_service import generate_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    mental_state: str | None = None
    history: list[dict] | None = None  # [{"role": "user", "content": "..."}, ...]
    user_id: str | None = None  # Just for logging/tracking, not auth

@router.post("/chat")
async def chat(req: ChatRequest):
    reply = await generate_response(
        req.message,
        req.mental_state,
        history=req.history or []
    )
    return {"response": reply}