from fastapi import APIRouter, Depends, HTTPException
from .chatbot import get_chatbot_response

router = APIRouter()

@router.get("/chat/")
async def chat(user_input: str):
    response = get_chatbot_response(user_input)
    return {"response": response}
