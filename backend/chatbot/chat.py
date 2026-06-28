"""
chatbot/chat.py
---------------
Chat router stub — LLM integration is added in Issue #7.
The router is registered here so main.py can import it without error;
the /chat endpoint returns a placeholder until Issue #7 is complete.
"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/chat", tags=["Chat"])
async def chat():
    return {
        "message": "Chat endpoint not yet implemented — coming in Issue #7.",
        "status": "stub",
    }
