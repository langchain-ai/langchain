"""API v1 router configuration."""

from fastapi import APIRouter

from app.api.v1 import agents, auth, chat, llm_configs

api_router = APIRouter()

# Include sub-routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(agents.router, prefix="/agents", tags=["Agents"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(llm_configs.router, prefix="/llm-configs", tags=["LLM Configs"])
