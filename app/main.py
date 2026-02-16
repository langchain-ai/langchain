"""
Orcest.ai - The Self-Adaptive LLM Orchestrator API
Platform for reliable AI agents
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Orcest.ai",
    description="The Self-Adaptive LLM Orchestrator platform for reliable AI agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    return {"status": "healthy", "service": "orcest.ai"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Orcest.ai",
        "description": "The Self-Adaptive LLM Orchestrator platform for reliable AI agents",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/api/info")
async def api_info():
    """API information."""
    return {
        "platform": "orcest.ai",
        "capabilities": ["llm-orchestration", "agents", "chains"],
        "docs": "https://docs.langchain.com/",
    }
