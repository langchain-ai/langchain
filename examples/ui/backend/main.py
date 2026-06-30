"""FastAPI backend exposing the four LangChain v1 example agents over HTTP.

Endpoints (all under `/api`):

- `GET  /api/health`  — liveness + whether the OpenAI key is configured.
- `POST /api/react`   — ReAct tool agent: question in, answer + tool calls out.
- `POST /api/extract` — structured extraction: free text in, typed `Person` out.
- `POST /api/chat`    — memory chatbot: per-`thread_id` multi-turn conversation.
- `POST /api/plan`    — planner agent: goal in, todo plan + final answer out.

Run it with `run.sh`, or directly from `libs/langchain_v1`:

    uv run --with langchain-openai --with fastapi --with "uvicorn[standard]" \\
        python -m uvicorn main:app --app-dir ../../examples/ui/backend --port 8000
"""

from __future__ import annotations

import os

from agents import memory_agent, planner_agent, react_agent, structured_agent
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import ToolMessage
from pydantic import BaseModel

app = FastAPI(title="LangChain v1 examples API")

# The Vite dev server proxies `/api`, so CORS is not strictly needed in dev; allow
# all origins anyway so the frontend also works when served from another host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_key() -> None:
    """Fail fast with a clear message if the server has no OpenAI key."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set on the server. Restart it with the key.",
        )


class ReactRequest(BaseModel):
    """A single question for the ReAct tool agent."""

    question: str


class ExtractRequest(BaseModel):
    """Free text to extract a `Person` from."""

    text: str


class ChatRequest(BaseModel):
    """One chat turn, tied to a conversation thread."""

    thread_id: str
    message: str


class PlanRequest(BaseModel):
    """A multi-step goal for the planner agent."""

    goal: str


@app.get("/api/health")
def health() -> dict[str, bool]:
    """Report liveness and whether the OpenAI key is configured."""
    return {"ok": True, "openai_key": bool(os.environ.get("OPENAI_API_KEY"))}


@app.post("/api/react")
def react(req: ReactRequest) -> dict[str, object]:
    """Answer a question with the ReAct tool agent, surfacing any tool calls."""
    _require_key()
    result = react_agent.invoke(
        {"messages": [{"role": "user", "content": req.question}]}
    )
    tool_calls = [
        {"tool": m.name, "result": m.content}
        for m in result["messages"]
        if isinstance(m, ToolMessage)
    ]
    return {"answer": result["messages"][-1].content, "tool_calls": tool_calls}


@app.post("/api/extract")
def extract(req: ExtractRequest) -> dict[str, object]:
    """Extract a typed `Person` from free text via `response_format`."""
    _require_key()
    result = structured_agent.invoke(
        {"messages": [{"role": "user", "content": req.text}]}
    )
    return result["structured_response"].model_dump()


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict[str, str]:
    """Continue a conversation; memory is keyed by `thread_id`."""
    _require_key()
    config = {"configurable": {"thread_id": req.thread_id}}
    result = memory_agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
    )
    return {"reply": result["messages"][-1].content}


@app.post("/api/plan")
def plan(req: PlanRequest) -> dict[str, object]:
    """Decompose a goal into a todo plan and return it with the final answer."""
    _require_key()
    result = planner_agent.invoke(
        {"messages": [{"role": "user", "content": req.goal}]}
    )
    return {"todos": result.get("todos") or [], "answer": result["messages"][-1].content}
