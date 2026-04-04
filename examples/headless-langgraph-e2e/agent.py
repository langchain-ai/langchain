"""LangGraph dev graph: `create_agent` with a headless (interrupting) tool.

Headless tools (`browser_navigate`, `geolocation_get`) have no local implementation;
they call LangGraph `interrupt()` so the client can execute them and resume with a result.

Run the API: ``uv run langgraph dev`` from this directory (see README).
"""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

# Model id for init_chat_model (requires OPENAI_API_KEY in .env for OpenAI models).
_model = os.environ.get("OPENAI_MODEL", "openai:gpt-4o-mini")


class BrowserNavigateArgs(BaseModel):
    """Arguments for the headless browser navigation tool."""

    url: str = Field(..., description="URL to open in the headless browser.")


class GeolocationGetArgs(BaseModel):
    """Arguments for the headless geolocation tool."""

    high_accuracy: bool | None = Field(
        default=None,
        description="If true, request high-accuracy GPS from the browser when supported.",
    )


browser_navigate = tool(
    name="browser_navigate",
    description=(
        "Navigate a headless browser to a URL. Execution is delegated to the client: "
        "the graph pauses until the client resumes with the page result."
    ),
    args_schema=BrowserNavigateArgs,
)

geolocation_get = tool(
    name="geolocation_get",
    description=(
        "Get the user's current GPS coordinates using the browser Geolocation API. "
        "The client shows their position on a map (OpenStreetMap). "
        "The browser may prompt for permission the first time."
    ),
    args_schema=GeolocationGetArgs,
)

graph = create_agent(
    _model,
    tools=[browser_navigate, geolocation_get],
    system_prompt=(
        "You are a helpful assistant. When the user asks to open, visit, or browse "
        "a URL, call browser_navigate with that URL. When they ask where they are, "
        "their location, or to show them on a map, call geolocation_get. "
        "After the client returns out-of-process results, reply briefly using that information."
    ),
    # No custom checkpointer: LangGraph API (`langgraph dev`) supplies persistence.
)
