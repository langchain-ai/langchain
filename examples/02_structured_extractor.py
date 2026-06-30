"""Structured extraction: unstructured text to a typed object (one-shot).

Demonstrates `response_format`: pass a Pydantic model to `create_agent` and the agent
returns a validated instance of it under `result["structured_response"]`. The
framework re-prompts automatically if the model's first output doesn't match the
schema.

Run from `libs/langchain_v1`:

    uv run --with langchain-openai python ../../examples/02_structured_extractor.py

Requires `OPENAI_API_KEY` in the environment.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

MODEL = "openai:gpt-4o-mini"


class Person(BaseModel):
    """A person described in free text."""

    name: str = Field(description="The person's full name.")
    age: int = Field(description="The person's age in years.")
    email: str = Field(description="The person's email address.")


def build_agent() -> CompiledStateGraph:
    """Build an agent that returns a `Person` as structured output."""
    model = init_chat_model(MODEL)
    return create_agent(model, tools=[], response_format=Person)


def main() -> None:
    """Extract a `Person` from one hardcoded sentence and print the typed object."""
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY before running this example.")

    agent = build_agent()
    text = (
        "Hi, I'm Ada Lovelace, I'm 36 years old, and you can reach me at ada@analytical-engine.org."
    )
    print(f"Input: {text}\n")
    result = agent.invoke({"messages": [{"role": "user", "content": text}]})
    person = result["structured_response"]
    print(f"Extracted (type {type(person).__name__}):")
    print(f"  name  = {person.name}")
    print(f"  age   = {person.age}")
    print(f"  email = {person.email}")


if __name__ == "__main__":
    main()
