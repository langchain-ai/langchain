"""Generate Mermaid diagrams for all middleware hook combinations with binary naming.

Binary naming scheme (6 bits):
- Bit 0: has_tools
- Bit 1: before_agent
- Bit 2: before_model
- Bit 3: after_model
- Bit 4: after_agent
- Bit 5: wrap_model
- Bit 6: wrap_tool

Example: "0100110" = no tools, before_agent, no before_model, after_model, after_agent, no wrap_model, no wrap_tool
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.tools import tool

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState


class DemoModel(SimpleChatModel):
    """Demo model for generating diagrams."""

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        return "Demo response"

    @property
    def _llm_type(self) -> str:
        return "demo"


@tool
def demo_tool(query: str) -> str:
    """Demo tool for testing."""
    return f"Result for: {query}"


class BeforeModelMiddleware(AgentMiddleware):
    """Middleware with only before_model hook."""

    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        return None


class AfterModelMiddleware(AgentMiddleware):
    """Middleware with only after_model hook."""

    def after_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        return None


class BeforeAgentMiddleware(AgentMiddleware):
    """Middleware with only before_agent hook."""

    def before_agent(self, state: AgentState, runtime) -> dict[str, Any] | None:
        return None


class AfterAgentMiddleware(AgentMiddleware):
    """Middleware with only after_agent hook."""

    def after_agent(self, state: AgentState, runtime) -> dict[str, Any] | None:
        return None


class WrapModelMiddleware(AgentMiddleware):
    """Middleware with wrap_model_call hook."""

    def wrap_model_call(self, request, handler):
        return handler(request)


class WrapToolMiddleware(AgentMiddleware):
    """Middleware with wrap_tool_call hook."""

    def wrap_tool_call(self, request, handler):
        return handler(request)


def binary_name(
    has_tools: bool,
    before_agent: bool,
    before_model: bool,
    after_model: bool,
    after_agent: bool,
    wrap_model: bool,
    wrap_tool: bool,
) -> str:
    """Generate binary name for configuration.

    Bit positions:
    - Bit 0: has_tools
    - Bit 1: before_agent
    - Bit 2: before_model
    - Bit 3: after_model
    - Bit 4: after_agent
    - Bit 5: wrap_model
    - Bit 6: wrap_tool
    """
    bits = [
        "1" if has_tools else "0",
        "1" if before_agent else "0",
        "1" if before_model else "0",
        "1" if after_model else "0",
        "1" if after_agent else "0",
        "1" if wrap_model else "0",
        "1" if wrap_tool else "0",
    ]
    return "".join(bits)


def generate_all_diagrams() -> dict[str, str]:
    """Generate Mermaid diagrams for all 128 possible hook combinations.

    Returns:
        Dictionary mapping binary configuration names to Mermaid diagram strings.
    """
    model = DemoModel()
    diagrams = {}

    # Generate all 128 combinations (2^7)
    for i in range(128):
        # Extract bits
        has_tools = bool(i & 0b0000001)
        before_agent = bool(i & 0b0000010)
        before_model = bool(i & 0b0000100)
        after_model = bool(i & 0b0001000)
        after_agent = bool(i & 0b0010000)
        wrap_model = bool(i & 0b0100000)
        wrap_tool = bool(i & 0b1000000)

        # Build middleware list
        middleware = []
        if before_agent:
            middleware.append(BeforeAgentMiddleware())
        if before_model:
            middleware.append(BeforeModelMiddleware())
        if after_model:
            middleware.append(AfterModelMiddleware())
        if after_agent:
            middleware.append(AfterAgentMiddleware())
        if wrap_model:
            middleware.append(WrapModelMiddleware())
        if wrap_tool:
            middleware.append(WrapToolMiddleware())

        # Generate binary name
        name = binary_name(
            has_tools,
            before_agent,
            before_model,
            after_model,
            after_agent,
            wrap_model,
            wrap_tool,
        )

        # Create agent and generate diagram
        tools = [demo_tool] if has_tools else []
        agent = create_agent(model=model, tools=tools, middleware=middleware)

        mermaid = agent.get_graph().draw_mermaid()
        diagrams[name] = mermaid

        print(f"Generated: {name} (tools={int(has_tools)}, ba={int(before_agent)}, "
              f"bm={int(before_model)}, am={int(after_model)}, aa={int(after_agent)}, "
              f"wm={int(wrap_model)}, wt={int(wrap_tool)})")

    return diagrams


def save_diagrams_to_json(diagrams: dict[str, str], output_path: Path) -> None:
    """Save diagrams to a JSON file.

    Args:
        diagrams: Dictionary mapping binary names to Mermaid diagrams.
        output_path: Path where the JSON file should be saved.
    """
    output_path.write_text(json.dumps(diagrams, indent=2))
    print(f"\nSaved {len(diagrams)} diagrams to {output_path}")


def main() -> None:
    """Generate all diagrams and save to JSON."""
    diagrams = generate_all_diagrams()

    # Save to JSON file
    output_dir = Path(__file__).parent.parent / "docs" / "middleware_diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "diagrams.json"
    save_diagrams_to_json(diagrams, json_path)


if __name__ == "__main__":
    main()
