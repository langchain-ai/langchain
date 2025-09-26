from typing import Optional
from typing import TypedDict

from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import StateT
from langchain.chat_models import BaseChatModel


# Should tuck this in to create an enum of tool names to select from
class ToolSelection(TypedDict):
    """Selection of the most relevant tools."""

    selected_tools: list[str]


class LLMToolSelector(AgentMiddleware):
    """Middleware for selecting tools using an LLM-based strategy."""

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        system_prompt: str = "Your goal is to select the most relevant tool for answering the users query.",
        max_tools: Optional[int] = None,
        include_full_history: bool = False,
        # If the model selects incorrect tools (e.g., ones that don't exist) due to
        # hallucination, we can retry the selection process.
        max_retries: int = 1,
        # Perhaps parameterize with behavior to ignore incorrect tool names
    ) -> None:
        self.model = model
        self.max_tools = max_tools
        self.system_prompt = system_prompt
        self.include_full_history = include_full_history

    def modify_model_request(
        self,
        request: ModelRequest,
        state: StateT,  # noqa: ARG002
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> ModelRequest:
        """Modify the model request to include tool selection logic."""
        if self.max_tools is not None and self.max_tools <= 0:
            raise ValueError("max_tools must be a positive integer or None.")

        tool_names = [tool.name for tool in request.tools]
        tool_descriptions = [tool.description for tool in request.tools]

        tool_representation = "\n".join(
            f"- {name}: {desc}" for name, desc in zip(tool_names, tool_descriptions)
        )

        system_message = (
            f"You are an agent that can use the following tools:\n{tool_representation}\n"
            "Select the most relevant tool(s) to answer the user's query."
        )

        if self.include_full_history:
            user_messages = [msg["content"] for msg in request.messages if msg["role"] == "user"]
            full_history = "\n".join(user_messages)
            system_message += f"\nThe full conversation history is:\n{full_history}"

        if self.max_tools is not None:
            system_message += f" You can select up to {self.max_tools} tools."

        model = self.model.with_structured_output(ToolSelection)

        response = model.invoke(
            {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": request.messages[-1]["content"]},
                ],
            }
        )

        # Check if all tool selections are valid
        tool_names = response["selected_tools"]
        selected_tools = [tool for tool in request.tools if tool.name in tool_names]

        request.tools = selected_tools
        return request
