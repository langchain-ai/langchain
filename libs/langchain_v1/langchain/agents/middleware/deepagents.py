from langchain.agents.types import AgentMiddleware, AgentState, ModelRequest
from typing import NotRequired, Annotated
from typing import Literal
from typing_extensions import TypedDict


class Todo(TypedDict):
    """Todo to track."""

    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(l, r):
    if l is None:
        return r
    elif r is None:
        return l
    else:
        return {**l, **r}


class DeepAgentState(AgentState):
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]


from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import Annotated, Union
from langgraph.prebuilt import InjectedState

def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """write todos"""
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files"""
    return list(state.get("files", {}).keys())

class DeepAgentMiddleware(AgentMiddleware):

    state_schema = DeepAgentState

    def __init__(self, subagents: list = []):
        self.subagents = subagents

    @property
    def tools(self):
        return [write_todos, ls] + self.subagents

    def modify_model_request(self, request: ModelRequest, state: DeepAgentState) -> ModelRequest:
        if request.system_prompt:
            request.system_prompt += "\n\nUse the todo tool to plan as needed"
        else:
            request.system_prompt = "Use the todo tool to plan as needed"
        return request
