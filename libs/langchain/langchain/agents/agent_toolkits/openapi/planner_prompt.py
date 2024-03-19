from typing import Any

DEPRECATED_IMPORTS = [
    "API_PLANNER_PROMPT",
    "API_PLANNER_TOOL_NAME",
    "API_PLANNER_TOOL_DESCRIPTION",
    "API_CONTROLLER_PROMPT",
    "API_CONTROLLER_TOOL_NAME",
    "API_CONTROLLER_TOOL_DESCRIPTION",
    "API_ORCHESTRATOR_PROMPT",
    "REQUESTS_GET_TOOL_DESCRIPTION",
    "PARSING_GET_PROMPT",
    "REQUESTS_POST_TOOL_DESCRIPTION",
    "PARSING_POST_PROMPT",
    "REQUESTS_PATCH_TOOL_DESCRIPTION",
    "PARSING_PATCH_PROMPT",
    "REQUESTS_PUT_TOOL_DESCRIPTION",
    "PARSING_PUT_PROMPT",
    "REQUESTS_DELETE_TOOL_DESCRIPTION",
    "PARSING_DELETE_PROMPT",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.agent_toolkits.openapi.planner_prompt import {name}`"  # noqa: E501
        )

    raise AttributeError()
