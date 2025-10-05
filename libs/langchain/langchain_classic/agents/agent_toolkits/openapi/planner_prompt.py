from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.openapi.planner_prompt import (
        API_CONTROLLER_PROMPT,
        API_CONTROLLER_TOOL_DESCRIPTION,
        API_CONTROLLER_TOOL_NAME,
        API_ORCHESTRATOR_PROMPT,
        API_PLANNER_PROMPT,
        API_PLANNER_TOOL_DESCRIPTION,
        API_PLANNER_TOOL_NAME,
        PARSING_DELETE_PROMPT,
        PARSING_GET_PROMPT,
        PARSING_PATCH_PROMPT,
        PARSING_POST_PROMPT,
        PARSING_PUT_PROMPT,
        REQUESTS_DELETE_TOOL_DESCRIPTION,
        REQUESTS_GET_TOOL_DESCRIPTION,
        REQUESTS_PATCH_TOOL_DESCRIPTION,
        REQUESTS_POST_TOOL_DESCRIPTION,
        REQUESTS_PUT_TOOL_DESCRIPTION,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "API_CONTROLLER_PROMPT": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "API_CONTROLLER_TOOL_DESCRIPTION": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "API_CONTROLLER_TOOL_NAME": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "API_ORCHESTRATOR_PROMPT": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "API_PLANNER_PROMPT": ("langchain_community.agent_toolkits.openapi.planner_prompt"),
    "API_PLANNER_TOOL_DESCRIPTION": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "API_PLANNER_TOOL_NAME": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "PARSING_DELETE_PROMPT": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "PARSING_GET_PROMPT": ("langchain_community.agent_toolkits.openapi.planner_prompt"),
    "PARSING_PATCH_PROMPT": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "PARSING_POST_PROMPT": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "PARSING_PUT_PROMPT": ("langchain_community.agent_toolkits.openapi.planner_prompt"),
    "REQUESTS_DELETE_TOOL_DESCRIPTION": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "REQUESTS_GET_TOOL_DESCRIPTION": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "REQUESTS_PATCH_TOOL_DESCRIPTION": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "REQUESTS_POST_TOOL_DESCRIPTION": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
    "REQUESTS_PUT_TOOL_DESCRIPTION": (
        "langchain_community.agent_toolkits.openapi.planner_prompt"
    ),
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "API_CONTROLLER_PROMPT",
    "API_CONTROLLER_TOOL_DESCRIPTION",
    "API_CONTROLLER_TOOL_NAME",
    "API_ORCHESTRATOR_PROMPT",
    "API_PLANNER_PROMPT",
    "API_PLANNER_TOOL_DESCRIPTION",
    "API_PLANNER_TOOL_NAME",
    "PARSING_DELETE_PROMPT",
    "PARSING_GET_PROMPT",
    "PARSING_PATCH_PROMPT",
    "PARSING_POST_PROMPT",
    "PARSING_PUT_PROMPT",
    "REQUESTS_DELETE_TOOL_DESCRIPTION",
    "REQUESTS_GET_TOOL_DESCRIPTION",
    "REQUESTS_PATCH_TOOL_DESCRIPTION",
    "REQUESTS_POST_TOOL_DESCRIPTION",
    "REQUESTS_PUT_TOOL_DESCRIPTION",
]
