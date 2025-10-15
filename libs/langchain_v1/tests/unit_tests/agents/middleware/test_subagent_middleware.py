from langchain.agents.middleware.subagents import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    TASK_TOOL_DESCRIPTION,
    SubAgentMiddleware,
)

import pytest


@pytest.mark.requires("langchain_openai")
class TestSubagentMiddleware:
    """Test the SubagentMiddleware class."""

    def test_subagent_middleware_init(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
        )
        assert middleware is not None
        assert middleware.system_prompt is None
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"
        expected_desc = TASK_TOOL_DESCRIPTION.format(
            available_agents=f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}"
        )
        assert middleware.tools[0].description == expected_desc

    def test_default_subagent_with_tools(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
            default_tools=[],
        )
        assert middleware is not None
        assert middleware.system_prompt is None

    def test_default_subagent_custom_system_prompt(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
            default_tools=[],
            system_prompt="Use the task tool to call a subagent.",
        )
        assert middleware is not None
        assert middleware.system_prompt == "Use the task tool to call a subagent."
