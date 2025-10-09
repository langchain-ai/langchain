from langchain.agents.middleware.subagents import TASK_TOOL_DESCRIPTION, SubAgentMiddleware

class TestSubagentMiddleware:
    """Test the SubagentMiddleware class."""

    def test_subagent_middleware_init(self):
        middleware = SubAgentMiddleware(
            default_subagent_model="gpt-4o-mini",
        )
        assert middleware is not None
        assert middleware.system_prompt_extension == None
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"
        assert middleware.tools[0].description == TASK_TOOL_DESCRIPTION.format(other_agents="")

    def test_default_subagent_with_tools(self):
        middleware = SubAgentMiddleware(
            default_subagent_model="gpt-4o-mini",
            default_subagent_tools=[],
        )
        assert middleware is not None
        assert middleware.system_prompt_extension == None

    def test_default_subagent_custom_system_prompt_extension(self):
        middleware = SubAgentMiddleware(
            default_subagent_model="gpt-4o-mini",
            default_subagent_tools=[],
            system_prompt_extension="Use the task tool to call a subagent.",
        )
        assert middleware is not None
        assert middleware.system_prompt_extension == "Use the task tool to call a subagent."
