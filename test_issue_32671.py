"""
Test case to validate that the fix for issue #32671 works correctly.
This creates a proper test case that can be run as part of the test suite.
"""

import asyncio
from unittest.mock import Mock, patch
from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.agents import AgentAction, AgentStep
from langchain.agents.agent import AgentExecutor
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.agents import AgentFinish


@tool
def config_test_tool(input_text: str, config: RunnableConfig = None) -> str:
    """Tool that returns information about the config it receives."""
    if config is None:
        return "CONFIG_IS_NONE"
    else:
        return f"CONFIG_RECEIVED_WITH_KEYS:{','.join(sorted(config.keys()))}"


class TestIssue32671ConfigPassing:
    """Test cases for verifying config is passed to tools in AgentExecutor."""
    
    def test_perform_agent_action_passes_config(self):
        """Test that _perform_agent_action passes config to tools."""
        print("Testing _perform_agent_action with config...")
        
        # Create a minimal AgentExecutor
        llm = FakeListLLM(responses=["test"])
        tools = [config_test_tool]
        
        # Create a mock agent to avoid complex setup
        mock_agent = Mock()
        agent_executor = AgentExecutor(agent=mock_agent, tools=tools)
        
        # Create test config
        test_config = {
            "configurable": {"session_id": "test-123"},
            "tags": ["test"],
            "metadata": {"source": "test"}
        }
        
        # Create an agent action to test
        action = AgentAction(
            tool="config_test_tool",
            tool_input={"input_text": "test"},
            log="Testing config passing"
        )
        
        # Create name_to_tool_map
        name_to_tool_map = {tool.name: tool for tool in tools}
        
        # Call _perform_agent_action with config
        result = agent_executor._perform_agent_action(
            action=action,
            name_to_tool_map=name_to_tool_map,
            color_mapping={},
            config=test_config
        )
        
        # Verify the tool received the config
        assert isinstance(result, AgentStep), f"Expected AgentStep, got {type(result)}"
        assert "CONFIG_RECEIVED_WITH_KEYS" in result.observation, f"Config not passed. Got: {result.observation}"
        assert "configurable" in result.observation, f"Missing configurable key. Got: {result.observation}"
        assert "tags" in result.observation, f"Missing tags key. Got: {result.observation}"
        assert "metadata" in result.observation, f"Missing metadata key. Got: {result.observation}"
        
        print(f"✅ Tool received config correctly: {result.observation}")
    
    def test_perform_agent_action_without_config(self):
        """Test that _perform_agent_action works when config is None."""
        print("Testing _perform_agent_action without config...")
        
        # Create a minimal AgentExecutor
        llm = FakeListLLM(responses=["test"])
        tools = [config_test_tool]
        
        # Create a mock agent
        mock_agent = Mock()
        agent_executor = AgentExecutor(agent=mock_agent, tools=tools)
        
        # Create an agent action to test
        action = AgentAction(
            tool="config_test_tool",
            tool_input={"input_text": "test"},
            log="Testing without config"
        )
        
        # Create name_to_tool_map
        name_to_tool_map = {tool.name: tool for tool in tools}
        
        # Call _perform_agent_action without config
        result = agent_executor._perform_agent_action(
            action=action,
            name_to_tool_map=name_to_tool_map,
            color_mapping={},
            config=None
        )
        
        # Verify the tool got None for config
        assert isinstance(result, AgentStep), f"Expected AgentStep, got {type(result)}"
        assert "CONFIG_IS_NONE" in result.observation, f"Expected CONFIG_IS_NONE. Got: {result.observation}"
        
        print(f"✅ Tool correctly received None config: {result.observation}")
    
    async def test_aperform_agent_action_passes_config(self):
        """Test that _aperform_agent_action passes config to tools."""
        print("Testing _aperform_agent_action with config...")
        
        # Create a minimal AgentExecutor
        llm = FakeListLLM(responses=["test"])
        tools = [config_test_tool]
        
        # Create a mock agent
        mock_agent = Mock()
        agent_executor = AgentExecutor(agent=mock_agent, tools=tools)
        
        # Create test config
        test_config = {
            "configurable": {"session_id": "async-test-123"},
            "tags": ["async-test"],
            "metadata": {"source": "async-test"}
        }
        
        # Create an agent action to test
        action = AgentAction(
            tool="config_test_tool",
            tool_input={"input_text": "async test"},
            log="Testing async config passing"
        )
        
        # Create name_to_tool_map
        name_to_tool_map = {tool.name: tool for tool in tools}
        
        # Call _aperform_agent_action with config
        result = await agent_executor._aperform_agent_action(
            action=action,
            name_to_tool_map=name_to_tool_map,
            color_mapping={},
            config=test_config
        )
        
        # Verify the tool received the config
        assert isinstance(result, AgentStep), f"Expected AgentStep, got {type(result)}"
        assert "CONFIG_RECEIVED_WITH_KEYS" in result.observation, f"Config not passed. Got: {result.observation}"
        assert "configurable" in result.observation, f"Missing configurable key. Got: {result.observation}"
        assert "tags" in result.observation, f"Missing tags key. Got: {result.observation}"
        assert "metadata" in result.observation, f"Missing metadata key. Got: {result.observation}"
        
        print(f"✅ Async tool received config correctly: {result.observation}")


if __name__ == "__main__":
    # Run the tests directly
    
    test_case = TestIssue32671ConfigPassing()
    
    try:
        print("=" * 60)
        print("Testing fix for issue #32671: Config not passed to tools")
        print("=" * 60)
        
        print("\n1. Testing synchronous config passing...")
        test_case.test_perform_agent_action_passes_config()
        
        print("\n2. Testing behavior without config...")
        test_case.test_perform_agent_action_without_config()
        
        print("\n3. Testing asynchronous config passing...")
        asyncio.run(test_case.test_aperform_agent_action_passes_config())
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("The fix for issue #32671 is working correctly.")
        print("Config is now properly passed to tools in AgentExecutor.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
