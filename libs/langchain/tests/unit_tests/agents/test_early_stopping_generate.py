"""Test for early_stopping_method="generate" support in modern agents."""

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnableLambda

from langchain.agents.agent import (
    BaseSingleActionAgent, 
    BaseMultiActionAgent,
    RunnableAgent, 
    RunnableMultiActionAgent,
)


def test_base_single_action_agent_supports_generate():
    """Test that BaseSingleActionAgent supports early_stopping_method='generate'."""
    
    class TestAgent(BaseSingleActionAgent):
        @property 
        def input_keys(self):
            return ["input"]
            
        def plan(self, intermediate_steps, callbacks=None, **kwargs):
            return AgentFinish({"output": "done"}, "")
            
        async def aplan(self, intermediate_steps, callbacks=None, **kwargs):
            return AgentFinish({"output": "done"}, "")
    
    agent = TestAgent()
    
    # Test that 'generate' is supported (no ValueError should be raised)
    result = agent.return_stopped_response("generate", [])
    assert isinstance(result, AgentFinish)
    assert "output" in result.return_values
    
    # Test that 'force' still works
    result = agent.return_stopped_response("force", [])
    assert isinstance(result, AgentFinish)
    assert "iteration limit" in result.return_values["output"].lower()


def test_base_multi_action_agent_supports_generate():
    """Test that BaseMultiActionAgent supports early_stopping_method='generate'."""
    
    class TestMultiAgent(BaseMultiActionAgent):
        @property 
        def input_keys(self):
            return ["input"]
            
        def plan(self, intermediate_steps, callbacks=None, **kwargs):
            return AgentFinish({"output": "done"}, "")
            
        async def aplan(self, intermediate_steps, callbacks=None, **kwargs):
            return AgentFinish({"output": "done"}, "")
    
    agent = TestMultiAgent()
    
    # Test that 'generate' is supported (no ValueError should be raised)
    result = agent.return_stopped_response("generate", [])
    assert isinstance(result, AgentFinish)
    assert "output" in result.return_values
    
    # Test that 'force' still works
    result = agent.return_stopped_response("force", [])
    assert isinstance(result, AgentFinish)
    assert "max iterations" in result.return_values["output"].lower()


def test_runnable_agent_generate_calls_runnable():
    """Test that RunnableAgent properly calls the runnable for generate method."""
    
    def mock_invoke(inputs):
        return AgentFinish({"output": "Generated final answer!"}, "final")
    
    mock_runnable = RunnableLambda(mock_invoke)
    agent = RunnableAgent(runnable=mock_runnable, stream_runnable=False)
    
    # Test intermediate steps
    intermediate_steps = [
        (AgentAction("tool1", "input1", "log1"), "observation1"),
    ]
    
    # Test that 'generate' calls the runnable and returns its result
    result = agent.return_stopped_response("generate", intermediate_steps, input="test")
    assert isinstance(result, AgentFinish)
    assert result.return_values["output"] == "Generated final answer!"
    assert result.log == "final"


def test_runnable_multi_action_agent_generate_calls_runnable():
    """Test that RunnableMultiActionAgent properly calls the runnable for generate method."""
    
    def mock_invoke(inputs):
        return AgentFinish({"output": "Multi-action generated final answer!"}, "multi-final")
    
    mock_runnable = RunnableLambda(mock_invoke)
    agent = RunnableMultiActionAgent(runnable=mock_runnable, stream_runnable=False)
    
    # Test intermediate steps
    intermediate_steps = [
        (AgentAction("tool1", "input1", "log1"), "observation1"),
    ]
    
    # Test that 'generate' calls the runnable and returns its result
    result = agent.return_stopped_response("generate", intermediate_steps, input="test")
    assert isinstance(result, AgentFinish)
    assert result.return_values["output"] == "Multi-action generated final answer!"
    assert result.log == "multi-final"


def test_runnable_agent_generate_handles_exceptions():
    """Test that RunnableAgent gracefully handles runnable exceptions."""
    
    def failing_invoke(inputs):
        raise Exception("Mock runnable error")
    
    mock_runnable = RunnableLambda(failing_invoke)
    agent = RunnableAgent(runnable=mock_runnable, stream_runnable=False)
    
    # Test that exceptions are handled gracefully with fallback message
    result = agent.return_stopped_response("generate", [], input="test")
    assert isinstance(result, AgentFinish)
    assert "Unable to generate final response" in result.return_values["output"]


def test_unsupported_early_stopping_method_still_raises():
    """Test that unsupported early stopping methods still raise ValueError."""
    
    class TestAgent(BaseSingleActionAgent):
        @property 
        def input_keys(self):
            return ["input"]
            
        def plan(self, intermediate_steps, callbacks=None, **kwargs):
            return AgentFinish({"output": "done"}, "")
            
        async def aplan(self, intermediate_steps, callbacks=None, **kwargs):
            return AgentFinish({"output": "done"}, "")
    
    agent = TestAgent()
    
    # Test that unsupported methods still raise ValueError
    try:
        agent.return_stopped_response("unsupported_method", [])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unsupported early_stopping_method" in str(e)