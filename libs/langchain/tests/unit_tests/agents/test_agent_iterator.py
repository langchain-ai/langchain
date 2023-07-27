import pytest

from langchain.agents import (
    AgentExecutor,
    AgentExecutorIterator,
    AgentType,
    initialize_agent,
)
from langchain.agents.tools import Tool
from langchain.llms import FakeListLLM
from tests.unit_tests.agents.test_agent import _get_agent
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


def test_agent_iterator_bad_action() -> None:
    """Test react chain iterator when bad action given."""
    agent = _get_agent()
    agent_iter = agent.iter(inputs="when was langchain made")

    outputs = []
    for step in agent_iter:
        outputs.append(step)

    assert isinstance(outputs[-1], dict)
    assert outputs[-1]["output"] == "curses foiled again"


def test_agent_iterator_stopped_early() -> None:
    """
    Test react chain iterator when max iterations or
    max execution time is exceeded.
    """
    # iteration limit
    agent = _get_agent(max_iterations=1)
    agent_iter = agent.iter(inputs="when was langchain made")

    outputs = []
    for step in agent_iter:
        outputs.append(step)
    # NOTE: we don't use agent.run like in the test for the regular agent executor,
    # so the dict structure for outputs stays intact
    assert isinstance(outputs[-1], dict)
    assert (
        outputs[-1]["output"] == "Agent stopped due to iteration limit or time limit."
    )

    # execution time limit
    agent = _get_agent(max_execution_time=1e-5)
    agent_iter = agent.iter(inputs="when was langchain made")

    outputs = []
    for step in agent_iter:
        outputs.append(step)
    assert isinstance(outputs[-1], dict)
    assert (
        outputs[-1]["output"] == "Agent stopped due to iteration limit or time limit."
    )


@pytest.mark.asyncio
async def test_agent_async_iterator_stopped_early() -> None:
    """
    Test react chain async iterator when max iterations or
    max execution time is exceeded.
    """
    # iteration limit
    agent = _get_agent(max_iterations=1)
    agent_async_iter = agent.iter(inputs="when was langchain made", async_=True)

    outputs = []
    assert isinstance(agent_async_iter, AgentExecutorIterator)
    async for step in agent_async_iter:
        outputs.append(step)

    assert isinstance(outputs[-1], dict)
    assert (
        outputs[-1]["output"] == "Agent stopped due to iteration limit or time limit."
    )

    # execution time limit
    agent = _get_agent(max_execution_time=1e-5)
    agent_async_iter = agent.iter(inputs="when was langchain made", async_=True)
    assert isinstance(agent_async_iter, AgentExecutorIterator)

    outputs = []
    async for step in agent_async_iter:
        outputs.append(step)

    assert (
        outputs[-1]["output"] == "Agent stopped due to iteration limit or time limit."
    )


def test_agent_iterator_with_callbacks() -> None:
    """Test react chain iterator with callbacks by setting verbose globally."""
    handler1 = FakeCallbackHandler()
    handler2 = FakeCallbackHandler()
    bad_action_name = "BadAction"
    responses = [
        f"I'm turning evil\nAction: {bad_action_name}\nAction Input: misalignment",
        "Oh well\nFinal Answer: curses foiled again",
    ]
    fake_llm = FakeListLLM(cache=False, responses=responses, callbacks=[handler2])

    tools = [
        Tool(
            name="Search",
            func=lambda x: x,
            description="Useful for searching",
        ),
        Tool(
            name="Lookup",
            func=lambda x: x,
            description="Useful for looking up things in a table",
        ),
    ]

    agent = initialize_agent(
        tools, fake_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent_iter = agent.iter(inputs="when was langchain made", callbacks=[handler1])

    outputs = []
    for step in agent_iter:
        outputs.append(step)
    assert isinstance(outputs[-1], dict)
    assert outputs[-1]["output"] == "curses foiled again"

    # 1 top level chain run runs, 2 LLMChain runs, 2 LLM runs, 1 tool run
    assert handler1.chain_starts == handler1.chain_ends == 3
    assert handler1.llm_starts == handler1.llm_ends == 2
    assert handler1.tool_starts == 1
    assert handler1.tool_ends == 1
    # 1 extra agent action
    assert handler1.starts == 7
    # 1 extra agent end
    assert handler1.ends == 7
    print("h:", handler1)
    assert handler1.errors == 0
    # during LLMChain
    assert handler1.text == 2

    assert handler2.llm_starts == 2
    assert handler2.llm_ends == 2
    assert (
        handler2.chain_starts
        == handler2.tool_starts
        == handler2.tool_ends
        == handler2.chain_ends
        == 0
    )


@pytest.mark.asyncio
async def test_agent_async_iterator_with_callbacks() -> None:
    """Test react chain async iterator with callbacks by setting verbose globally."""
    handler1 = FakeCallbackHandler()
    handler2 = FakeCallbackHandler()

    bad_action_name = "BadAction"
    responses = [
        f"I'm turning evil\nAction: {bad_action_name}\nAction Input: misalignment",
        "Oh well\nFinal Answer: curses foiled again",
    ]
    fake_llm = FakeListLLM(cache=False, responses=responses, callbacks=[handler2])

    tools = [
        Tool(
            name="Search",
            func=lambda x: x,
            description="Useful for searching",
        ),
        Tool(
            name="Lookup",
            func=lambda x: x,
            description="Useful for looking up things in a table",
        ),
    ]

    agent = initialize_agent(
        tools, fake_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent_async_iter = agent.iter(
        inputs="when was langchain made",
        callbacks=[handler1],
        async_=True,
    )
    assert isinstance(agent_async_iter, AgentExecutorIterator)

    outputs = []
    async for step in agent_async_iter:
        outputs.append(step)

    assert outputs[-1]["output"] == "curses foiled again"

    # 1 top level chain run runs, 2 LLMChain runs, 2 LLM runs, 1 tool run
    assert handler1.chain_starts == handler1.chain_ends == 3
    assert handler1.llm_starts == handler1.llm_ends == 2
    assert handler1.tool_starts == 1
    assert handler1.tool_ends == 1
    # 1 extra agent action
    assert handler1.starts == 7
    # 1 extra agent end
    assert handler1.ends == 7
    assert handler1.errors == 0
    # during LLMChain
    assert handler1.text == 2

    assert handler2.llm_starts == 2
    assert handler2.llm_ends == 2
    assert (
        handler2.chain_starts
        == handler2.tool_starts
        == handler2.tool_ends
        == handler2.chain_ends
        == 0
    )


def test_agent_iterator_properties_and_setters() -> None:
    """Test properties and setters of AgentExecutorIterator."""
    agent = _get_agent()
    agent.tags = None
    agent_iter = agent.iter(inputs="when was langchain made")

    assert isinstance(agent_iter, AgentExecutorIterator)
    assert isinstance(agent_iter.inputs, dict)
    assert isinstance(agent_iter.callbacks, type(None))
    assert isinstance(agent_iter.tags, type(None))
    assert isinstance(agent_iter.agent_executor, AgentExecutor)

    agent_iter.inputs = "New input"  # type: ignore
    assert isinstance(agent_iter.inputs, dict)

    agent_iter.callbacks = [FakeCallbackHandler()]
    assert isinstance(agent_iter.callbacks, list)

    agent_iter.tags = ["test"]
    assert isinstance(agent_iter.tags, list)

    new_agent = _get_agent()
    agent_iter.agent_executor = new_agent
    assert isinstance(agent_iter.agent_executor, AgentExecutor)


def test_agent_iterator_reset() -> None:
    """Test reset functionality of AgentExecutorIterator."""
    agent = _get_agent()
    agent_iter = agent.iter(inputs="when was langchain made")
    assert isinstance(agent_iter, AgentExecutorIterator)

    # Perform one iteration
    next(agent_iter)

    # Check if properties are updated
    assert agent_iter.iterations == 1
    assert agent_iter.time_elapsed > 0.0
    assert agent_iter.intermediate_steps

    # Reset the iterator
    agent_iter.reset()

    # Check if properties are reset
    assert agent_iter.iterations == 0
    assert agent_iter.time_elapsed == 0.0
    assert not agent_iter.intermediate_steps


def test_agent_iterator_output_structure() -> None:
    """Test the output structure of AgentExecutorIterator."""
    agent = _get_agent()
    agent_iter = agent.iter(inputs="when was langchain made")

    for step in agent_iter:
        assert isinstance(step, dict)
        if "intermediate_step" in step:
            assert isinstance(step["intermediate_step"], list)
        elif "output" in step:
            assert isinstance(step["output"], str)
        else:
            assert False, "Unexpected output structure"


@pytest.mark.asyncio
async def test_agent_async_iterator_output_structure() -> None:
    """Test the async output structure of AgentExecutorIterator."""
    agent = _get_agent()
    agent_async_iter = agent.iter(inputs="when was langchain made", async_=True)

    assert isinstance(agent_async_iter, AgentExecutorIterator)
    async for step in agent_async_iter:
        assert isinstance(step, dict)
        if "intermediate_step" in step:
            assert isinstance(step["intermediate_step"], list)
        elif "output" in step:
            assert isinstance(step["output"], str)
        else:
            assert False, "Unexpected output structure"


def test_agent_iterator_empty_input() -> None:
    """Test AgentExecutorIterator with empty input."""
    agent = _get_agent()
    agent_iter = agent.iter(inputs="")

    outputs = []
    for step in agent_iter:
        outputs.append(step)

    assert isinstance(outputs[-1], dict)
    assert outputs[-1]["output"]  # Check if there is an output


def test_agent_iterator_custom_stopping_condition() -> None:
    """Test AgentExecutorIterator with a custom stopping condition."""
    agent = _get_agent()

    class CustomAgentExecutorIterator(AgentExecutorIterator):
        def _should_continue(self) -> bool:
            return self.iterations < 2  # Custom stopping condition

    agent_iter = CustomAgentExecutorIterator(agent, inputs="when was langchain made")

    outputs = []
    for step in agent_iter:
        outputs.append(step)

    assert len(outputs) == 2  # Check if the custom stopping condition is respected


def test_agent_iterator_failing_tool() -> None:
    """Test AgentExecutorIterator with a tool that raises an exception."""

    # Get agent for testing.
    bad_action_name = "FailingTool"
    responses = [
        f"I'm turning evil\nAction: {bad_action_name}\nAction Input: misalignment",
        "Oh well\nFinal Answer: curses foiled again",
    ]
    fake_llm = FakeListLLM(responses=responses)

    tools = [
        Tool(
            name="FailingTool",
            func=lambda x: 1 / 0,  # This tool will raise a ZeroDivisionError
            description="A tool that fails",
        ),
    ]

    agent = initialize_agent(
        tools, fake_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    agent_iter = agent.iter(inputs="when was langchain made")
    assert isinstance(agent_iter, AgentExecutorIterator)
    # initialise iterator
    iter(agent_iter)

    with pytest.raises(ZeroDivisionError):
        next(agent_iter)
