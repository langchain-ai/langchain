"""Unit tests for agents."""

from typing import Any, List, Mapping, Optional

from pydantic import BaseModel

from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain.callbacks.base import CallbackManager
from langchain.llms.base import LLM
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


class FakeListLLM(LLM, BaseModel):
    """Fake LLM for testing that outputs elements of a list."""

    responses: List[str]
    i: int = -1

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Increment counter, and then return response in that index."""
        self.i += 1
        print(f"=== Mock Response #{self.i} ===")
        print(self.responses[self.i])
        return self.responses[self.i]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake_list"


def _get_agent(**kwargs: Any) -> AgentExecutor:
    """Get agent for testing."""
    bad_action_name = "BadAction"
    responses = [
        f"I'm turning evil\nAction: {bad_action_name}\nAction Input: misalignment",
        "Oh well\nAction: Final Answer\nAction Input: curses foiled again",
    ]
    fake_llm = FakeListLLM(responses=responses)
    tools = [
        Tool("Search", lambda x: x, "Useful for searching"),
        Tool("Lookup", lambda x: x, "Useful for looking up things in a table"),
    ]
    agent = initialize_agent(
        tools, fake_llm, agent="zero-shot-react-description", verbose=True, **kwargs
    )
    return agent


def test_agent_bad_action() -> None:
    """Test react chain when bad action given."""
    agent = _get_agent()
    output = agent.run("when was langchain made")
    assert output == "curses foiled again"


def test_agent_stopped_early() -> None:
    """Test react chain when bad action given."""
    agent = _get_agent(max_iterations=0)
    output = agent.run("when was langchain made")
    assert output == "Agent stopped due to max iterations."


def test_agent_with_callbacks_global() -> None:
    """Test react chain with callbacks by setting verbose globally."""
    import langchain

    langchain.verbose = True
    handler = FakeCallbackHandler()
    manager = CallbackManager(handlers=[handler])
    tool = "Search"
    responses = [
        f"FooBarBaz\nAction: {tool}\nAction Input: misalignment",
        "Oh well\nAction: Final Answer\nAction Input: curses foiled again",
    ]
    fake_llm = FakeListLLM(responses=responses, callback_manager=manager, verbose=True)
    tools = [
        Tool("Search", lambda x: x, "Useful for searching"),
    ]
    agent = initialize_agent(
        tools,
        fake_llm,
        agent="zero-shot-react-description",
        verbose=True,
        callback_manager=manager,
    )

    output = agent.run("when was langchain made")
    assert output == "curses foiled again"

    # 1 top level chain run runs, 2 LLMChain runs, 2 LLM runs, 1 tool run
    assert handler.chain_starts == handler.chain_ends == 3
    assert handler.llm_starts == handler.llm_ends == 2
    assert handler.tool_starts == handler.tool_ends == 1
    assert handler.starts == 6
    # 1 extra agent end
    assert handler.ends == 7
    assert handler.errors == 0
    # during LLMChain
    assert handler.text == 2


def test_agent_with_callbacks_local() -> None:
    """Test react chain with callbacks by setting verbose locally."""
    import langchain

    langchain.verbose = False
    handler = FakeCallbackHandler()
    manager = CallbackManager(handlers=[handler])
    tool = "Search"
    responses = [
        f"FooBarBaz\nAction: {tool}\nAction Input: misalignment",
        "Oh well\nAction: Final Answer\nAction Input: curses foiled again",
    ]
    fake_llm = FakeListLLM(responses=responses, callback_manager=manager, verbose=True)
    tools = [
        Tool("Search", lambda x: x, "Useful for searching"),
    ]
    agent = initialize_agent(
        tools,
        fake_llm,
        agent="zero-shot-react-description",
        verbose=True,
        callback_manager=manager,
    )

    agent.agent.llm_chain.verbose = True

    output = agent.run("when was langchain made")
    assert output == "curses foiled again"

    # 1 top level chain run, 2 LLMChain starts, 2 LLM runs, 1 tool run
    assert handler.chain_starts == handler.chain_ends == 3
    assert handler.llm_starts == handler.llm_ends == 2
    assert handler.tool_starts == handler.tool_ends == 1
    assert handler.starts == 6
    # 1 extra agent end
    assert handler.ends == 7
    assert handler.errors == 0
    # during LLMChain
    assert handler.text == 2


def test_agent_with_callbacks_not_verbose() -> None:
    """Test react chain with callbacks but not verbose."""
    import langchain

    langchain.verbose = False
    handler = FakeCallbackHandler()
    manager = CallbackManager(handlers=[handler])
    tool = "Search"
    responses = [
        f"FooBarBaz\nAction: {tool}\nAction Input: misalignment",
        "Oh well\nAction: Final Answer\nAction Input: curses foiled again",
    ]
    fake_llm = FakeListLLM(responses=responses, callback_manager=manager)
    tools = [
        Tool("Search", lambda x: x, "Useful for searching"),
    ]
    agent = initialize_agent(
        tools,
        fake_llm,
        agent="zero-shot-react-description",
        callback_manager=manager,
    )

    output = agent.run("when was langchain made")
    assert output == "curses foiled again"

    # 1 top level chain run, 2 LLMChain runs, 2 LLM runs, 1 tool run
    assert handler.starts == 0
    assert handler.ends == 0
    assert handler.errors == 0


def test_agent_tool_return_direct() -> None:
    """Test agent using tools that return directly."""
    tool = "Search"
    responses = [
        f"FooBarBaz\nAction: {tool}\nAction Input: misalignment",
        "Oh well\nAction: Final Answer\nAction Input: curses foiled again",
    ]
    fake_llm = FakeListLLM(responses=responses)
    tools = [
        Tool("Search", lambda x: x, "Useful for searching", return_direct=True),
    ]
    agent = initialize_agent(
        tools,
        fake_llm,
        agent="zero-shot-react-description",
    )

    output = agent.run("when was langchain made")
    assert output == "misalignment"


def test_agent_with_new_prefix_suffix() -> None:
    """Test agent initilization kwargs with new prefix and suffix."""
    fake_llm = FakeListLLM(
        responses=["FooBarBaz\nAction: Search\nAction Input: misalignment"]
    )
    tools = [
        Tool("Search", lambda x: x, "Useful for searching", return_direct=True),
    ]
    prefix = "FooBarBaz"

    suffix = "Begin now!\nInput: {input}\nThought: {agent_scratchpad}"

    agent = initialize_agent(
        tools=tools,
        llm=fake_llm,
        agent="zero-shot-react-description",
        agent_kwargs={"prefix": prefix, "suffix": suffix},
    )

    # avoids "BasePromptTemplate" has no attribute "template" error
    assert hasattr(agent.agent.llm_chain.prompt, "template")
    prompt_str = agent.agent.llm_chain.prompt.template
    assert prompt_str.startswith(prefix), "Prompt does not start with prefix"
    assert prompt_str.endswith(suffix), "Prompt does not end with suffix"
