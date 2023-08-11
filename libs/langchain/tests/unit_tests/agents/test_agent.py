"""Unit tests for agents."""

from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.agents.tools import Tool
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


class FakeListLLM(LLM):
    """Fake LLM for testing that outputs elements of a list."""

    responses: List[str]
    i: int = -1

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Increment counter, and then return response in that index."""
        self.i += 1
        print(f"=== Mock Response #{self.i} ===")
        print(self.responses[self.i])
        return self.responses[self.i]

    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens in text."""
        return len(text.split())

    async def _acall(self, *args: Any, **kwargs: Any) -> str:
        return self._call(*args, **kwargs)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
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
        "Oh well\nFinal Answer: curses foiled again",
    ]
    fake_llm = FakeListLLM(cache=False, responses=responses)

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
        tools,
        fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        **kwargs,
    )
    return agent


def test_agent_bad_action() -> None:
    """Test react chain when bad action given."""
    agent = _get_agent()
    output = agent.run("when was langchain made")
    assert output == "curses foiled again"


def test_agent_stopped_early() -> None:
    """Test react chain when max iterations or max execution time is exceeded."""
    # iteration limit
    agent = _get_agent(max_iterations=0)
    output = agent.run("when was langchain made")
    assert output == "Agent stopped due to iteration limit or time limit."

    # execution time limit
    agent = _get_agent(max_execution_time=0.0)
    output = agent.run("when was langchain made")
    assert output == "Agent stopped due to iteration limit or time limit."


def test_agent_with_callbacks() -> None:
    """Test react chain with callbacks by setting verbose globally."""
    handler1 = FakeCallbackHandler()
    handler2 = FakeCallbackHandler()

    tool = "Search"
    responses = [
        f"FooBarBaz\nAction: {tool}\nAction Input: misalignment",
        "Oh well\nFinal Answer: curses foiled again",
    ]
    # Only fake LLM gets callbacks for handler2
    fake_llm = FakeListLLM(responses=responses, callbacks=[handler2])
    tools = [
        Tool(
            name="Search",
            func=lambda x: x,
            description="Useful for searching",
        ),
    ]
    agent = initialize_agent(
        tools,
        fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    output = agent.run("when was langchain made", callbacks=[handler1])
    assert output == "curses foiled again"

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


def test_agent_tool_return_direct() -> None:
    """Test agent using tools that return directly."""
    tool = "Search"
    responses = [
        f"FooBarBaz\nAction: {tool}\nAction Input: misalignment",
        "Oh well\nFinal Answer: curses foiled again",
    ]
    fake_llm = FakeListLLM(responses=responses)
    tools = [
        Tool(
            name="Search",
            func=lambda x: x,
            description="Useful for searching",
            return_direct=True,
        ),
    ]
    agent = initialize_agent(
        tools,
        fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    output = agent.run("when was langchain made")
    assert output == "misalignment"


def test_agent_tool_return_direct_in_intermediate_steps() -> None:
    """Test agent using tools that return directly."""
    tool = "Search"
    responses = [
        f"FooBarBaz\nAction: {tool}\nAction Input: misalignment",
        "Oh well\nFinal Answer: curses foiled again",
    ]
    fake_llm = FakeListLLM(responses=responses)
    tools = [
        Tool(
            name="Search",
            func=lambda x: x,
            description="Useful for searching",
            return_direct=True,
        ),
    ]
    agent = initialize_agent(
        tools,
        fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        return_intermediate_steps=True,
    )

    resp = agent("when was langchain made")
    assert isinstance(resp, dict)
    assert resp["output"] == "misalignment"
    assert len(resp["intermediate_steps"]) == 1
    action, _action_intput = resp["intermediate_steps"][0]
    assert action.tool == "Search"


def test_agent_with_new_prefix_suffix() -> None:
    """Test agent initialization kwargs with new prefix and suffix."""
    fake_llm = FakeListLLM(
        responses=["FooBarBaz\nAction: Search\nAction Input: misalignment"]
    )
    tools = [
        Tool(
            name="Search",
            func=lambda x: x,
            description="Useful for searching",
            return_direct=True,
        ),
    ]
    prefix = "FooBarBaz"

    suffix = "Begin now!\nInput: {input}\nThought: {agent_scratchpad}"

    agent = initialize_agent(
        tools=tools,
        llm=fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"prefix": prefix, "suffix": suffix},
    )

    # avoids "BasePromptTemplate" has no attribute "template" error
    assert hasattr(agent.agent.llm_chain.prompt, "template")  # type: ignore
    prompt_str = agent.agent.llm_chain.prompt.template  # type: ignore
    assert prompt_str.startswith(prefix), "Prompt does not start with prefix"
    assert prompt_str.endswith(suffix), "Prompt does not end with suffix"


def test_agent_lookup_tool() -> None:
    """Test agent lookup tool."""
    fake_llm = FakeListLLM(
        responses=["FooBarBaz\nAction: Search\nAction Input: misalignment"]
    )
    tools = [
        Tool(
            name="Search",
            func=lambda x: x,
            description="Useful for searching",
            return_direct=True,
        ),
    ]
    agent = initialize_agent(
        tools=tools,
        llm=fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    assert agent.lookup_tool("Search") == tools[0]


def test_agent_invalid_tool() -> None:
    """Test agent invalid tool and correct suggestions."""
    fake_llm = FakeListLLM(responses=["FooBarBaz\nAction: Foo\nAction Input: Bar"])
    tools = [
        Tool(
            name="Search",
            func=lambda x: x,
            description="Useful for searching",
            return_direct=True,
        ),
    ]
    agent = initialize_agent(
        tools=tools,
        llm=fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        return_intermediate_steps=True,
        max_iterations=1,
    )

    resp = agent("when was langchain made")
    resp["intermediate_steps"][0][1] == "Foo is not a valid tool, try one of [Search]."
