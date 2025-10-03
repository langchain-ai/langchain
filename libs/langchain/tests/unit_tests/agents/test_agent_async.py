"""Unit tests for agents."""

from typing import Any, Optional

from langchain_core.agents import AgentAction, AgentStep
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.utils import add
from langchain_core.tools import Tool
from typing_extensions import override

from langchain_classic.agents import AgentExecutor, AgentType, initialize_agent
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


class FakeListLLM(LLM):
    """Fake LLM for testing that outputs elements of a list."""

    responses: list[str]
    i: int = -1

    @override
    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Increment counter, and then return response in that index."""
        self.i += 1
        print(f"=== Mock Response #{self.i} ===")  # noqa: T201
        print(self.responses[self.i])  # noqa: T201
        return self.responses[self.i]

    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens in text."""
        return len(text.split())

    async def _acall(self, *args: Any, **kwargs: Any) -> str:
        return self._call(*args, **kwargs)

    @property
    def _identifying_params(self) -> dict[str, Any]:
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

    return initialize_agent(
        tools,
        fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        **kwargs,
    )


async def test_agent_bad_action() -> None:
    """Test react chain when bad action given."""
    agent = _get_agent()
    output = await agent.arun("when was langchain made")
    assert output == "curses foiled again"


async def test_agent_stopped_early() -> None:
    """Test react chain when max iterations or max execution time is exceeded."""
    # iteration limit
    agent = _get_agent(max_iterations=0)
    output = await agent.arun("when was langchain made")
    assert output == "Agent stopped due to iteration limit or time limit."

    # execution time limit
    agent = _get_agent(max_execution_time=0.0)
    output = await agent.arun("when was langchain made")
    assert output == "Agent stopped due to iteration limit or time limit."


async def test_agent_with_callbacks() -> None:
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

    output = await agent.arun("when was langchain made", callbacks=[handler1])
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


async def test_agent_stream() -> None:
    """Test react chain with callbacks by setting verbose globally."""
    tool = "Search"
    responses = [
        f"FooBarBaz\nAction: {tool}\nAction Input: misalignment",
        f"FooBarBaz\nAction: {tool}\nAction Input: something else",
        "Oh well\nFinal Answer: curses foiled again",
    ]
    # Only fake LLM gets callbacks for handler2
    fake_llm = FakeListLLM(responses=responses)
    tools = [
        Tool(
            name="Search",
            func=lambda x: f"Results for: {x}",
            description="Useful for searching",
        ),
    ]
    agent = initialize_agent(
        tools,
        fake_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    output = [a async for a in agent.astream("when was langchain made")]
    assert output == [
        {
            "actions": [
                AgentAction(
                    tool="Search",
                    tool_input="misalignment",
                    log="FooBarBaz\nAction: Search\nAction Input: misalignment",
                ),
            ],
            "messages": [
                AIMessage(
                    content="FooBarBaz\nAction: Search\nAction Input: misalignment",
                ),
            ],
        },
        {
            "steps": [
                AgentStep(
                    action=AgentAction(
                        tool="Search",
                        tool_input="misalignment",
                        log="FooBarBaz\nAction: Search\nAction Input: misalignment",
                    ),
                    observation="Results for: misalignment",
                ),
            ],
            "messages": [HumanMessage(content="Results for: misalignment")],
        },
        {
            "actions": [
                AgentAction(
                    tool="Search",
                    tool_input="something else",
                    log="FooBarBaz\nAction: Search\nAction Input: something else",
                ),
            ],
            "messages": [
                AIMessage(
                    content="FooBarBaz\nAction: Search\nAction Input: something else",
                ),
            ],
        },
        {
            "steps": [
                AgentStep(
                    action=AgentAction(
                        tool="Search",
                        tool_input="something else",
                        log="FooBarBaz\nAction: Search\nAction Input: something else",
                    ),
                    observation="Results for: something else",
                ),
            ],
            "messages": [HumanMessage(content="Results for: something else")],
        },
        {
            "output": "curses foiled again",
            "messages": [
                AIMessage(content="Oh well\nFinal Answer: curses foiled again"),
            ],
        },
    ]
    assert add(output) == {
        "actions": [
            AgentAction(
                tool="Search",
                tool_input="misalignment",
                log="FooBarBaz\nAction: Search\nAction Input: misalignment",
            ),
            AgentAction(
                tool="Search",
                tool_input="something else",
                log="FooBarBaz\nAction: Search\nAction Input: something else",
            ),
        ],
        "steps": [
            AgentStep(
                action=AgentAction(
                    tool="Search",
                    tool_input="misalignment",
                    log="FooBarBaz\nAction: Search\nAction Input: misalignment",
                ),
                observation="Results for: misalignment",
            ),
            AgentStep(
                action=AgentAction(
                    tool="Search",
                    tool_input="something else",
                    log="FooBarBaz\nAction: Search\nAction Input: something else",
                ),
                observation="Results for: something else",
            ),
        ],
        "messages": [
            AIMessage(content="FooBarBaz\nAction: Search\nAction Input: misalignment"),
            HumanMessage(content="Results for: misalignment"),
            AIMessage(
                content="FooBarBaz\nAction: Search\nAction Input: something else",
            ),
            HumanMessage(content="Results for: something else"),
            AIMessage(content="Oh well\nFinal Answer: curses foiled again"),
        ],
        "output": "curses foiled again",
    }


async def test_agent_tool_return_direct() -> None:
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

    output = await agent.arun("when was langchain made")
    assert output == "misalignment"


async def test_agent_tool_return_direct_in_intermediate_steps() -> None:
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

    resp = await agent.acall("when was langchain made")
    assert isinstance(resp, dict)
    assert resp["output"] == "misalignment"
    assert len(resp["intermediate_steps"]) == 1
    action, _action_intput = resp["intermediate_steps"][0]
    assert action.tool == "Search"


async def test_agent_invalid_tool() -> None:
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

    resp = await agent.acall("when was langchain made")
    assert (
        resp["intermediate_steps"][0][1]
        == "Foo is not a valid tool, try one of [Search]."
    )
