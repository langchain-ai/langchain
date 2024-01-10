"""Unit tests for agents."""
import json
from itertools import cycle
from typing import Any, Dict, List, Optional, Union, cast

from langchain_core.agents import (
    AgentAction,
    AgentActionMessageLog,
    AgentFinish,
    AgentStep,
)
from langchain_core.language_models.llms import LLM
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    FunctionMessage,
    HumanMessage,
)
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.utils import add
from langchain_core.tools import Tool
from langchain_core.tracers import RunLog, RunLogPatch

from langchain.agents import (
    AgentExecutor,
    AgentType,
    create_openai_functions_agent,
    create_openai_tools_agent,
    initialize_agent,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolAgentAction
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler
from tests.unit_tests.llms.fake_chat_model import GenericFakeChatModel


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


def test_agent_stream() -> None:
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

    output = [a for a in agent.stream("when was langchain made")]
    assert output == [
        {
            "actions": [
                AgentAction(
                    tool="Search",
                    tool_input="misalignment",
                    log="FooBarBaz\nAction: Search\nAction Input: misalignment",
                )
            ],
            "messages": [
                AIMessage(
                    content="FooBarBaz\nAction: Search\nAction Input: misalignment"
                )
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
                )
            ],
            "messages": [HumanMessage(content="Results for: misalignment")],
        },
        {
            "actions": [
                AgentAction(
                    tool="Search",
                    tool_input="something else",
                    log="FooBarBaz\nAction: Search\nAction Input: something else",
                )
            ],
            "messages": [
                AIMessage(
                    content="FooBarBaz\nAction: Search\nAction Input: something else"
                )
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
                )
            ],
            "messages": [HumanMessage(content="Results for: something else")],
        },
        {
            "output": "curses foiled again",
            "messages": [
                AIMessage(content="Oh well\nFinal Answer: curses foiled again")
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
                content="FooBarBaz\nAction: Search\nAction Input: something else"
            ),
            HumanMessage(content="Results for: something else"),
            AIMessage(content="Oh well\nFinal Answer: curses foiled again"),
        ],
        "output": "curses foiled again",
    }


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


async def test_runnable_agent() -> None:
    """Simple test to verify that an agent built with LCEL works."""

    # Will alternate between responding with hello and goodbye
    infinite_cycle = cycle([AIMessage(content="hello world!")])
    # When streaming GenericFakeChatModel breaks AIMessage into chunks based on spaces
    model = GenericFakeChatModel(messages=infinite_cycle)

    template = ChatPromptTemplate.from_messages(
        [("system", "You are Cat Agent 007"), ("human", "{question}")]
    )

    def fake_parse(inputs: dict) -> Union[AgentFinish, AgentAction]:
        """A parser."""
        return AgentFinish(return_values={"foo": "meow"}, log="hard-coded-message")

    agent = template | model | fake_parse
    executor = AgentExecutor(agent=agent, tools=[])

    # Invoke
    result = executor.invoke({"question": "hello"})
    assert result == {"foo": "meow", "question": "hello"}

    # ainvoke
    result = await executor.ainvoke({"question": "hello"})
    assert result == {"foo": "meow", "question": "hello"}

    # Batch
    result = executor.batch(  # type: ignore[assignment]
        [{"question": "hello"}, {"question": "hello"}]
    )
    assert result == [
        {"foo": "meow", "question": "hello"},
        {"foo": "meow", "question": "hello"},
    ]

    # abatch
    result = await executor.abatch(  # type: ignore[assignment]
        [{"question": "hello"}, {"question": "hello"}]
    )
    assert result == [
        {"foo": "meow", "question": "hello"},
        {"foo": "meow", "question": "hello"},
    ]

    # Stream
    results = list(executor.stream({"question": "hello"}))
    assert results == [
        {"foo": "meow", "messages": [AIMessage(content="hard-coded-message")]}
    ]

    # astream
    results = [r async for r in executor.astream({"question": "hello"})]
    assert results == [
        {
            "foo": "meow",
            "messages": [
                AIMessage(content="hard-coded-message"),
            ],
        }
    ]

    # stream log
    results: List[RunLogPatch] = [  # type: ignore[no-redef]
        r async for r in executor.astream_log({"question": "hello"})
    ]
    # # Let's stream just the llm tokens.
    messages = []
    for log_record in results:
        for op in log_record.ops:  # type: ignore[attr-defined]
            if op["op"] == "add" and isinstance(op["value"], AIMessageChunk):
                messages.append(op["value"])

    assert messages != []

    # Aggregate state
    run_log = None

    for result in results:
        if run_log is None:
            run_log = result
        else:
            # `+` is defined for RunLogPatch
            run_log = run_log + result  # type: ignore[union-attr]

    assert isinstance(run_log, RunLog)

    assert run_log.state["final_output"] == {
        "foo": "meow",
        "messages": [AIMessage(content="hard-coded-message")],
    }


async def test_runnable_agent_with_function_calls() -> None:
    """Test agent with intermediate agent actions."""
    # Will alternate between responding with hello and goodbye
    infinite_cycle = cycle(
        [AIMessage(content="looking for pet..."), AIMessage(content="Found Pet")]
    )
    model = GenericFakeChatModel(messages=infinite_cycle)

    template = ChatPromptTemplate.from_messages(
        [("system", "You are Cat Agent 007"), ("human", "{question}")]
    )

    parser_responses = cycle(
        [
            AgentAction(
                tool="find_pet",
                tool_input={
                    "pet": "cat",
                },
                log="find_pet()",
            ),
            AgentFinish(
                return_values={"foo": "meow"},
                log="hard-coded-message",
            ),
        ],
    )

    def fake_parse(inputs: dict) -> Union[AgentFinish, AgentAction]:
        """A parser."""
        return cast(Union[AgentFinish, AgentAction], next(parser_responses))

    @tool
    def find_pet(pet: str) -> str:
        """Find the given pet."""
        if pet != "cat":
            raise ValueError("Only cats allowed")
        return "Spying from under the bed."

    agent = template | model | fake_parse
    executor = AgentExecutor(agent=agent, tools=[find_pet])

    # Invoke
    result = executor.invoke({"question": "hello"})
    assert result == {"foo": "meow", "question": "hello"}

    # ainvoke
    result = await executor.ainvoke({"question": "hello"})
    assert result == {"foo": "meow", "question": "hello"}

    # astream
    results = [r async for r in executor.astream({"question": "hello"})]
    assert results == [
        {
            "actions": [
                AgentAction(
                    tool="find_pet", tool_input={"pet": "cat"}, log="find_pet()"
                )
            ],
            "messages": [AIMessage(content="find_pet()")],
        },
        {
            "messages": [HumanMessage(content="Spying from under the bed.")],
            "steps": [
                AgentStep(
                    action=AgentAction(
                        tool="find_pet", tool_input={"pet": "cat"}, log="find_pet()"
                    ),
                    observation="Spying from under the bed.",
                )
            ],
        },
        {"foo": "meow", "messages": [AIMessage(content="hard-coded-message")]},
    ]

    # astream log

    messages = []
    async for patch in executor.astream_log({"question": "hello"}):
        for op in patch.ops:
            if op["op"] != "add":
                continue

            value = op["value"]

            if not isinstance(value, AIMessageChunk):
                continue

            if value.content == "":  # Then it's a function invocation message
                continue

            messages.append(value.content)

    assert messages == ["looking", " ", "for", " ", "pet...", "Found", " ", "Pet"]


async def test_runnable_with_multi_action_per_step() -> None:
    """Test an agent that can make multiple function calls at once."""
    # Will alternate between responding with hello and goodbye
    infinite_cycle = cycle(
        [AIMessage(content="looking for pet..."), AIMessage(content="Found Pet")]
    )
    model = GenericFakeChatModel(messages=infinite_cycle)

    template = ChatPromptTemplate.from_messages(
        [("system", "You are Cat Agent 007"), ("human", "{question}")]
    )

    parser_responses = cycle(
        [
            [
                AgentAction(
                    tool="find_pet",
                    tool_input={
                        "pet": "cat",
                    },
                    log="find_pet()",
                ),
                AgentAction(
                    tool="pet_pet",  # A function that allows you to pet the given pet.
                    tool_input={
                        "pet": "cat",
                    },
                    log="pet_pet()",
                ),
            ],
            AgentFinish(
                return_values={"foo": "meow"},
                log="hard-coded-message",
            ),
        ],
    )

    def fake_parse(inputs: dict) -> Union[AgentFinish, AgentAction]:
        """A parser."""
        return cast(Union[AgentFinish, AgentAction], next(parser_responses))

    @tool
    def find_pet(pet: str) -> str:
        """Find the given pet."""
        if pet != "cat":
            raise ValueError("Only cats allowed")
        return "Spying from under the bed."

    @tool
    def pet_pet(pet: str) -> str:
        """Pet the given pet."""
        if pet != "cat":
            raise ValueError("Only cats should be petted.")
        return "purrrr"

    agent = template | model | fake_parse
    executor = AgentExecutor(agent=agent, tools=[find_pet])

    # Invoke
    result = executor.invoke({"question": "hello"})
    assert result == {"foo": "meow", "question": "hello"}

    # ainvoke
    result = await executor.ainvoke({"question": "hello"})
    assert result == {"foo": "meow", "question": "hello"}

    # astream
    results = [r async for r in executor.astream({"question": "hello"})]
    assert results == [
        {
            "actions": [
                AgentAction(
                    tool="find_pet", tool_input={"pet": "cat"}, log="find_pet()"
                )
            ],
            "messages": [AIMessage(content="find_pet()")],
        },
        {
            "actions": [
                AgentAction(tool="pet_pet", tool_input={"pet": "cat"}, log="pet_pet()")
            ],
            "messages": [AIMessage(content="pet_pet()")],
        },
        {
            # By-default observation gets converted into human message.
            "messages": [HumanMessage(content="Spying from under the bed.")],
            "steps": [
                AgentStep(
                    action=AgentAction(
                        tool="find_pet", tool_input={"pet": "cat"}, log="find_pet()"
                    ),
                    observation="Spying from under the bed.",
                )
            ],
        },
        {
            "messages": [
                HumanMessage(
                    content="pet_pet is not a valid tool, try one of [find_pet]."
                )
            ],
            "steps": [
                AgentStep(
                    action=AgentAction(
                        tool="pet_pet", tool_input={"pet": "cat"}, log="pet_pet()"
                    ),
                    observation="pet_pet is not a valid tool, try one of [find_pet].",
                )
            ],
        },
        {"foo": "meow", "messages": [AIMessage(content="hard-coded-message")]},
    ]

    # astream log

    messages = []
    async for patch in executor.astream_log({"question": "hello"}):
        for op in patch.ops:
            if op["op"] != "add":
                continue

            value = op["value"]

            if not isinstance(value, AIMessageChunk):
                continue

            if value.content == "":  # Then it's a function invocation message
                continue

            messages.append(value.content)

    assert messages == ["looking", " ", "for", " ", "pet...", "Found", " ", "Pet"]


def _make_func_invocation(name: str, **kwargs: Any) -> AIMessage:
    """Create an AIMessage that represents a function invocation.

    Args:
        name: Name of the function to invoke.
        kwargs: Keyword arguments to pass to the function.

    Returns:
        AIMessage that represents a request to invoke a function.
    """
    return AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": name,
                "arguments": json.dumps(kwargs),
            }
        },
    )


async def test_openai_agent_with_streaming() -> None:
    """Test openai agent with streaming."""
    infinite_cycle = cycle(
        [
            _make_func_invocation("find_pet", pet="cat"),
            AIMessage(content="The cat is spying from under the bed."),
        ]
    )

    model = GenericFakeChatModel(messages=infinite_cycle)

    @tool
    def find_pet(pet: str) -> str:
        """Find the given pet."""
        if pet != "cat":
            raise ValueError("Only cats allowed")
        return "Spying from under the bed."

    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI bot. Your name is kitty power meow."),
            ("human", "{question}"),
            MessagesPlaceholder(
                variable_name="agent_scratchpad",
            ),
        ]
    )

    # type error due to base tool type below -- would need to be adjusted on tool
    # decorator.
    agent = create_openai_functions_agent(
        model,
        [find_pet],  # type: ignore[list-item]
        template,
    )
    executor = AgentExecutor(agent=agent, tools=[find_pet])

    # Invoke
    result = executor.invoke({"question": "hello"})
    assert result == {
        "output": "The cat is spying from under the bed.",
        "question": "hello",
    }

    # astream
    chunks = [chunk async for chunk in executor.astream({"question": "hello"})]
    assert chunks == [
        {
            "actions": [
                AgentActionMessageLog(
                    tool="find_pet",
                    tool_input={"pet": "cat"},
                    log="\nInvoking: `find_pet` with `{'pet': 'cat'}`\n\n\n",
                    message_log=[
                        AIMessageChunk(
                            content="",
                            additional_kwargs={
                                "function_call": {
                                    "name": "find_pet",
                                    "arguments": '{"pet": "cat"}',
                                }
                            },
                        )
                    ],
                )
            ],
            "messages": [
                AIMessageChunk(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": "find_pet",
                            "arguments": '{"pet": "cat"}',
                        }
                    },
                )
            ],
        },
        {
            "messages": [
                FunctionMessage(content="Spying from under the bed.", name="find_pet")
            ],
            "steps": [
                AgentStep(
                    action=AgentActionMessageLog(
                        tool="find_pet",
                        tool_input={"pet": "cat"},
                        log="\nInvoking: `find_pet` with `{'pet': 'cat'}`\n\n\n",
                        message_log=[
                            AIMessageChunk(
                                content="",
                                additional_kwargs={
                                    "function_call": {
                                        "name": "find_pet",
                                        "arguments": '{"pet": "cat"}',
                                    }
                                },
                            )
                        ],
                    ),
                    observation="Spying from under the bed.",
                )
            ],
        },
        {
            "messages": [AIMessage(content="The cat is spying from under the bed.")],
            "output": "The cat is spying from under the bed.",
        },
    ]
    #
    # # astream_log
    log_patches = [
        log_patch async for log_patch in executor.astream_log({"question": "hello"})
    ]

    messages = []

    for log_patch in log_patches:
        for op in log_patch.ops:
            if op["op"] == "add" and isinstance(op["value"], AIMessageChunk):
                value = op["value"]
                if value.content:  # Filter out function call messages
                    messages.append(value.content)

    assert messages == [
        "The",
        " ",
        "cat",
        " ",
        "is",
        " ",
        "spying",
        " ",
        "from",
        " ",
        "under",
        " ",
        "the",
        " ",
        "bed.",
    ]


def _make_tools_invocation(name_to_arguments: Dict[str, Dict[str, Any]]) -> AIMessage:
    """Create an AIMessage that represents a tools invocation.

    Args:
        name_to_arguments: A dictionary mapping tool names to an invocation.

    Returns:
        AIMessage that represents a request to invoke a tool.
    """
    tool_calls = [
        {"function": {"name": name, "arguments": json.dumps(arguments)}, "id": idx}
        for idx, (name, arguments) in enumerate(name_to_arguments.items())
    ]

    return AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": tool_calls,
        },
    )


async def test_openai_agent_tools_agent() -> None:
    """Test OpenAI tools agent."""
    infinite_cycle = cycle(
        [
            _make_tools_invocation(
                {
                    "find_pet": {"pet": "cat"},
                    "check_time": {},
                }
            ),
            AIMessage(content="The cat is spying from under the bed."),
        ]
    )

    model = GenericFakeChatModel(messages=infinite_cycle)

    @tool
    def find_pet(pet: str) -> str:
        """Find the given pet."""
        if pet != "cat":
            raise ValueError("Only cats allowed")
        return "Spying from under the bed."

    @tool
    def check_time() -> str:
        """Find the given pet."""
        return "It's time to pet the cat."

    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI bot. Your name is kitty power meow."),
            ("human", "{question}"),
            MessagesPlaceholder(
                variable_name="agent_scratchpad",
            ),
        ]
    )

    # type error due to base tool type below -- would need to be adjusted on tool
    # decorator.
    agent = create_openai_tools_agent(
        model,
        [find_pet],  # type: ignore[list-item]
        template,
    )
    executor = AgentExecutor(agent=agent, tools=[find_pet])

    # Invoke
    result = executor.invoke({"question": "hello"})
    assert result == {
        "output": "The cat is spying from under the bed.",
        "question": "hello",
    }

    # astream
    chunks = [chunk async for chunk in executor.astream({"question": "hello"})]
    assert chunks == [
        {
            "actions": [
                OpenAIToolAgentAction(
                    tool="find_pet",
                    tool_input={"pet": "cat"},
                    log="\nInvoking: `find_pet` with `{'pet': 'cat'}`\n\n\n",
                    message_log=[
                        AIMessageChunk(
                            content="",
                            additional_kwargs={
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "find_pet",
                                            "arguments": '{"pet": "cat"}',
                                        },
                                        "id": 0,
                                    },
                                    {
                                        "function": {
                                            "name": "check_time",
                                            "arguments": "{}",
                                        },
                                        "id": 1,
                                    },
                                ]
                            },
                        )
                    ],
                    tool_call_id="0",
                )
            ],
            "messages": [
                AIMessageChunk(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "find_pet",
                                    "arguments": '{"pet": "cat"}',
                                },
                                "id": 0,
                            },
                            {
                                "function": {"name": "check_time", "arguments": "{}"},
                                "id": 1,
                            },
                        ]
                    },
                )
            ],
        },
        {
            "actions": [
                OpenAIToolAgentAction(
                    tool="check_time",
                    tool_input={},
                    log="\nInvoking: `check_time` with `{}`\n\n\n",
                    message_log=[
                        AIMessageChunk(
                            content="",
                            additional_kwargs={
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "find_pet",
                                            "arguments": '{"pet": "cat"}',
                                        },
                                        "id": 0,
                                    },
                                    {
                                        "function": {
                                            "name": "check_time",
                                            "arguments": "{}",
                                        },
                                        "id": 1,
                                    },
                                ]
                            },
                        )
                    ],
                    tool_call_id="1",
                )
            ],
            "messages": [
                AIMessageChunk(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "find_pet",
                                    "arguments": '{"pet": "cat"}',
                                },
                                "id": 0,
                            },
                            {
                                "function": {"name": "check_time", "arguments": "{}"},
                                "id": 1,
                            },
                        ]
                    },
                )
            ],
        },
        {
            "messages": [
                FunctionMessage(content="Spying from under the bed.", name="find_pet")
            ],
            "steps": [
                AgentStep(
                    action=OpenAIToolAgentAction(
                        tool="find_pet",
                        tool_input={"pet": "cat"},
                        log="\nInvoking: `find_pet` with `{'pet': 'cat'}`\n\n\n",
                        message_log=[
                            AIMessageChunk(
                                content="",
                                additional_kwargs={
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "find_pet",
                                                "arguments": '{"pet": "cat"}',
                                            },
                                            "id": 0,
                                        },
                                        {
                                            "function": {
                                                "name": "check_time",
                                                "arguments": "{}",
                                            },
                                            "id": 1,
                                        },
                                    ]
                                },
                            )
                        ],
                        tool_call_id="0",
                    ),
                    observation="Spying from under the bed.",
                )
            ],
        },
        {
            "messages": [
                FunctionMessage(
                    content="check_time is not a valid tool, try one of [find_pet].",
                    name="check_time",
                )
            ],
            "steps": [
                AgentStep(
                    action=OpenAIToolAgentAction(
                        tool="check_time",
                        tool_input={},
                        log="\nInvoking: `check_time` with `{}`\n\n\n",
                        message_log=[
                            AIMessageChunk(
                                content="",
                                additional_kwargs={
                                    "tool_calls": [
                                        {
                                            "function": {
                                                "name": "find_pet",
                                                "arguments": '{"pet": "cat"}',
                                            },
                                            "id": 0,
                                        },
                                        {
                                            "function": {
                                                "name": "check_time",
                                                "arguments": "{}",
                                            },
                                            "id": 1,
                                        },
                                    ]
                                },
                            )
                        ],
                        tool_call_id="1",
                    ),
                    observation="check_time is not a valid tool, "
                    "try one of [find_pet].",
                )
            ],
        },
        {
            "messages": [AIMessage(content="The cat is spying from under the bed.")],
            "output": "The cat is spying from under the bed.",
        },
    ]

    # astream_log
    log_patches = [
        log_patch async for log_patch in executor.astream_log({"question": "hello"})
    ]

    # Get the tokens from the astream log response.
    messages = []

    for log_patch in log_patches:
        for op in log_patch.ops:
            if op["op"] == "add" and isinstance(op["value"], AIMessageChunk):
                value = op["value"]
                if value.content:  # Filter out function call messages
                    messages.append(value.content)

    assert messages == [
        "The",
        " ",
        "cat",
        " ",
        "is",
        " ",
        "spying",
        " ",
        "from",
        " ",
        "under",
        " ",
        "the",
        " ",
        "bed.",
    ]
